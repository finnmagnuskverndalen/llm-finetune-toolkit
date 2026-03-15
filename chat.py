"""
Chat interface for base and fine-tuned models.

Fixes over the original:
  1. Dtype consistency — uses same compute_dtype detection as training
  2. Sliding window for history — prevents context overflow
  3. Clean history management — no empty assistant turns
  4. Repetition penalty — reduces degenerate loops
  5. Config-driven — reads from config.yaml
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import yaml
import torch
import time
import psutil
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from peft import PeftModel
from threading import Thread
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.prompt import Prompt
from rich.rule import Rule
from rich.align import Align
from rich import box

console = Console()

# ── Load Config ──
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

BASE_MODEL_ID = CFG["model"]["name"]
FINETUNED_DIR = CFG["model"]["output_dir"]
MERGED_DIR = CFG["model"]["merged_dir"]
CHAT_CFG = CFG.get("chat", {})
SYSTEM_PROMPT = CFG.get("system_prompt", "You are a helpful assistant.")


def get_sys_info():
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory()
    ram_used = ram.used / (1024**3)
    ram_total = ram.total / (1024**3)
    gpu_str = ""
    try:
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / (1024**3)
            gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_str = f"  GPU: {gpu_mem:.1f}/{gpu_total:.1f}GB"
    except Exception:
        pass
    return f"CPU: {cpu:.0f}%  RAM: {ram_used:.1f}/{ram_total:.1f}GB{gpu_str}"


def print_header(mode_label):
    console.clear()
    title = Text()
    title.append("🤖  QWEN CHAT  ", style="bold bright_cyan")
    title.append(mode_label, style="bold yellow")
    console.print(Panel(
        Align.center(title),
        border_style="bright_cyan",
        padding=(1, 4),
    ))
    console.print(Align.center(
        Text(
            "type 'quit' to exit  •  'reset' to clear history  •  'switch' to toggle model",
            style="dim",
        )
    ))
    console.print()


def print_user_bubble(text):
    console.print()
    console.print(Align.right(
        Panel(
            Text(text, style="bold white"),
            title="[bright_blue]You[/bright_blue]",
            border_style="bright_blue",
            width=min(len(text) + 6, console.width - 10),
            padding=(0, 2),
        )
    ))


def get_compute_dtype():
    """Match training dtype exactly."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    elif torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_model(mode="finetuned"):
    """
    Load model in one of three modes:
      - 'finetuned': base + LoRA adapters
      - 'merged':    fully merged model (if available)
      - 'base':      base model only
    """
    console.print(Rule(f"[bold cyan]Loading Model ({mode})[/bold cyan]"))

    compute_dtype = get_compute_dtype()

    with console.status("[cyan]Loading tokenizer...[/cyan]"):
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    console.print("[green]✓[/green] Tokenizer loaded")

    # Try merged model first (fastest)
    if mode == "merged" and Path(MERGED_DIR).exists():
        with console.status("[cyan]Loading merged model...[/cyan]"):
            model = AutoModelForCausalLM.from_pretrained(
                MERGED_DIR,
                torch_dtype=compute_dtype,
                device_map="auto",
                trust_remote_code=True,
            )
        console.print("[green]✓[/green] Merged model loaded")
        actual_mode = "merged"

    else:
        # Load quantized base
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,  # FIXED: matches training dtype
        )

        with console.status("[cyan]Loading base model (4-bit)...[/cyan]"):
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_ID,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        console.print("[green]✓[/green] Base model loaded")

        actual_mode = "base"
        if mode == "finetuned":
            try:
                with console.status("[cyan]Loading LoRA adapters...[/cyan]"):
                    model = PeftModel.from_pretrained(model, FINETUNED_DIR)
                console.print("[green]✓[/green] Fine-tuned adapters loaded")
                actual_mode = "finetuned"
            except Exception as e:
                console.print(f"[yellow]⚠ Could not load adapters: {e}[/yellow]")
                console.print("[yellow]  Falling back to base model[/yellow]")

    model.eval()

    device = "GPU 🟢" if torch.cuda.is_available() else "CPU 🔴"
    console.print(f"[dim]Device: {device} | dtype: {compute_dtype}[/dim]")
    time.sleep(0.5)
    return model, tokenizer, actual_mode


def build_messages(history, max_turns=None):
    """Build message list from history with sliding window."""
    if max_turns is None:
        max_turns = CHAT_CFG.get("max_history_turns", 6)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Keep only the last N turns to prevent context overflow
    recent = history[-max_turns:] if max_turns > 0 else history

    for turn in recent:
        messages.append({"role": "user", "content": turn["user"]})
        if turn["assistant"]:
            messages.append({"role": "assistant", "content": turn["assistant"]})

    return messages


def generate_response(model, tokenizer, history):
    """Generate streamed response."""
    messages = build_messages(history)

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=CHAT_CFG.get("max_new_tokens", 512),
        temperature=CHAT_CFG.get("temperature", 0.7),
        top_p=CHAT_CFG.get("top_p", 0.9),
        repetition_penalty=CHAT_CFG.get("repetition_penalty", 1.1),
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()
    return streamer, thread


def chat(model, tokenizer, mode):
    history = []
    mode_labels = {
        "finetuned": "finetuned",
        "merged": "merged (exported)",
        "base": "base model",
    }
    print_header(mode_labels.get(mode, mode))

    while True:
        sys_info = get_sys_info()
        console.print(Rule(f"[dim]{sys_info}[/dim]"))

        try:
            user_input = Prompt.ask("[bold bright_blue]You[/bold bright_blue]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye![/dim]")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd == "quit":
            console.print(Panel("[bold green]Goodbye! 👋[/bold green]", border_style="green"))
            break

        if cmd == "reset":
            history = []
            print_header(mode_labels.get(mode, mode))
            console.print(Panel("[yellow]Conversation reset[/yellow]", border_style="yellow", width=30))
            continue

        if cmd == "switch":
            # Cycle: finetuned -> base -> merged -> finetuned
            next_mode = {
                "finetuned": "base",
                "base": "merged" if Path(MERGED_DIR).exists() else "finetuned",
                "merged": "finetuned",
            }
            mode = next_mode.get(mode, "finetuned")
            model, tokenizer, mode = load_model(mode)
            history = []
            print_header(mode_labels.get(mode, mode))
            continue

        print_user_bubble(user_input)

        # Build history for generation (current turn has no response yet)
        history.append({"user": user_input, "assistant": ""})

        console.print()
        streamer, thread = generate_response(model, tokenizer, history)

        response_text = ""
        with Live(
            Panel(
                Text("", style="bright_green"),
                title="[bright_green]Qwen[/bright_green]",
                border_style="bright_green",
                padding=(0, 2),
            ),
            refresh_per_second=10,
            console=console,
        ) as live:
            for token in streamer:
                response_text += token
                live.update(Panel(
                    Text(response_text, style="bright_green"),
                    title="[bright_green]Qwen[/bright_green]",
                    border_style="bright_green",
                    padding=(0, 2),
                ))
        thread.join()

        # Update history with actual response
        history[-1]["assistant"] = response_text
        console.print()


if __name__ == "__main__":
    mode = "finetuned"
    if "--base" in sys.argv:
        mode = "base"
    elif "--merged" in sys.argv:
        mode = "merged"

    model, tokenizer, mode = load_model(mode)
    chat(model, tokenizer, mode)