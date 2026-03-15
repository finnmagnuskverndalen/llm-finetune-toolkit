"""
Interactive streaming chat for base, fine-tuned, and merged models.

Fixes:
  - Dynamic model name in header (no more hardcoded "QWEN")
  - GPU cleanup on startup
  - Graceful Ctrl+C handling
  - Config-driven inference params
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import torch
import time
from pathlib import Path
from transformers import TextIteratorStreamer
from threading import Thread
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.prompt import Prompt
from rich.rule import Rule
from rich.align import Align

from utils import load_config, get_model_short_name, cleanup_gpu, load_model_for_inference

console = Console()
CFG = load_config()

BASE_MODEL_ID = CFG["model"]["name"]
FINETUNED_DIR = CFG["model"]["output_dir"]
MERGED_DIR = CFG["model"]["merged_dir"]
CHAT_CFG = CFG.get("chat", {})
SYSTEM_PROMPT = CFG.get("system_prompt", "You are a helpful assistant.")
MODEL_SHORT = get_model_short_name(CFG)


def get_sys_info():
    import psutil
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
    title.append(f"  {MODEL_SHORT}  ", style="bold bright_cyan")
    title.append(mode_label, style="bold yellow")
    console.print(Panel(
        Align.center(title),
        border_style="bright_cyan",
        padding=(1, 4),
    ))
    console.print(Align.center(
        Text(
            "type 'quit' to exit  •  'reset' to clear  •  'switch' to toggle model  •  'help' for commands",
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


def print_help():
    console.print(Panel(
        "[bold]Commands:[/bold]\n"
        "  [cyan]quit[/cyan]     — exit chat\n"
        "  [cyan]reset[/cyan]    — clear conversation history\n"
        "  [cyan]switch[/cyan]   — cycle between finetuned / base / merged\n"
        "  [cyan]help[/cyan]     — show this message\n"
        "  [cyan]status[/cyan]   — show GPU/RAM usage\n"
        "  [cyan]Ctrl+C[/cyan]   — stop current generation",
        title="[cyan]Help[/cyan]",
        border_style="cyan",
        width=50,
    ))


def load_model(mode="finetuned"):
    console.print(Rule(f"[bold cyan]Loading model ({mode})[/bold cyan]"))

    with console.status("[cyan]Loading...[/cyan]"):
        model, tokenizer, actual_mode = load_model_for_inference(CFG, mode, console)

    device = "GPU" if torch.cuda.is_available() else "CPU"
    console.print(f"[green]✓[/green] {MODEL_SHORT} loaded ({actual_mode}) on {device}")
    time.sleep(0.3)
    return model, tokenizer, actual_mode


def build_messages(history, max_turns=None):
    if max_turns is None:
        max_turns = CHAT_CFG.get("max_history_turns", 6)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    recent = history[-max_turns:] if max_turns > 0 else history
    for turn in recent:
        messages.append({"role": "user", "content": turn["user"]})
        if turn["assistant"]:
            messages.append({"role": "assistant", "content": turn["assistant"]})
    return messages


def generate_response(model, tokenizer, history):
    messages = build_messages(history)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
    model_label = f"[bright_green]{MODEL_SHORT}[/bright_green]"
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
            console.print(Panel("[bold green]Goodbye![/bold green]", border_style="green"))
            break

        if cmd == "reset":
            history = []
            print_header(mode_labels.get(mode, mode))
            console.print(Panel("[yellow]Conversation reset[/yellow]", border_style="yellow", width=30))
            continue

        if cmd == "help":
            print_help()
            continue

        if cmd == "status":
            console.print(Panel(get_sys_info(), title="[cyan]System[/cyan]", border_style="cyan", width=50))
            continue

        if cmd == "switch":
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
        history.append({"user": user_input, "assistant": ""})

        console.print()
        streamer, thread = generate_response(model, tokenizer, history)

        response_text = ""
        interrupted = False
        try:
            with Live(
                Panel(Text("", style="bright_green"), title=model_label, border_style="bright_green", padding=(0, 2)),
                refresh_per_second=10,
                console=console,
            ) as live:
                for token in streamer:
                    response_text += token
                    live.update(Panel(
                        Text(response_text, style="bright_green"),
                        title=model_label,
                        border_style="bright_green",
                        padding=(0, 2),
                    ))
        except KeyboardInterrupt:
            interrupted = True
            console.print("\n[dim]Generation stopped[/dim]")

        thread.join(timeout=2)
        history[-1]["assistant"] = response_text

        if interrupted:
            console.print()
        console.print()


if __name__ == "__main__":
    # GPU cleanup
    cleanup_gpu(console)

    mode = "finetuned"
    if "--base" in sys.argv:
        mode = "base"
    elif "--merged" in sys.argv:
        mode = "merged"

    try:
        model, tokenizer, mode = load_model(mode)
        chat(model, tokenizer, mode)
    except torch.cuda.OutOfMemoryError:
        console.print("\n[bold red]Out of GPU memory![/bold red]")
        console.print("[yellow]Try:[/yellow]")
        console.print("  [cyan]1. Kill other GPU processes: nvidia-smi[/cyan]")
        console.print("  [cyan]2. Use a smaller model in config.yaml[/cyan]")
    except KeyboardInterrupt:
        console.print("\n[dim]Bye![/dim]")