"""
Shared utilities used by all scripts.

Centralizes: config loading, model loading, dtype detection, GPU cleanup,
data formatting, and common helpers.
"""

import os
import yaml
import torch
import subprocess
import signal
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config():
    """Load and return the config dict."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_model_short_name(cfg=None):
    """Extract short display name from model ID. E.g., 'Qwen2.5-0.5B-Instruct'."""
    if cfg is None:
        cfg = load_config()
    return cfg["model"]["name"].split("/")[-1]


def get_compute_dtype():
    """Detect best compute dtype. Consistent across train and inference."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    elif torch.cuda.is_available():
        return torch.float16
    return torch.float32


def cleanup_gpu(console=None):
    """
    Check for stale GPU processes and offer to kill them.
    Returns True if GPU is clean, False if user declined cleanup.
    """
    if not torch.cuda.is_available():
        return True

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory,name", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return True

        # Parse processes
        current_pid = os.getpid()
        stale = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                pid = int(parts[0])
                mem = parts[1]
                name = parts[2]
                if pid != current_pid:
                    stale.append((pid, mem, name))

        if not stale:
            return True

        if console:
            from rich.table import Table
            from rich import box
            console.print()
            table = Table(box=box.ROUNDED, title="GPU processes found", style="yellow")
            table.add_column("PID", style="bold")
            table.add_column("Memory (MB)")
            table.add_column("Process")
            for pid, mem, name in stale:
                table.add_row(str(pid), mem, name)
            console.print(table)

            try:
                answer = console.input("[yellow]Kill these processes to free GPU memory? (y/N): [/yellow]").strip().lower()
            except (KeyboardInterrupt, EOFError):
                return False

            if answer in ("y", "yes"):
                for pid, _, _ in stale:
                    try:
                        os.kill(pid, signal.SIGTERM)
                        console.print(f"  [green]Killed PID {pid}[/green]")
                    except PermissionError:
                        console.print(f"  [red]Cannot kill PID {pid} — try: sudo kill {pid}[/red]")
                    except ProcessLookupError:
                        pass
                import time
                time.sleep(1)
                torch.cuda.empty_cache()
                console.print("[green]GPU memory cleared[/green]\n")
                return True
            else:
                console.print("[dim]Skipping cleanup — may run out of memory[/dim]\n")
                return True
        return True

    except Exception:
        return True


def load_model_for_inference(cfg, mode="finetuned", console=None):
    """
    Load model for inference (chat, benchmark).
    Returns (model, tokenizer, actual_mode).
    """
    compute_dtype = get_compute_dtype()
    base_id = cfg["model"]["name"]
    finetuned_dir = cfg["model"]["output_dir"]
    merged_dir = cfg["model"]["merged_dir"]

    tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if mode == "merged" and Path(merged_dir).exists():
        model = AutoModelForCausalLM.from_pretrained(
            merged_dir,
            torch_dtype=compute_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        actual_mode = "merged"
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        actual_mode = "base"

        if mode == "finetuned":
            try:
                model = PeftModel.from_pretrained(model, finetuned_dir)
                actual_mode = "finetuned"
            except Exception as e:
                if console:
                    console.print(f"[yellow]Could not load adapters: {e}[/yellow]")
                    console.print("[yellow]  Falling back to base model[/yellow]")

    model.eval()
    return model, tokenizer, actual_mode


# ── Data Formatting ──────────────────────────────────────────
# Single source of truth — used by finetune.py and validate.py

def format_guanaco(example, system_prompt):
    """Parse ### Human: / ### Assistant: multi-turn format."""
    text = example.get("text", "")
    if "### Human:" not in text:
        return {"messages": None}
    turns = text.split("### Human:")[1:]
    messages = [{"role": "system", "content": system_prompt}]
    for turn in turns:
        if "### Assistant:" not in turn:
            continue
        parts = turn.split("### Assistant:", 1)
        user_content = parts[0].strip()
        assistant_content = parts[1].strip()
        if user_content and assistant_content:
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": assistant_content})
    if len(messages) <= 1:
        return {"messages": None}
    return {"messages": messages}


def format_alpaca(example, system_prompt):
    """Parse instruction / input / output format."""
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    output = example.get("output", "")
    if not instruction or not output:
        return {"messages": None}
    user_msg = f"{instruction}\n{inp}".strip() if inp else instruction
    return {"messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": output},
    ]}


def format_messages(example, system_prompt):
    """Handle datasets that already have a 'messages' column."""
    msgs = example.get("messages", None)
    if msgs and isinstance(msgs, list) and len(msgs) >= 2:
        if msgs[0].get("role") != "system":
            msgs = [{"role": "system", "content": system_prompt}] + msgs
        return {"messages": msgs}
    return {"messages": None}


def detect_and_format(example, system_prompt):
    """Auto-detect dataset format and convert to messages."""
    if "messages" in example and example["messages"]:
        return format_messages(example, system_prompt)
    if "text" in example and "### Human:" in str(example.get("text", "")):
        return format_guanaco(example, system_prompt)
    if "instruction" in example and "output" in example:
        return format_alpaca(example, system_prompt)
    return {"messages": None}