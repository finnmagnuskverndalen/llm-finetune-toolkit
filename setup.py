"""
Interactive setup wizard for LLM fine-tuning.

Detects your hardware, asks a few questions, and generates a safe config.yaml
tuned for your GPU. Prevents the OOM crashes and bad hyperparameter choices
that plague first-time setups.

Usage:
    python3 setup.py            # Interactive wizard
    python3 setup.py --auto     # Auto-detect everything, no questions
"""

import argparse
import yaml
import torch
import shutil
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from rich.prompt import Prompt, IntPrompt, Confirm
from rich import box

console = Console()
CONFIG_PATH = Path(__file__).parent / "config.yaml"

# ── Hardware detection ──

def detect_gpu():
    """Detect GPU and return specs."""
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "vram_gb": round(props.total_memory / (1024**3), 1),
        "bf16": torch.cuda.is_bf16_supported(),
    }


def detect_ram():
    """Detect system RAM in GB."""
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        return None


# ── Model recommendations based on VRAM ──

MODELS = [
    {"name": "Qwen/Qwen2.5-0.5B-Instruct",          "params": "0.5B", "min_vram": 1.5, "vocab": "151K"},
    {"name": "meta-llama/Llama-3.2-1B-Instruct",     "params": "1B",   "min_vram": 2.0, "vocab": "128K"},
    {"name": "HuggingFaceTB/SmolLM2-1.7B-Instruct",  "params": "1.7B", "min_vram": 2.5, "vocab": "49K"},
    {"name": "Qwen/Qwen2.5-1.5B-Instruct",           "params": "1.5B", "min_vram": 3.0, "vocab": "151K"},
    {"name": "google/gemma-2-2b-it",                  "params": "2B",   "min_vram": 3.0, "vocab": "256K"},
    {"name": "Qwen/Qwen2.5-3B-Instruct",             "params": "3B",   "min_vram": 5.0, "vocab": "151K"},
    {"name": "meta-llama/Llama-3.2-3B-Instruct",     "params": "3B",   "min_vram": 5.0, "vocab": "128K"},
    {"name": "microsoft/Phi-3.5-mini-instruct",       "params": "3.8B", "min_vram": 6.0, "vocab": "32K"},
]


def get_safe_config(vram_gb, model_name=None):
    """Generate safe training parameters based on available VRAM."""
    # Find compatible models
    compatible = [m for m in MODELS if m["min_vram"] <= vram_gb]

    if not compatible:
        compatible = [MODELS[0]]  # Fallback to smallest

    # Default to largest compatible model, or user's choice
    if model_name:
        chosen = next((m for m in MODELS if m["name"] == model_name), compatible[-1])
    else:
        chosen = compatible[-1]

    # Scale parameters based on VRAM
    if vram_gb <= 2:
        max_length = 256
        batch_size = 1
        grad_accum = 16
    elif vram_gb <= 4:
        max_length = 512
        batch_size = 1
        grad_accum = 16
    elif vram_gb <= 8:
        max_length = 1024
        batch_size = 2
        grad_accum = 8
    elif vram_gb <= 16:
        max_length = 1024
        batch_size = 4
        grad_accum = 4
    else:
        max_length = 2048
        batch_size = 4
        grad_accum = 4

    return {
        "model": chosen,
        "compatible_models": compatible,
        "max_length": max_length,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
    }


def run_wizard(auto=False):
    console.print(Panel(
        "[bold]LLM Fine-tuning Setup Wizard[/bold]\n"
        "[dim]Detects your hardware and generates a safe config.yaml[/dim]",
        border_style="cyan",
        padding=(1, 4),
    ))

    # ── Detect hardware ──
    console.print(Rule("[bold cyan]Hardware detection[/bold cyan]"))

    gpu = detect_gpu()
    ram = detect_ram()

    if gpu:
        console.print(f"  GPU:  [green]{gpu['name']}[/green] ({gpu['vram_gb']} GB VRAM)")
        console.print(f"  BF16: {'[green]supported[/green]' if gpu['bf16'] else '[yellow]not supported (using FP16)[/yellow]'}")
        vram = gpu["vram_gb"]
    else:
        console.print("  GPU:  [red]not detected[/red] — training on CPU (very slow)")
        vram = 0

    if ram:
        console.print(f"  RAM:  {ram} GB")

    # ── Get safe defaults ──
    safe = get_safe_config(vram)

    # ── Show compatible models ──
    console.print()
    console.print(Rule("[bold cyan]Compatible models[/bold cyan]"))

    model_table = Table(box=box.ROUNDED, style="cyan")
    model_table.add_column("#", style="dim", width=3)
    model_table.add_column("Model", style="bold white")
    model_table.add_column("Params", justify="center")
    model_table.add_column("Vocab", justify="center")
    model_table.add_column("Min VRAM", justify="center")
    model_table.add_column("Fits?", justify="center")

    for i, m in enumerate(MODELS):
        fits = m["min_vram"] <= vram if vram > 0 else True
        fits_str = "[green]yes[/green]" if fits else "[red]no[/red]"
        style = "" if fits else "dim"
        model_table.add_row(
            str(i + 1), m["name"], m["params"], m["vocab"],
            f"{m['min_vram']} GB", fits_str,
            style=style,
        )

    console.print(model_table)

    # ── Choose model ──
    if auto:
        chosen = safe["model"]
        console.print(f"\n  Auto-selected: [green]{chosen['name']}[/green]")
    else:
        default_idx = MODELS.index(safe["model"]) + 1
        console.print(f"\n  Recommended: [green]{safe['model']['name']}[/green] (#{default_idx})")

        try:
            choice = IntPrompt.ask(
                "  Choose model number",
                default=default_idx,
            )
            if 1 <= choice <= len(MODELS):
                chosen = MODELS[choice - 1]
                if vram > 0 and chosen["min_vram"] > vram:
                    console.print(f"[yellow]  Warning: {chosen['name']} needs {chosen['min_vram']}GB, you have {vram}GB[/yellow]")
                    if not Confirm.ask("  Continue anyway?", default=False):
                        chosen = safe["model"]
            else:
                chosen = safe["model"]
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Using default[/dim]")
            chosen = safe["model"]

    # Recalculate safe params for the chosen model
    safe = get_safe_config(vram, chosen["name"])

    # ── Training config ──
    console.print()
    console.print(Rule("[bold cyan]Training settings[/bold cyan]"))

    if auto:
        max_steps = 50
        num_epochs = 2
        console.print(f"  Max steps: {max_steps} (test run)")
        console.print(f"  Batch size: {safe['batch_size']}")
        console.print(f"  Max length: {safe['max_length']} tokens")
    else:
        console.print(f"  Recommended for your GPU: batch_size={safe['batch_size']}, max_length={safe['max_length']}")
        console.print()

        try:
            max_steps = IntPrompt.ask(
                "  Training steps (-1 for full epochs, 50 for a quick test)",
                default=50,
            )
            num_epochs = IntPrompt.ask("  Number of epochs (used when max_steps=-1)", default=2)
        except (KeyboardInterrupt, EOFError):
            max_steps = 50
            num_epochs = 2

    # ── Datasets ──
    console.print()
    console.print(Rule("[bold cyan]Datasets[/bold cyan]"))

    if auto:
        max_samples = 2500
        console.print(f"  Using default datasets (2x {max_samples} samples)")
    else:
        try:
            max_samples = IntPrompt.ask(
                "  Max samples per dataset (more = better quality, slower training)",
                default=2500,
            )
        except (KeyboardInterrupt, EOFError):
            max_samples = 2500

    # ── Generate config ──
    config = {
        "model": {
            "name": chosen["name"],
            "output_dir": f"./{chosen['name'].split('/')[-1].lower()}-finetuned",
            "merged_dir": f"./{chosen['name'].split('/')[-1].lower()}-finetuned-merged",
        },
        "quantization": {
            "load_in_4bit": True,
            "use_double_quant": True,
            "quant_type": "nf4",
        },
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "bias": "none",
        },
        "datasets": [
            {"name": "timdettmers/openassistant-guanaco", "split": "train", "max_samples": max_samples},
            {"name": "yahma/alpaca-cleaned", "split": "train", "max_samples": max_samples},
        ],
        "data": {
            "max_assistant_chars": 1500,
            "min_assistant_chars": 20,
            "max_length": safe["max_length"],
            "eval_split": 0.05,
        },
        "training": {
            "max_steps": max_steps,
            "num_epochs": num_epochs,
            "batch_size": safe["batch_size"],
            "gradient_accumulation_steps": safe["gradient_accumulation_steps"],
            "learning_rate": 2.0e-5,
            "warmup_ratio": 0.06,
            "lr_scheduler": "cosine",
            "max_grad_norm": 0.3,
            "weight_decay": 0.01,
            "logging_steps": 5,
            "save_steps": 200,
            "eval_steps": 50,
            "gradient_checkpointing": True,
            "optim": "paged_adamw_8bit",
            "neftune_noise_alpha": 5.0,
            "report_to": "none",
        },
        "system_prompt": "You are a helpful, accurate assistant. Provide clear and informative responses.",
        "chat": {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "max_history_turns": 6,
        },
    }

    # ── Preview ──
    console.print()
    console.print(Rule("[bold green]Generated config[/bold green]"))

    preview = Table(box=box.ROUNDED, style="green")
    preview.add_column("Setting", style="bold white")
    preview.add_column("Value", style="cyan")
    preview.add_row("Model", chosen["name"])
    preview.add_row("Parameters", chosen["params"])
    preview.add_row("Batch size", str(safe["batch_size"]))
    preview.add_row("Grad accumulation", str(safe["gradient_accumulation_steps"]))
    preview.add_row("Effective batch", str(safe["batch_size"] * safe["gradient_accumulation_steps"]))
    preview.add_row("Max length", f"{safe['max_length']} tokens")
    preview.add_row("Learning rate", "2e-5")
    preview.add_row("Max steps", str(max_steps))
    preview.add_row("Datasets", f"2 x {max_samples} samples")
    console.print(preview)

    # ── Save ──
    if CONFIG_PATH.exists() and not auto:
        try:
            overwrite = Confirm.ask(f"\n  [yellow]config.yaml already exists. Overwrite?[/yellow]", default=True)
            if not overwrite:
                backup = CONFIG_PATH.with_suffix(".yaml.bak")
                shutil.copy2(CONFIG_PATH, backup)
                console.print(f"  [dim]Backup saved to {backup}[/dim]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Cancelled[/dim]")
            return

    # Write with comments
    with open(CONFIG_PATH, "w") as f:
        f.write("# ============================================================\n")
        f.write("#  Fine-tuning Configuration\n")
        f.write(f"#  Generated by setup.py for {gpu['name'] if gpu else 'CPU'}\n")
        f.write(f"#  VRAM: {vram}GB | Batch: {safe['batch_size']} | Max length: {safe['max_length']}\n")
        f.write("# ============================================================\n\n")
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    console.print(f"\n[bold green]✓ config.yaml written![/bold green]")
    console.print()
    console.print("[dim]Next steps:[/dim]")
    console.print("  [cyan]1. python3 validate.py[/cyan]     — verify everything looks good")
    console.print("  [cyan]2. python3 finetune.py[/cyan]     — start training")
    console.print("  [cyan]3. python3 chat.py[/cyan]         — test your model")
    console.print("  [cyan]4. python3 benchmark.py[/cyan]    — score it")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup wizard for LLM fine-tuning")
    parser.add_argument("--auto", action="store_true", help="Auto-detect everything, no questions")
    args = parser.parse_args()
    run_wizard(auto=args.auto)