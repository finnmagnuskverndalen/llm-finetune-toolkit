"""
Pre-flight validation for fine-tuning.

Run this BEFORE training to catch issues early:
  - Config sanity checks
  - Dataset loading and format detection
  - Token length distribution analysis
  - Memory estimation
  - GPU availability check
  - Sample formatted examples

Usage:
    python3 validate.py                # Full validation
    python3 validate.py --samples 3    # Show N formatted samples
"""

import argparse
import torch
from pathlib import Path
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from utils import load_config, detect_and_format, get_model_short_name

console = Console()
CFG = load_config()


def validate(show_samples=2):
    errors = []
    warnings = []

    model_name = get_model_short_name(CFG)
    console.print(Panel(f"[bold]Pre-flight validation — {model_name}[/bold]", border_style="cyan"))

    # ── 1. Config checks ──
    console.print("\n[bold cyan]1. Config checks[/bold cyan]")

    lr = CFG["training"]["learning_rate"]
    if lr > 5e-5:
        errors.append(f"Learning rate {lr} is dangerously high — will destroy model knowledge. Use 1e-5 to 5e-5.")
    elif lr > 3e-5:
        warnings.append(f"Learning rate {lr} is on the high side. Consider 2e-5 for small models.")
    console.print(f"  Learning rate: {lr} {'[red]✗[/red]' if lr > 5e-5 else '[green]✓[/green]'}")

    max_len = CFG["data"]["max_length"]
    if max_len < 512:
        warnings.append(f"max_length={max_len} is short — model can't learn detailed responses.")
    console.print(f"  Max length: {max_len} tokens {'[yellow]⚠[/yellow]' if max_len < 512 else '[green]✓[/green]'}")

    r = CFG["lora"]["r"]
    alpha = CFG["lora"]["alpha"]
    if alpha < r:
        warnings.append(f"lora_alpha ({alpha}) < r ({r}). Typical: alpha = 2*r.")
    console.print(f"  LoRA: r={r}, alpha={alpha} {'[yellow]⚠[/yellow]' if alpha < r else '[green]✓[/green]'}")

    max_chars = CFG["data"]["max_assistant_chars"]
    if max_chars < 500:
        warnings.append(f"max_assistant_chars={max_chars} is aggressive — discards detailed responses.")
    console.print(f"  Max assistant chars: {max_chars} {'[yellow]⚠[/yellow]' if max_chars < 500 else '[green]✓[/green]'}")

    grad_norm = CFG["training"].get("max_grad_norm", None)
    if grad_norm is None:
        warnings.append("No gradient clipping — risk of catastrophic updates.")
    console.print(f"  Gradient clipping: {grad_norm or 'OFF'} {'[yellow]⚠[/yellow]' if not grad_norm else '[green]✓[/green]'}")

    bs = CFG["training"]["batch_size"]
    max_steps = CFG["training"].get("max_steps", -1)
    console.print(f"  Batch size: {bs}")
    if max_steps > 0:
        console.print(f"  Max steps: {max_steps} (overrides epochs)")
    else:
        console.print(f"  Epochs: {CFG['training']['num_epochs']}")

    # ── 2. GPU check ──
    console.print("\n[bold cyan]2. Hardware[/bold cyan]")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_properties(0).name
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        console.print(f"  GPU: [green]{gpu_name}[/green] ({gpu_mem:.1f} GB)")
        if gpu_mem < 2.5:
            warnings.append(f"GPU has only {gpu_mem:.1f} GB — use batch_size=1 and max_length=256.")
    else:
        console.print("  GPU: [red]not available — training on CPU (slow)[/red]")
        warnings.append("No GPU detected. Training will be very slow.")

    # ── 3. Tokenizer ──
    console.print("\n[bold cyan]3. Tokenizer[/bold cyan]")
    model_id = CFG["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    console.print(f"  Model: {model_id}")
    console.print(f"  Vocab size: {tokenizer.vocab_size}")
    console.print(f"  Chat template: {'[green]yes[/green]' if tokenizer.chat_template else '[red]no[/red]'}")

    if tokenizer.vocab_size > 100000 and torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_mem < 4:
            warnings.append(f"Large vocabulary ({tokenizer.vocab_size}) with small GPU ({gpu_mem:.1f}GB). "
                            "The lm_head projection may OOM. Consider SmolLM2 or Llama 3.2 (smaller vocab).")

    # ── 4. Datasets (uses shared format functions) ──
    console.print("\n[bold cyan]4. Datasets[/bold cyan]")
    system_prompt = CFG["system_prompt"]

    all_datasets = []
    for ds_cfg in CFG["datasets"]:
        name = ds_cfg["name"]
        max_s = ds_cfg.get("max_samples")
        ds = load_dataset(name, split=ds_cfg.get("split", "train"))
        if max_s and max_s < len(ds):
            ds = ds.select(range(max_s))

        ds = ds.map(lambda ex: detect_and_format(ex, system_prompt))
        before = len(ds)
        ds = ds.filter(lambda x: x["messages"] is not None and len(x["messages"]) >= 3)
        after = len(ds)
        console.print(f"  {name}: {after}/{before} valid ({after/before*100:.0f}%)")
        all_datasets.append(ds)

    combined = concatenate_datasets(all_datasets)

    min_c = CFG["data"]["min_assistant_chars"]
    max_c = CFG["data"]["max_assistant_chars"]
    before = len(combined)
    combined = combined.filter(lambda x: all(
        min_c <= len(m["content"]) <= max_c
        for m in x["messages"] if m["role"] == "assistant"
    ))
    after = len(combined)
    console.print(f"  After quality filter: {after}/{before} ({after/before*100:.0f}%)")

    # ── 5. Token statistics ──
    console.print("\n[bold cyan]5. Token distribution[/bold cyan]")

    sample_size = min(1000, len(combined))
    sample = combined.select(range(sample_size))

    token_lengths = []
    for ex in sample:
        text = tokenizer.apply_chat_template(ex["messages"], tokenize=False)
        toks = tokenizer(text, truncation=False)["input_ids"]
        token_lengths.append(len(toks))

    token_lengths.sort()

    def percentile(data, p):
        idx = int(len(data) * p / 100)
        return data[min(idx, len(data) - 1)]

    stats_table = Table(box=box.SIMPLE, title="Token lengths")
    stats_table.add_column("Stat", style="bold")
    stats_table.add_column("Value", style="yellow")
    stats_table.add_row("Min", str(min(token_lengths)))
    stats_table.add_row("Median", str(percentile(token_lengths, 50)))
    stats_table.add_row("Mean", f"{sum(token_lengths)/len(token_lengths):.0f}")
    stats_table.add_row("P90", str(percentile(token_lengths, 90)))
    stats_table.add_row("P95", str(percentile(token_lengths, 95)))
    stats_table.add_row("Max", str(max(token_lengths)))
    console.print(stats_table)

    truncated = sum(1 for l in token_lengths if l > max_len)
    pct = truncated / len(token_lengths) * 100
    if pct > 20:
        warnings.append(f"{pct:.0f}% of samples will be truncated at max_length={max_len}.")
    console.print(f"  Truncated at {max_len}: {truncated}/{len(token_lengths)} ({pct:.1f}%)")

    # ── 6. Show samples ──
    if show_samples > 0:
        console.print(f"\n[bold cyan]6. Sample formatted examples[/bold cyan]")
        for i in range(min(show_samples, len(combined))):
            ex = combined[i]
            text = tokenizer.apply_chat_template(ex["messages"], tokenize=False)
            toks = len(tokenizer(text, truncation=False)["input_ids"])
            console.print(Panel(
                text[:800] + ("..." if len(text) > 800 else ""),
                title=f"[yellow]Sample {i+1}[/yellow] ({toks} tokens)",
                border_style="dim",
            ))

    # ── 7. Resource estimate ──
    console.print(f"\n[bold cyan]7. Resource estimate[/bold cyan]")
    model_params = {"0.5B": 0.5, "1.5B": 1.5, "2B": 2, "3B": 3, "7B": 7}
    param_count = None
    for key, val in model_params.items():
        if key.lower() in model_id.lower():
            param_count = val
            break
    if param_count:
        vram_est = param_count * 0.5 + 1.5
        console.print(f"  Estimated VRAM: ~{vram_est:.1f} GB (4-bit quantized)")

    ga = CFG["training"]["gradient_accumulation_steps"]
    epochs = CFG["training"]["num_epochs"]
    effective_bs = bs * ga
    steps = (after // effective_bs) * epochs
    max_s = CFG["training"].get("max_steps", -1)
    if max_s > 0:
        steps = max_s

    console.print(f"  Effective batch size: {bs} × {ga} = {effective_bs}")
    console.print(f"  Estimated steps: ~{steps}")
    console.print(f"  Dataset: {after} examples × {epochs} epochs")

    # ── Summary ──
    console.print()
    if errors:
        console.print(Panel(
            "\n".join(f"[red]✗ {e}[/red]" for e in errors),
            title="[red bold]Errors — fix before training[/red bold]",
            border_style="red",
        ))
    if warnings:
        console.print(Panel(
            "\n".join(f"[yellow]⚠ {w}[/yellow]" for w in warnings),
            title="[yellow bold]Warnings[/yellow bold]",
            border_style="yellow",
        ))
    if not errors and not warnings:
        console.print("[bold green]✓ All checks passed! Ready to train.[/bold green]")
    elif not errors:
        console.print("[bold green]✓ No blockers. Review warnings above.[/bold green]")
    else:
        console.print("[bold red]✗ Fix errors above before training.[/bold red]")

    console.print()
    console.print("[dim]Next: python3 finetune.py[/dim]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2, help="Number of examples to show")
    args = parser.parse_args()
    validate(show_samples=args.samples)