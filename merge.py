"""
Merge LoRA adapters into the base model and export.

This produces a standalone model that:
  - Doesn't need PEFT at inference time
  - Can be uploaded to HuggingFace Hub
  - Loads faster (no adapter merging at startup)
  - Can be quantized with other methods (GGUF, AWQ, etc.)

Usage:
    python merge.py                    # Merge with defaults from config.yaml
    python merge.py --push username/my-model   # Merge and push to HF Hub
"""

import argparse
import yaml
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from rich.console import Console
from rich.rule import Rule

console = Console()

CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)


def merge(push_to: str = None):
    base_id = CFG["model"]["name"]
    adapter_dir = CFG["model"]["output_dir"]
    output_dir = CFG["model"]["merged_dir"]

    if not Path(adapter_dir).exists():
        console.print(f"[red]✗ Adapter directory not found: {adapter_dir}[/red]")
        console.print("[yellow]  Run finetune.py first.[/yellow]")
        return

    console.print(Rule("[bold cyan]Merging LoRA Adapters[/bold cyan]"))

    # Load base model in full precision for merging
    with console.status("[cyan]Loading base model (full precision)...[/cyan]"):
        model = AutoModelForCausalLM.from_pretrained(
            base_id,
            torch_dtype=torch.float16,
            device_map="cpu",  # CPU to avoid OOM during merge
            trust_remote_code=True,
        )
    console.print("[green]✓[/green] Base model loaded")

    with console.status("[cyan]Loading LoRA adapters...[/cyan]"):
        model = PeftModel.from_pretrained(model, adapter_dir)
    console.print("[green]✓[/green] Adapters loaded")

    with console.status("[cyan]Merging weights...[/cyan]"):
        model = model.merge_and_unload()
    console.print("[green]✓[/green] Weights merged")

    with console.status(f"[cyan]Saving to {output_dir}...[/cyan]"):
        model.save_pretrained(output_dir)
        tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
        tokenizer.save_pretrained(output_dir)
    console.print(f"[green]✓[/green] Saved to {output_dir}")

    if push_to:
        console.print(Rule("[bold cyan]Pushing to HuggingFace Hub[/bold cyan]"))
        with console.status(f"[cyan]Uploading to {push_to}...[/cyan]"):
            model.push_to_hub(push_to)
            tokenizer.push_to_hub(push_to)
        console.print(f"[green]✓[/green] Pushed to https://huggingface.co/{push_to}")

    console.print()
    console.print("[bold green]✓ Merge complete![/bold green]")
    console.print(f"[cyan]  Test with: python chat.py --merged[/cyan]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into base model")
    parser.add_argument("--push", type=str, default=None, help="Push to HF Hub (e.g., username/model-name)")
    args = parser.parse_args()
    merge(push_to=args.push)
