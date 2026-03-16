"""
Clean up generated models, caches, and training artifacts.

Removes fine-tuned adapters, merged models, GGUF exports, HuggingFace cache,
and benchmark history. Useful when switching models, freeing disk space,
or starting fresh.

Usage:
    python3 cleanup.py              # Interactive — choose what to remove
    python3 cleanup.py --all        # Remove everything (no prompts)
    python3 cleanup.py --models     # Remove only trained/merged/exported models
    python3 cleanup.py --cache      # Remove only HuggingFace model cache
    python3 cleanup.py --dry-run    # Show what would be removed without deleting
"""

import argparse
import shutil
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule
from rich.prompt import Confirm
from rich import box

from utils import load_config, get_model_short_name

console = Console()


def get_size(path):
    """Get total size of a file or directory in bytes."""
    path = Path(path)
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def format_size(size_bytes):
    """Format bytes into human-readable string."""
    if size_bytes == 0:
        return "0 B"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def find_hf_cache_dirs(model_name):
    """
    Find HuggingFace cache directories for the base model.
    HF stores models in ~/.cache/huggingface/hub/models--org--name/
    """
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    if not cache_root.exists():
        return []

    # Convert "org/model-name" to "models--org--model-name"
    cache_name = f"models--{model_name.replace('/', '--')}"
    cache_dir = cache_root / cache_name

    found = []
    if cache_dir.exists():
        found.append(cache_dir)

    return found


def find_hf_dataset_cache():
    """Find HuggingFace datasets cache directory."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
    if cache_dir.exists() and any(cache_dir.iterdir()):
        return cache_dir
    return None


def scan_targets(cfg):
    """
    Scan for all removable targets based on config.
    Returns list of dicts: {name, path, size, category}
    """
    targets = []
    model_name = cfg["model"]["name"]
    short_name = get_model_short_name(cfg)

    # ── Trained model artifacts ──

    # LoRA adapters (output_dir)
    adapter_dir = Path(cfg["model"]["output_dir"])
    if adapter_dir.exists():
        targets.append({
            "name": f"LoRA adapters ({short_name})",
            "path": adapter_dir,
            "size": get_size(adapter_dir),
            "category": "models",
        })

    # Merged model
    merged_dir = Path(cfg["model"]["merged_dir"])
    if merged_dir.exists():
        targets.append({
            "name": f"Merged model ({short_name})",
            "path": merged_dir,
            "size": get_size(merged_dir),
            "category": "models",
        })

    # GGUF export directory
    gguf_dir = Path(cfg["model"].get("gguf_dir", f"./{short_name.lower()}-finetuned-gguf"))
    if not gguf_dir.exists():
        # Try the default pattern from export.py
        gguf_dir = Path(f"./{short_name.lower()}-finetuned-gguf")
    # Also scan for any *-gguf directories in the project root
    for p in Path(".").glob("*-gguf"):
        if p.is_dir():
            targets.append({
                "name": f"GGUF export ({p.name})",
                "path": p,
                "size": get_size(p),
                "category": "models",
            })

    if gguf_dir.exists() and not any(t["path"] == gguf_dir for t in targets):
        targets.append({
            "name": f"GGUF export ({gguf_dir.name})",
            "path": gguf_dir,
            "size": get_size(gguf_dir),
            "category": "models",
        })

    # ── HuggingFace cache ──

    for cache_dir in find_hf_cache_dirs(model_name):
        targets.append({
            "name": f"HF cache: {model_name}",
            "path": cache_dir,
            "size": get_size(cache_dir),
            "category": "cache",
        })

    # Also check for tokenizer-only caches (some models download separately)
    # and any other model caches in the HF hub directory
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    if cache_root.exists():
        # Find caches for datasets used in training
        for ds_cfg in cfg.get("datasets", []):
            ds_name = ds_cfg["name"]
            ds_cache_name = f"datasets--{ds_name.replace('/', '--')}"
            ds_cache = cache_root / ds_cache_name
            if ds_cache.exists():
                targets.append({
                    "name": f"HF cache: {ds_name}",
                    "path": ds_cache,
                    "size": get_size(ds_cache),
                    "category": "cache",
                })

        # Abliteration dataset caches
        ablit = cfg.get("abliteration", {})
        for ds_key in ("harmful_dataset", "harmless_dataset"):
            ds_name = ablit.get(ds_key)
            if ds_name:
                ds_cache_name = f"datasets--{ds_name.replace('/', '--')}"
                ds_cache = cache_root / ds_cache_name
                if ds_cache.exists() and not any(t["path"] == ds_cache for t in targets):
                    targets.append({
                        "name": f"HF cache: {ds_name}",
                        "path": ds_cache,
                        "size": get_size(ds_cache),
                        "category": "cache",
                    })

    # ── Training artifacts ──

    # Benchmark history
    history_path = Path(__file__).parent / "benchmark_history.json"
    if history_path.exists():
        targets.append({
            "name": "Benchmark history",
            "path": history_path,
            "size": get_size(history_path),
            "category": "artifacts",
        })

    # Training config backup
    config_bak = Path(__file__).parent / "config.yaml.bak"
    if config_bak.exists():
        targets.append({
            "name": "Config backup",
            "path": config_bak,
            "size": get_size(config_bak),
            "category": "artifacts",
        })

    # Checkpoint directories inside adapter dir (leftover from interrupted training)
    for p in Path(".").glob("*-finetuned/checkpoint-*"):
        if p.is_dir():
            targets.append({
                "name": f"Checkpoint ({p.parent.name}/{p.name})",
                "path": p,
                "size": get_size(p),
                "category": "artifacts",
            })

    return targets


def display_targets(targets):
    """Display all found targets in a table."""
    if not targets:
        console.print("[green]Nothing to clean up — no generated files found.[/green]")
        return

    table = Table(box=box.ROUNDED, title="Found files and directories", style="cyan")
    table.add_column("#", style="dim", width=4)
    table.add_column("Category", style="bold", width=12)
    table.add_column("Name", style="white")
    table.add_column("Size", justify="right", style="yellow", width=12)
    table.add_column("Path", style="dim")

    total_size = 0
    for i, t in enumerate(targets):
        cat_style = {
            "models": "[magenta]models[/magenta]",
            "cache": "[cyan]cache[/cyan]",
            "artifacts": "[dim]artifacts[/dim]",
        }.get(t["category"], t["category"])

        table.add_row(
            str(i + 1),
            cat_style,
            t["name"],
            format_size(t["size"]),
            str(t["path"]),
        )
        total_size += t["size"]

    console.print(table)
    console.print(f"\n  Total: [bold yellow]{format_size(total_size)}[/bold yellow]")


def remove_targets(targets, dry_run=False):
    """Remove the given targets. Returns (removed_count, freed_bytes)."""
    removed = 0
    freed = 0

    for t in targets:
        path = Path(t["path"])
        if not path.exists():
            continue

        if dry_run:
            console.print(f"  [dim]Would remove:[/dim] {path}")
            removed += 1
            freed += t["size"]
            continue

        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            console.print(f"  [green]Removed:[/green] {path} ({format_size(t['size'])})")
            removed += 1
            freed += t["size"]
        except PermissionError:
            console.print(f"  [red]Permission denied:[/red] {path}")
        except Exception as e:
            console.print(f"  [red]Failed:[/red] {path} — {e}")

    return removed, freed


def main():
    parser = argparse.ArgumentParser(
        description="Clean up generated models, caches, and training artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 cleanup.py              # Interactive\n"
            "  python3 cleanup.py --all        # Remove everything\n"
            "  python3 cleanup.py --models     # Only trained/merged/exported models\n"
            "  python3 cleanup.py --cache      # Only HuggingFace cache\n"
            "  python3 cleanup.py --dry-run    # Preview without deleting\n"
        ),
    )
    parser.add_argument("--all", action="store_true",
                        help="Remove everything without prompting")
    parser.add_argument("--models", action="store_true",
                        help="Remove only trained/merged/exported models")
    parser.add_argument("--cache", action="store_true",
                        help="Remove only HuggingFace model and dataset caches")
    parser.add_argument("--artifacts", action="store_true",
                        help="Remove only training artifacts (history, checkpoints)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be removed without deleting")
    args = parser.parse_args()

    cfg = load_config()
    model_short = get_model_short_name(cfg)

    console.print(Panel(
        f"[bold]Cleanup — {model_short}[/bold]\n"
        "[dim]Remove generated models, caches, and training artifacts[/dim]",
        border_style="cyan",
    ))

    # Scan
    all_targets = scan_targets(cfg)

    if not all_targets:
        console.print("[green]Nothing to clean up — no generated files found.[/green]")
        return

    # Filter by category if flags are set
    if args.models or args.cache or args.artifacts:
        categories = set()
        if args.models:
            categories.add("models")
        if args.cache:
            categories.add("cache")
        if args.artifacts:
            categories.add("artifacts")
        targets = [t for t in all_targets if t["category"] in categories]
    else:
        targets = all_targets

    if not targets:
        console.print("[green]Nothing matching the selected category.[/green]")
        return

    display_targets(targets)
    console.print()

    if args.dry_run:
        console.print(Rule("[bold yellow]Dry run[/bold yellow]"))
        removed, freed = remove_targets(targets, dry_run=True)
        console.print(f"\n  [yellow]Would remove {removed} items, freeing {format_size(freed)}[/yellow]")
        return

    if args.all:
        # No prompting
        console.print(Rule("[bold red]Removing all[/bold red]"))
        removed, freed = remove_targets(targets)
        console.print(f"\n[bold green]✓ Removed {removed} items, freed {format_size(freed)}[/bold green]")
        return

    # Interactive mode — let user choose categories
    console.print(Rule("[bold cyan]What to remove?[/bold cyan]"))

    categories_present = sorted(set(t["category"] for t in targets))
    category_labels = {
        "models": "Trained models (adapters, merged, GGUF)",
        "cache": "HuggingFace cache (base model + datasets)",
        "artifacts": "Training artifacts (benchmark history, checkpoints)",
    }

    selected = []
    for cat in categories_present:
        cat_targets = [t for t in targets if t["category"] == cat]
        cat_size = sum(t["size"] for t in cat_targets)
        label = category_labels.get(cat, cat)

        try:
            if Confirm.ask(
                f"  Remove {label}? [yellow]({format_size(cat_size)})[/yellow]",
                default=False,
            ):
                selected.extend(cat_targets)
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Cancelled[/dim]")
            return

    if not selected:
        console.print("[dim]Nothing selected. Exiting.[/dim]")
        return

    total_size = sum(t["size"] for t in selected)
    console.print()

    try:
        confirm = Confirm.ask(
            f"  [bold red]Delete {len(selected)} items ({format_size(total_size)})?[/bold red]",
            default=False,
        )
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]Cancelled[/dim]")
        return

    if not confirm:
        console.print("[dim]Cancelled[/dim]")
        return

    console.print()
    removed, freed = remove_targets(selected)
    console.print(f"\n[bold green]✓ Removed {removed} items, freed {format_size(freed)}[/bold green]")


if __name__ == "__main__":
    main()