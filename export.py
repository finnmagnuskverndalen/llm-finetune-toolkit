"""
Export fine-tuned model to GGUF format for use with Ollama and llama.cpp.

Pipeline: merge adapters → convert to GGUF → quantize → create Ollama model

Prerequisites:
    pip install llama-cpp-python --break-system-packages
    # OR clone llama.cpp for the convert script:
    git clone https://github.com/ggml-org/llama.cpp.git
    pip install -r llama.cpp/requirements.txt --break-system-packages

Usage:
    python3 export.py                        # Convert to GGUF (Q8_0) and register with Ollama
    python3 export.py --quantize q4_k_m      # Use Q4_K_M quantization (smaller, faster)
    python3 export.py --quantize f16          # Full precision GGUF (largest, most accurate)
    python3 export.py --name my-model         # Custom Ollama model name (skips interactive prompt)
    python3 export.py --skip-ollama           # Only produce GGUF file, don't register with Ollama
    python3 export.py --ollama-only model.gguf  # Skip conversion, just register existing GGUF
    python3 export.py --no-interactive        # Use default name without prompting (for scripting)

Naming:
    By default, the script proposes a name based on the model (e.g. "qwen2.5-0.5b-instruct-finetuned")
    and prompts you to accept or type a custom name. Use --name to skip the prompt, or
    --no-interactive to accept the default silently.
"""

import argparse
import os
import shutil
import subprocess
import sys
import yaml
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich import box

console = Console()

CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

MERGED_DIR = Path(CFG["model"]["merged_dir"])
ADAPTER_DIR = Path(CFG["model"]["output_dir"])
GGUF_DIR = Path(CFG["model"].get("gguf_dir", "./qwen-finetuned-gguf"))

QUANTIZATION_OPTIONS = {
    "f32":    "Full 32-bit float — largest, original quality",
    "f16":    "16-bit float — large, near-original quality",
    "q8_0":   "8-bit quantized — good balance of quality and size (default)",
    "q6_k":   "6-bit quantized — slightly smaller, minimal quality loss",
    "q5_k_m": "5-bit quantized — smaller, good quality for most use cases",
    "q4_k_m": "4-bit quantized — small and fast, some quality loss",
    "q4_0":   "4-bit quantized — smallest, most quality loss",
    "q3_k_m": "3-bit quantized — very small, noticeable quality loss",
    "q2_k":   "2-bit quantized — tiny, significant quality loss",
}


def find_llama_cpp():
    """Find llama.cpp convert script."""
    # Check common locations
    candidates = [
        Path("llama.cpp"),
        Path.home() / "llama.cpp",
        Path("/opt/llama.cpp"),
    ]
    for p in candidates:
        convert_script = p / "convert_hf_to_gguf.py"
        if convert_script.exists():
            return convert_script

    # Check if llama-cpp-python installed the converter
    try:
        result = subprocess.run(
            ["python3", "-c", "import llama_cpp; print(llama_cpp.__file__)"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            pkg_dir = Path(result.stdout.strip()).parent
            convert_script = pkg_dir / "llama_cpp" / "convert_hf_to_gguf.py"
            if convert_script.exists():
                return convert_script
    except Exception:
        pass

    return None


def check_ollama():
    """Check if Ollama is installed and running."""
    if not shutil.which("ollama"):
        return False, "Ollama not found. Install from https://ollama.com"
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return False, "Ollama installed but not running. Start with: ollama serve"
        return True, "OK"
    except subprocess.TimeoutExpired:
        return False, "Ollama not responding. Start with: ollama serve"
    except Exception as e:
        return False, f"Ollama error: {e}"


def ensure_merged():
    """Make sure the merged model exists, run merge if not."""
    if MERGED_DIR.exists() and (MERGED_DIR / "config.json").exists():
        console.print(f"[green]✓[/green] Merged model found at {MERGED_DIR}")
        return True

    console.print("[yellow]Merged model not found. Running merge first...[/yellow]")
    try:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "merge.py")],
            check=True,
        )
        return MERGED_DIR.exists()
    except subprocess.CalledProcessError:
        console.print("[red]✗ Merge failed. Run merge.py manually to see errors.[/red]")
        return False


def convert_to_gguf(quantize="q8_0"):
    """Convert merged HF model to GGUF format."""
    console.print(Rule("[bold cyan]Converting to GGUF[/bold cyan]"))

    GGUF_DIR.mkdir(parents=True, exist_ok=True)

    model_name = CFG["model"]["name"].split("/")[-1].lower()
    gguf_filename = f"{model_name}-finetuned-{quantize}.gguf"
    gguf_path = GGUF_DIR / gguf_filename

    # Method 1: Try llama.cpp convert script
    convert_script = find_llama_cpp()

    if convert_script:
        console.print(f"[cyan]Using llama.cpp converter: {convert_script}[/cyan]")
        console.print(f"[cyan]Quantization: {quantize} — {QUANTIZATION_OPTIONS.get(quantize, '')}[/cyan]")

        cmd = [
            sys.executable, str(convert_script),
            str(MERGED_DIR),
            "--outfile", str(gguf_path),
            "--outtype", quantize,
        ]

        with console.status("[cyan]Converting (this may take a few minutes)...[/cyan]"):
            result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            size_mb = gguf_path.stat().st_size / (1024 * 1024)
            console.print(f"[green]✓[/green] GGUF created: {gguf_path} ({size_mb:.1f} MB)")
            return gguf_path
        else:
            console.print(f"[red]✗ Conversion failed:[/red]")
            console.print(f"[dim]{result.stderr[-500:]}[/dim]")
            return None

    # Method 2: Try ollama create directly from safetensors
    ollama_ok, _ = check_ollama()
    if ollama_ok:
        console.print("[yellow]llama.cpp not found. Attempting direct Ollama import from safetensors...[/yellow]")
        return "direct_ollama"

    # Neither method available
    console.print(Panel(
        "[red]Could not find a GGUF converter.[/red]\n\n"
        "Install llama.cpp:\n"
        "  [cyan]git clone https://github.com/ggml-org/llama.cpp.git[/cyan]\n"
        "  [cyan]pip3 install -r llama.cpp/requirements.txt --break-system-packages[/cyan]\n\n"
        "Or install Ollama (can import safetensors directly):\n"
        "  [cyan]curl -fsSL https://ollama.com/install.sh | sh[/cyan]",
        title="[red]Missing Dependencies[/red]",
        border_style="red",
    ))
    return None


def prompt_model_name(proposed_name):
    """Interactively ask the user to confirm or override the Ollama model name."""
    console.print(f"\n[cyan]Proposed Ollama model name:[/cyan] [bold]{proposed_name}[/bold]")
    console.print("[dim]Press Enter to accept, or type a custom name:[/dim]")
    try:
        user_input = input("  Model name: ").strip()
    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]Cancelled.[/yellow]")
        sys.exit(0)

    if user_input:
        # Sanitize: lowercase, replace spaces with hyphens, strip invalid chars
        sanitized = user_input.lower().replace(" ", "-")
        sanitized = "".join(c for c in sanitized if c.isalnum() or c in "-_.:").strip("-_.")
        if sanitized != user_input.strip():
            console.print(f"[dim]  (sanitized to: {sanitized})[/dim]")
        if not sanitized:
            console.print(f"[yellow]Invalid name, using default: {proposed_name}[/yellow]")
            return proposed_name
        return sanitized
    return proposed_name


def create_ollama_model(gguf_path, model_name=None, interactive=True):
    """Create an Ollama model from GGUF file or safetensors directory."""
    console.print(Rule("[bold cyan]Creating Ollama Model[/bold cyan]"))

    ollama_ok, msg = check_ollama()
    if not ollama_ok:
        console.print(f"[red]✗ {msg}[/red]")
        return False

    # Build the default name
    base_name = CFG["model"]["name"].split("/")[-1].lower()
    default_name = f"{base_name}-finetuned"

    if model_name is not None:
        # Explicit --name flag: use it directly, no prompt
        pass
    elif interactive:
        # No --name flag: propose the default and let the user override
        model_name = prompt_model_name(default_name)
    else:
        model_name = default_name

    # Determine the FROM source
    if gguf_path == "direct_ollama":
        # Ollama can import directly from a safetensors directory
        from_source = str(MERGED_DIR.resolve())
    else:
        from_source = str(Path(gguf_path).resolve())

    # Build chat template based on model
    model_id = CFG["model"]["name"].lower()
    if "qwen" in model_id:
        template = """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>"""
        stop_tokens = ['PARAMETER stop "<|im_end|>"', 'PARAMETER stop "<|im_start|>"']
    elif "llama" in model_id:
        template = """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"""
        stop_tokens = ['PARAMETER stop "<|eot_id|>"']
    elif "phi" in model_id:
        template = """{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
{{ end }}<|assistant|>
{{ .Response }}<|end|>"""
        stop_tokens = ['PARAMETER stop "<|end|>"']
    elif "gemma" in model_id:
        template = """{{ if .System }}<start_of_turn>user
{{ .System }}

{{ end }}{{ if .Prompt }}<start_of_turn>user
{{ .Prompt }}<end_of_turn>
{{ end }}<start_of_turn>model
{{ .Response }}<end_of_turn>"""
        stop_tokens = ['PARAMETER stop "<end_of_turn>"']
    else:
        # Generic ChatML
        template = """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>"""
        stop_tokens = ['PARAMETER stop "<|im_end|>"']

    system_prompt = CFG.get("system_prompt", "You are a helpful assistant.")

    # Write Modelfile
    modelfile_path = GGUF_DIR / "Modelfile"
    modelfile_content = f"""FROM {from_source}

TEMPLATE \"\"\"{template}\"\"\"

SYSTEM \"\"\"{system_prompt}\"\"\"

PARAMETER temperature {CFG.get('chat', {}).get('temperature', 0.7)}
PARAMETER top_p {CFG.get('chat', {}).get('top_p', 0.9)}
PARAMETER repeat_penalty {CFG.get('chat', {}).get('repetition_penalty', 1.1)}
{chr(10).join(stop_tokens)}
"""

    modelfile_path.write_text(modelfile_content)
    console.print(f"[green]✓[/green] Modelfile written to {modelfile_path}")
    console.print(f"[cyan]  Model name: {model_name}[/cyan]")

    # Create the model in Ollama
    with console.status(f"[cyan]Registering model '{model_name}' with Ollama...[/cyan]"):
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            capture_output=True, text=True,
        )

    if result.returncode == 0:
        console.print(f"[green]✓[/green] Model registered with Ollama!")
        console.print()
        console.print(Panel(
            f"[bold green]Run your fine-tuned model with:[/bold green]\n\n"
            f"  [cyan]ollama run {model_name}[/cyan]\n\n"
            f"[dim]Or use the API:[/dim]\n\n"
            f"  [cyan]curl http://localhost:11434/api/chat -d '{{\n"
            f"    \"model\": \"{model_name}\",\n"
            f"    \"messages\": [{{\"role\": \"user\", \"content\": \"Hello!\"}}]\n"
            f"  }}'[/cyan]",
            title="[green]Ready![/green]",
            border_style="green",
        ))
        return True
    else:
        console.print(f"[red]✗ Ollama create failed:[/red]")
        console.print(f"[dim]{result.stderr}[/dim]")
        console.print(f"\n[yellow]You can try manually:[/yellow]")
        console.print(f"  [cyan]ollama create {model_name} -f {modelfile_path}[/cyan]")
        return False


def list_quantizations():
    """Show available quantization options."""
    table = table(box=box.ROUNDED, title="Available Quantization Options", style="cyan")
    table.add_column("Type", style="bold white")
    table.add_column("Description", style="dim white")
    for qtype, desc in QUANTIZATION_OPTIONS.items():
        table.add_row(qtype, desc)
    console.print(table)


def main():
    parser = argparse.ArgumentParser(
        description="Export fine-tuned model to GGUF/Ollama format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python3 export.py                      # Default: Q8_0 + register with Ollama\n"
               "  python3 export.py --quantize q4_k_m    # Smaller, faster model\n"
               "  python3 export.py --name my-assistant   # Custom Ollama model name\n"
               "  python3 export.py --list-quantizations  # Show all quantization options\n"
               "  python3 export.py --skip-ollama         # Only create GGUF file\n"
               "  python3 export.py --ollama-only model.gguf  # Register existing GGUF\n",
    )
    parser.add_argument("--quantize", "-q", type=str, default="q8_0",
                        help="Quantization type (default: q8_0)")
    parser.add_argument("--name", "-n", type=str, default=None,
                        help="Ollama model name (skips interactive prompt)")
    parser.add_argument("--skip-ollama", action="store_true",
                        help="Only create GGUF file, don't register with Ollama")
    parser.add_argument("--ollama-only", type=str, default=None, metavar="GGUF_PATH",
                        help="Skip conversion, register existing GGUF with Ollama")
    parser.add_argument("--list-quantizations", action="store_true",
                        help="Show available quantization types")
    parser.add_argument("--no-interactive", action="store_true",
                        help="Skip name prompt, use default name (for scripting)")
    args = parser.parse_args()

    if args.list_quantizations:
        list_quantizations()
        return

    console.print(Panel(
        "[bold]Export Pipeline[/bold]: merge → GGUF → Ollama",
        border_style="cyan",
    ))

    # If just registering an existing GGUF
    if args.ollama_only:
        gguf_path = Path(args.ollama_only)
        if not gguf_path.exists():
            console.print(f"[red]✗ GGUF file not found: {gguf_path}[/red]")
            return
        GGUF_DIR.mkdir(parents=True, exist_ok=True)
        create_ollama_model(str(gguf_path), model_name=args.name, interactive=not args.no_interactive)
        return

    # Full pipeline
    # Step 1: Ensure merged model exists
    if not ensure_merged():
        return

    # Step 2: Convert to GGUF
    if args.quantize not in QUANTIZATION_OPTIONS:
        console.print(f"[red]✗ Unknown quantization: {args.quantize}[/red]")
        list_quantizations()
        return

    gguf_path = convert_to_gguf(quantize=args.quantize)
    if gguf_path is None:
        return

    # Step 3: Register with Ollama
    if not args.skip_ollama:
        create_ollama_model(gguf_path, model_name=args.name, interactive=not args.no_interactive)
    else:
        console.print()
        console.print("[bold green]✓ GGUF export complete![/bold green]")
        if gguf_path != "direct_ollama":
            console.print(f"[cyan]  File: {gguf_path}[/cyan]")
        console.print(f"[cyan]  Register with Ollama later: python3 export.py --ollama-only {gguf_path}[/cyan]")


if __name__ == "__main__":
    main()