"""
Benchmark base vs fine-tuned model.

Runs the same prompts through both models and compares:
  - Response quality (perplexity on held-out eval data)
  - Generation speed (tokens/sec)
  - Side-by-side outputs for manual inspection
  - Coherence score (repetition ratio)

Usage:
    python3 benchmark.py              # Full benchmark
    python3 benchmark.py --quick      # Just side-by-side comparison (fewer prompts)
    python3 benchmark.py --prompts 5  # Custom number of test prompts
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import math
import time
import yaml
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich.columns import Columns
from rich import box

console = Console()

CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

BASE_MODEL_ID = CFG["model"]["name"]
FINETUNED_DIR = CFG["model"]["output_dir"]
SYSTEM_PROMPT = CFG.get("system_prompt", "You are a helpful assistant.")
CHAT_CFG = CFG.get("chat", {})

# ── Test prompts across different categories ──

TEST_PROMPTS = [
    # Factual knowledge
    {"category": "Factual", "prompt": "What is photosynthesis and why is it important?"},
    {"category": "Factual", "prompt": "Explain the difference between TCP and UDP."},
    # Reasoning
    {"category": "Reasoning", "prompt": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?"},
    {"category": "Reasoning", "prompt": "A bat and a ball cost $1.10 together. The bat costs $1.00 more than the ball. How much does the ball cost?"},
    # Instruction following
    {"category": "Instruction", "prompt": "List 3 pros and 3 cons of remote work."},
    {"category": "Instruction", "prompt": "Write a short professional email declining a meeting invitation."},
    # Creative
    {"category": "Creative", "prompt": "Write a haiku about programming."},
    {"category": "Creative", "prompt": "Explain quantum computing to a 10-year-old."},
    # Conversational
    {"category": "Conversational", "prompt": "What's a good beginner-friendly programming language and why?"},
    {"category": "Conversational", "prompt": "I'm feeling overwhelmed with too many tasks. Any advice?"},
    # Coding
    {"category": "Coding", "prompt": "Write a Python function that checks if a string is a palindrome."},
    {"category": "Coding", "prompt": "What does the 'yield' keyword do in Python?"},
]


def get_compute_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    elif torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_model(mode="base"):
    """Load model in base or finetuned mode."""
    compute_dtype = get_compute_dtype()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if mode == "finetuned":
        try:
            model = PeftModel.from_pretrained(model, FINETUNED_DIR)
        except Exception as e:
            console.print(f"[red]Failed to load adapters: {e}[/red]")
            return None, None

    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_response(model, tokenizer, prompt, max_new_tokens=256):
    """Generate a response and measure speed."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=CHAT_CFG.get("temperature", 0.7),
        top_p=CHAT_CFG.get("top_p", 0.9),
        repetition_penalty=CHAT_CFG.get("repetition_penalty", 1.1),
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    elapsed = time.time() - start

    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    num_tokens = len(new_tokens)
    tokens_per_sec = num_tokens / elapsed if elapsed > 0 else 0

    return {
        "response": response,
        "tokens": num_tokens,
        "time": elapsed,
        "tokens_per_sec": tokens_per_sec,
    }


@torch.no_grad()
def compute_perplexity(model, tokenizer, texts, max_length=256):
    """Compute perplexity on a list of texts. Lower = model is more confident."""
    total_loss = 0.0
    total_tokens = 0

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
        if inputs["input_ids"].shape[1] < 2:
            continue
        outputs = model(**inputs, labels=inputs["input_ids"])
        total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
        total_tokens += inputs["input_ids"].shape[1]

    if total_tokens == 0:
        return float("inf")
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


def compute_repetition_ratio(text):
    """Measure how repetitive a response is. Lower = less repetitive = better."""
    words = text.lower().split()
    if len(words) < 5:
        return 0.0
    # Check for repeated n-grams (trigrams)
    trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
    if not trigrams:
        return 0.0
    unique_ratio = len(set(trigrams)) / len(trigrams)
    return 1.0 - unique_ratio  # 0 = no repetition, 1 = fully repetitive


def run_benchmark(num_prompts=None, quick=False):
    prompts = TEST_PROMPTS[:num_prompts] if num_prompts else TEST_PROMPTS

    # ── Load base model ──
    console.print(Rule("[bold cyan]Loading Base Model[/bold cyan]"))
    base_model, tokenizer = load_model("base")
    if base_model is None:
        console.print("[red]Failed to load base model[/red]")
        return

    console.print("[green]✓[/green] Base model ready\n")

    # ── Generate base responses ──
    console.print(Rule("[bold cyan]Generating Base Model Responses[/bold cyan]"))
    base_results = []
    for i, item in enumerate(prompts):
        console.print(f"  [{i+1}/{len(prompts)}] {item['category']}: {item['prompt'][:50]}...")
        result = generate_response(base_model, tokenizer, item["prompt"])
        result["category"] = item["category"]
        result["prompt"] = item["prompt"]
        base_results.append(result)

    # ── Perplexity on eval data ──
    console.print("\n  Computing perplexity on eval samples...")
    eval_texts = []
    try:
        ds = load_dataset(CFG["datasets"][0]["name"], split="test")
        eval_texts = [ex["text"] for ex in ds.select(range(min(50, len(ds))))]
    except Exception:
        # Use our test prompts as fallback
        eval_texts = [p["prompt"] for p in TEST_PROMPTS]
    base_ppl = compute_perplexity(base_model, tokenizer, eval_texts)
    console.print(f"  Base perplexity: {base_ppl:.2f}")

    # ── Free base model, load finetuned ──
    del base_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    import gc; gc.collect()

    console.print(Rule("\n[bold cyan]Loading Fine-tuned Model[/bold cyan]"))
    ft_model, tokenizer = load_model("finetuned")
    if ft_model is None:
        console.print("[red]Failed to load fine-tuned model. Run finetune.py first.[/red]")
        return

    console.print("[green]✓[/green] Fine-tuned model ready\n")

    # ── Generate finetuned responses ──
    console.print(Rule("[bold cyan]Generating Fine-tuned Model Responses[/bold cyan]"))
    ft_results = []
    for i, item in enumerate(prompts):
        console.print(f"  [{i+1}/{len(prompts)}] {item['category']}: {item['prompt'][:50]}...")
        result = generate_response(ft_model, tokenizer, item["prompt"])
        result["category"] = item["category"]
        result["prompt"] = item["prompt"]
        ft_results.append(result)

    # ── Finetuned perplexity ──
    console.print("\n  Computing perplexity on eval samples...")
    ft_ppl = compute_perplexity(ft_model, tokenizer, eval_texts)
    console.print(f"  Fine-tuned perplexity: {ft_ppl:.2f}")

    # ── Summary table ──
    console.print()
    console.print(Rule("[bold green]Benchmark Results[/bold green]"))
    console.print()

    summary = Table(box=box.ROUNDED, title="Overall Metrics", style="cyan")
    summary.add_column("Metric", style="bold white")
    summary.add_column("Base", style="yellow", justify="center")
    summary.add_column("Fine-tuned", style="green", justify="center")
    summary.add_column("Winner", justify="center")

    # Avg tokens/sec
    base_tps = sum(r["tokens_per_sec"] for r in base_results) / len(base_results)
    ft_tps = sum(r["tokens_per_sec"] for r in ft_results) / len(ft_results)
    tps_winner = "[green]Fine-tuned[/green]" if ft_tps >= base_tps else "[yellow]Base[/yellow]"
    summary.add_row("Tokens/sec (avg)", f"{base_tps:.1f}", f"{ft_tps:.1f}", tps_winner)

    # Avg response length
    base_len = sum(r["tokens"] for r in base_results) / len(base_results)
    ft_len = sum(r["tokens"] for r in ft_results) / len(ft_results)
    summary.add_row("Avg response tokens", f"{base_len:.0f}", f"{ft_len:.0f}", "—")

    # Perplexity
    ppl_winner = "[green]Fine-tuned[/green]" if ft_ppl <= base_ppl else "[yellow]Base[/yellow]"
    summary.add_row("Perplexity", f"{base_ppl:.2f}", f"{ft_ppl:.2f}", ppl_winner)

    # Avg repetition
    base_rep = sum(compute_repetition_ratio(r["response"]) for r in base_results) / len(base_results)
    ft_rep = sum(compute_repetition_ratio(r["response"]) for r in ft_results) / len(ft_results)
    rep_winner = "[green]Fine-tuned[/green]" if ft_rep <= base_rep else "[yellow]Base[/yellow]"
    summary.add_row("Repetition ratio", f"{base_rep:.3f}", f"{ft_rep:.3f}", rep_winner)

    console.print(summary)

    # ── Side-by-side responses ──
    console.print()
    console.print(Rule("[bold green]Side-by-Side Comparison[/bold green]"))

    for i, (base_r, ft_r) in enumerate(zip(base_results, ft_results)):
        console.print()
        console.print(Panel(
            f"[bold]{base_r['prompt']}[/bold]",
            title=f"[cyan]Prompt {i+1}[/cyan] — {base_r['category']}",
            border_style="cyan",
        ))

        # Truncate long responses for display
        base_text = base_r["response"][:600] + ("..." if len(base_r["response"]) > 600 else "")
        ft_text = ft_r["response"][:600] + ("..." if len(ft_r["response"]) > 600 else "")

        base_rep_score = compute_repetition_ratio(base_r["response"])
        ft_rep_score = compute_repetition_ratio(ft_r["response"])

        base_panel = Panel(
            Text(base_text),
            title="[yellow]Base[/yellow]",
            subtitle=f"[dim]{base_r['tokens']}tok | {base_r['tokens_per_sec']:.1f}t/s | rep:{base_rep_score:.2f}[/dim]",
            border_style="yellow",
            width=console.width // 2 - 2,
        )
        ft_panel = Panel(
            Text(ft_text),
            title="[green]Fine-tuned[/green]",
            subtitle=f"[dim]{ft_r['tokens']}tok | {ft_r['tokens_per_sec']:.1f}t/s | rep:{ft_rep_score:.2f}[/dim]",
            border_style="green",
            width=console.width // 2 - 2,
        )

        console.print(Columns([base_panel, ft_panel], padding=1))

    # ── Per-category breakdown ──
    console.print()
    categories = sorted(set(r["category"] for r in base_results))
    cat_table = Table(box=box.ROUNDED, title="Per-Category Scores", style="cyan")
    cat_table.add_column("Category", style="bold white")
    cat_table.add_column("Base rep.", style="yellow", justify="center")
    cat_table.add_column("FT rep.", style="green", justify="center")
    cat_table.add_column("Base tok/s", style="yellow", justify="center")
    cat_table.add_column("FT tok/s", style="green", justify="center")

    for cat in categories:
        b_items = [r for r in base_results if r["category"] == cat]
        f_items = [r for r in ft_results if r["category"] == cat]
        b_rep = sum(compute_repetition_ratio(r["response"]) for r in b_items) / len(b_items)
        f_rep = sum(compute_repetition_ratio(r["response"]) for r in f_items) / len(f_items)
        b_tps = sum(r["tokens_per_sec"] for r in b_items) / len(b_items)
        f_tps = sum(r["tokens_per_sec"] for r in f_items) / len(f_items)
        cat_table.add_row(cat, f"{b_rep:.3f}", f"{f_rep:.3f}", f"{b_tps:.1f}", f"{f_tps:.1f}")

    console.print(cat_table)

    # ── Save results ──
    output_path = Path("benchmark_results.json")
    results = {
        "base": {
            "perplexity": base_ppl,
            "avg_tokens_per_sec": base_tps,
            "avg_response_tokens": base_len,
            "avg_repetition": base_rep,
            "responses": [{
                "category": r["category"],
                "prompt": r["prompt"],
                "response": r["response"],
                "tokens": r["tokens"],
                "tokens_per_sec": r["tokens_per_sec"],
            } for r in base_results],
        },
        "finetuned": {
            "perplexity": ft_ppl,
            "avg_tokens_per_sec": ft_tps,
            "avg_response_tokens": ft_len,
            "avg_repetition": ft_rep,
            "responses": [{
                "category": r["category"],
                "prompt": r["prompt"],
                "response": r["response"],
                "tokens": r["tokens"],
                "tokens_per_sec": r["tokens_per_sec"],
            } for r in ft_results],
        },
    }
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    console.print(f"\n[dim]Full results saved to {output_path}[/dim]")

    # ── Verdict ──
    console.print()
    wins = 0
    if ft_ppl <= base_ppl:
        wins += 1
    if ft_rep <= base_rep:
        wins += 1
    if ft_tps >= base_tps:
        wins += 1

    if wins >= 2:
        console.print(Panel(
            "[bold green]Fine-tuned model wins overall[/bold green]\n"
            "[dim]Check side-by-side responses above to verify quality subjectively.[/dim]",
            border_style="green",
        ))
    elif wins == 1:
        console.print(Panel(
            "[bold yellow]Mixed results — check responses manually[/bold yellow]\n"
            "[dim]Metrics are close. Read the side-by-side comparison to judge quality.[/dim]",
            border_style="yellow",
        ))
    else:
        console.print(Panel(
            "[bold red]Base model may still be better[/bold red]\n"
            "[dim]Fine-tuning may need more steps, better data, or lower learning rate.\n"
            "Try increasing max_steps in config.yaml and re-training.[/dim]",
            border_style="red",
        ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark base vs fine-tuned model")
    parser.add_argument("--quick", action="store_true", help="Quick mode: fewer prompts")
    parser.add_argument("--prompts", type=int, default=None, help="Number of test prompts to use")
    args = parser.parse_args()

    num = args.prompts or (4 if args.quick else None)
    run_benchmark(num_prompts=num)