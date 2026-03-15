"""
Benchmark base vs fine-tuned model with scoring and historical tracking.

Every run is saved to benchmark_history.json with a composite score (0-100).
Use --history to view score trends over time and see if training is improving.

Scoring breakdown (100 points max):
  - Perplexity improvement over base:  35 points
  - Coherence (low repetition):        25 points
  - Response quality (length + detail): 20 points
  - Speed (tokens/sec vs base):        10 points
  - Consistency across categories:      10 points

Usage:
    python3 benchmark.py              # Full benchmark (12 prompts)
    python3 benchmark.py --quick      # Quick mode (4 prompts)
    python3 benchmark.py --prompts 5  # Custom number of prompts
    python3 benchmark.py --history    # View historical scores and trends
    python3 benchmark.py --tag "50 steps lr=2e-5"  # Tag this run with a note
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import math
import time
import yaml
import torch
import gc
from datetime import datetime
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
HISTORY_PATH = Path(__file__).parent / "benchmark_history.json"

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

BASE_MODEL_ID = CFG["model"]["name"]
FINETUNED_DIR = CFG["model"]["output_dir"]
SYSTEM_PROMPT = CFG.get("system_prompt", "You are a helpful assistant.")
CHAT_CFG = CFG.get("chat", {})

# ── Test prompts across different categories ──

TEST_PROMPTS = [
    {"category": "Factual", "prompt": "What is photosynthesis and why is it important?"},
    {"category": "Factual", "prompt": "Explain the difference between TCP and UDP."},
    {"category": "Reasoning", "prompt": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?"},
    {"category": "Reasoning", "prompt": "A bat and a ball cost $1.10 together. The bat costs $1.00 more than the ball. How much does the ball cost?"},
    {"category": "Instruction", "prompt": "List 3 pros and 3 cons of remote work."},
    {"category": "Instruction", "prompt": "Write a short professional email declining a meeting invitation."},
    {"category": "Creative", "prompt": "Write a haiku about programming."},
    {"category": "Creative", "prompt": "Explain quantum computing to a 10-year-old."},
    {"category": "Conversational", "prompt": "What's a good beginner-friendly programming language and why?"},
    {"category": "Conversational", "prompt": "I'm feeling overwhelmed with too many tasks. Any advice?"},
    {"category": "Coding", "prompt": "Write a Python function that checks if a string is a palindrome."},
    {"category": "Coding", "prompt": "What does the 'yield' keyword do in Python?"},
]


# ── Model loading ──

def get_compute_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    elif torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_model(mode="base"):
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


# ── Metrics ──

@torch.no_grad()
def generate_response(model, tokenizer, prompt, max_new_tokens=256):
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
    return math.exp(total_loss / total_tokens)


def compute_repetition_ratio(text):
    words = text.lower().split()
    if len(words) < 5:
        return 0.0
    trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
    if not trigrams:
        return 0.0
    return 1.0 - (len(set(trigrams)) / len(trigrams))


def compute_response_quality(response):
    """Score response quality based on length and structure. 0-1 scale."""
    text = response.strip()
    if not text:
        return 0.0

    score = 0.0

    # Length score: reward 50-300 tokens, penalize very short or excessively long
    word_count = len(text.split())
    if word_count < 10:
        score += 0.1
    elif word_count < 30:
        score += 0.4
    elif word_count <= 200:
        score += 0.7
    elif word_count <= 400:
        score += 0.5
    else:
        score += 0.3

    # Structure: has sentences (periods, question marks)
    sentence_endings = text.count(".") + text.count("?") + text.count("!")
    if sentence_endings >= 2:
        score += 0.2
    elif sentence_endings >= 1:
        score += 0.1

    # Not just repeating the prompt or giving empty filler
    unique_words = len(set(text.lower().split()))
    vocab_richness = unique_words / max(word_count, 1)
    score += min(vocab_richness * 0.15, 0.1)

    return min(score, 1.0)


# ── Composite Scoring ──

def compute_composite_score(base_metrics, ft_metrics, category_scores):
    """
    Compute a 0-100 composite score for the fine-tuned model.

    Breakdown:
      Perplexity improvement:  35 pts  (lower is better)
      Coherence:               25 pts  (low repetition)
      Response quality:        20 pts  (length, structure, vocab)
      Speed:                   10 pts  (tokens/sec vs base)
      Category consistency:    10 pts  (even performance across categories)
    """
    scores = {}

    # 1. Perplexity (35 pts) — reward improvement over base
    base_ppl = base_metrics["perplexity"]
    ft_ppl = ft_metrics["perplexity"]
    if base_ppl > 0 and ft_ppl > 0 and base_ppl != float("inf"):
        ppl_ratio = ft_ppl / base_ppl
        if ppl_ratio <= 0.8:
            scores["perplexity"] = 35.0     # 20%+ improvement
        elif ppl_ratio <= 0.95:
            scores["perplexity"] = 30.0     # Solid improvement
        elif ppl_ratio <= 1.05:
            scores["perplexity"] = 22.0     # About the same
        elif ppl_ratio <= 1.2:
            scores["perplexity"] = 12.0     # Slightly worse
        else:
            scores["perplexity"] = max(0, 35 - (ppl_ratio - 1.0) * 50)
    else:
        scores["perplexity"] = 10.0

    # 2. Coherence (25 pts) — low repetition ratio
    rep = ft_metrics["avg_repetition"]
    if rep <= 0.02:
        scores["coherence"] = 25.0
    elif rep <= 0.05:
        scores["coherence"] = 22.0
    elif rep <= 0.10:
        scores["coherence"] = 18.0
    elif rep <= 0.20:
        scores["coherence"] = 12.0
    elif rep <= 0.40:
        scores["coherence"] = 6.0
    else:
        scores["coherence"] = 0.0

    # 3. Response quality (20 pts)
    avg_quality = ft_metrics.get("avg_quality", 0.5)
    scores["quality"] = avg_quality * 20.0

    # 4. Speed (10 pts) — not slower than base
    base_tps = base_metrics["avg_tokens_per_sec"]
    ft_tps = ft_metrics["avg_tokens_per_sec"]
    if base_tps > 0:
        speed_ratio = ft_tps / base_tps
        scores["speed"] = min(speed_ratio * 10.0, 10.0)
    else:
        scores["speed"] = 5.0

    # 5. Category consistency (10 pts) — even scores across categories
    if category_scores:
        cat_values = list(category_scores.values())
        if len(cat_values) >= 2:
            mean_cat = sum(cat_values) / len(cat_values)
            variance = sum((v - mean_cat) ** 2 for v in cat_values) / len(cat_values)
            std_dev = variance ** 0.5
            # Lower std = more consistent = better
            scores["consistency"] = max(0, 10.0 - std_dev * 20)
        else:
            scores["consistency"] = 5.0
    else:
        scores["consistency"] = 5.0

    total = sum(scores.values())
    return round(min(total, 100.0), 1), scores


def get_score_grade(score):
    """Convert numeric score to letter grade with color."""
    if score >= 90:
        return "[bold green]A+[/bold green]"
    elif score >= 80:
        return "[bold green]A[/bold green]"
    elif score >= 70:
        return "[green]B+[/green]"
    elif score >= 60:
        return "[green]B[/green]"
    elif score >= 50:
        return "[yellow]C+[/yellow]"
    elif score >= 40:
        return "[yellow]C[/yellow]"
    elif score >= 30:
        return "[red]D[/red]"
    else:
        return "[bold red]F[/bold red]"


# ── History ──

def load_history():
    if HISTORY_PATH.exists():
        return json.loads(HISTORY_PATH.read_text())
    return {"runs": []}


def save_history(history):
    HISTORY_PATH.write_text(json.dumps(history, indent=2, ensure_ascii=False))


def save_run(base_metrics, ft_metrics, score, score_breakdown, category_scores, tag=None):
    """Save benchmark run to history."""
    history = load_history()

    run = {
        "id": len(history["runs"]) + 1,
        "timestamp": datetime.now().isoformat(),
        "model": CFG["model"]["name"],
        "tag": tag,
        "config": {
            "max_steps": CFG["training"].get("max_steps", -1),
            "num_epochs": CFG["training"].get("num_epochs", 2),
            "learning_rate": CFG["training"].get("learning_rate", 2e-5),
            "lora_r": CFG["lora"].get("r", 16),
            "max_length": CFG["data"].get("max_length", 512),
            "batch_size": CFG["training"].get("batch_size", 1),
        },
        "score": score,
        "grade": get_score_grade(score).replace("[bold green]", "").replace("[/bold green]", "")
                     .replace("[green]", "").replace("[/green]", "")
                     .replace("[yellow]", "").replace("[/yellow]", "")
                     .replace("[red]", "").replace("[/red]", "")
                     .replace("[bold red]", "").replace("[/bold red]", ""),
        "breakdown": score_breakdown,
        "metrics": {
            "base_perplexity": base_metrics["perplexity"],
            "ft_perplexity": ft_metrics["perplexity"],
            "base_tokens_per_sec": base_metrics["avg_tokens_per_sec"],
            "ft_tokens_per_sec": ft_metrics["avg_tokens_per_sec"],
            "ft_repetition": ft_metrics["avg_repetition"],
            "ft_avg_quality": ft_metrics.get("avg_quality", 0),
        },
        "category_scores": category_scores,
    }

    history["runs"].append(run)
    save_history(history)
    return run


def show_history():
    """Display historical benchmark scores with trends."""
    history = load_history()

    if not history["runs"]:
        console.print("[yellow]No benchmark history yet. Run a benchmark first.[/yellow]")
        return

    runs = history["runs"]

    console.print(Rule("[bold green]Benchmark History[/bold green]"))
    console.print()

    # Main history table
    table = Table(box=box.ROUNDED, title="Score History", style="cyan")
    table.add_column("#", style="dim", width=4)
    table.add_column("Date", style="white", width=18)
    table.add_column("Score", justify="center", width=8)
    table.add_column("Grade", justify="center", width=7)
    table.add_column("Trend", justify="center", width=7)
    table.add_column("Tag", style="dim", max_width=30)
    table.add_column("Config", style="dim", max_width=35)

    for i, run in enumerate(runs):
        # Trend arrow
        if i == 0:
            trend = "[dim]—[/dim]"
        else:
            diff = run["score"] - runs[i-1]["score"]
            if diff > 2:
                trend = f"[green]+{diff:.1f} ↑[/green]"
            elif diff < -2:
                trend = f"[red]{diff:.1f} ↓[/red]"
            else:
                trend = f"[yellow]{diff:+.1f} =[/yellow]"

        # Score coloring
        score = run["score"]
        if score >= 70:
            score_str = f"[green]{score}[/green]"
        elif score >= 50:
            score_str = f"[yellow]{score}[/yellow]"
        else:
            score_str = f"[red]{score}[/red]"

        grade = get_score_grade(run["score"])

        # Config summary
        cfg = run.get("config", {})
        steps = cfg.get("max_steps", -1)
        lr = cfg.get("learning_rate", "?")
        r = cfg.get("lora_r", "?")
        steps_str = str(steps) if steps > 0 else f"{cfg.get('num_epochs', '?')}ep"
        config_str = f"steps={steps_str} lr={lr} r={r}"

        ts = run.get("timestamp", "")[:16].replace("T", " ")
        tag = run.get("tag", "") or ""

        table.add_row(
            str(run.get("id", i+1)),
            ts,
            score_str,
            grade,
            trend,
            tag,
            config_str,
        )

    console.print(table)

    # Score breakdown for latest run
    latest = runs[-1]
    console.print()
    breakdown_table = Table(box=box.ROUNDED, title="Latest Run Breakdown", style="cyan")
    breakdown_table.add_column("Component", style="bold white")
    breakdown_table.add_column("Score", justify="center", width=10)
    breakdown_table.add_column("Max", justify="center", style="dim", width=6)
    breakdown_table.add_column("Bar", width=22)

    component_names = {
        "perplexity": "Perplexity",
        "coherence": "Coherence",
        "quality": "Response quality",
        "speed": "Speed",
        "consistency": "Consistency",
    }
    component_max = {
        "perplexity": 35,
        "coherence": 25,
        "quality": 20,
        "speed": 10,
        "consistency": 10,
    }

    for key, name in component_names.items():
        val = latest.get("breakdown", {}).get(key, 0)
        mx = component_max[key]
        filled = int((val / mx) * 15) if mx > 0 else 0
        bar = "█" * filled + "░" * (15 - filled)
        color = "green" if val / mx >= 0.7 else "yellow" if val / mx >= 0.4 else "red"
        breakdown_table.add_row(name, f"[{color}]{val:.1f}[/{color}]", str(mx), f"[{color}]{bar}[/{color}]")

    console.print(breakdown_table)

    # Category scores for latest run
    cat_scores = latest.get("category_scores", {})
    if cat_scores:
        console.print()
        cat_table = Table(box=box.ROUNDED, title="Latest Category Scores", style="cyan")
        cat_table.add_column("Category", style="bold white")
        cat_table.add_column("Score", justify="center", width=10)
        cat_table.add_column("Bar", width=22)

        for cat, val in sorted(cat_scores.items()):
            filled = int(val * 15)
            bar = "█" * filled + "░" * (15 - filled)
            color = "green" if val >= 0.7 else "yellow" if val >= 0.4 else "red"
            cat_table.add_row(cat, f"[{color}]{val:.2f}[/{color}]", f"[{color}]{bar}[/{color}]")

        console.print(cat_table)

    # Best/worst/average
    all_scores = [r["score"] for r in runs]
    console.print()
    console.print(f"  Best:    [green]{max(all_scores):.1f}[/green] (run #{[r['score'] for r in runs].index(max(all_scores)) + 1})")
    console.print(f"  Worst:   [red]{min(all_scores):.1f}[/red] (run #{[r['score'] for r in runs].index(min(all_scores)) + 1})")
    console.print(f"  Average: [cyan]{sum(all_scores)/len(all_scores):.1f}[/cyan]")
    console.print(f"  Runs:    {len(runs)}")

    if len(runs) >= 2:
        recent_trend = all_scores[-1] - all_scores[0]
        if recent_trend > 0:
            console.print(f"  Overall: [green]+{recent_trend:.1f} improvement since first run[/green]")
        elif recent_trend < 0:
            console.print(f"  Overall: [red]{recent_trend:.1f} regression since first run[/red]")
        else:
            console.print(f"  Overall: [yellow]No change since first run[/yellow]")


# ── Main benchmark ──

def run_benchmark(num_prompts=None, tag=None):
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
        eval_texts = [p["prompt"] for p in TEST_PROMPTS]
    base_ppl = compute_perplexity(base_model, tokenizer, eval_texts)
    console.print(f"  Base perplexity: {base_ppl:.2f}")

    # ── Free base model, load finetuned ──
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

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
        result["quality"] = compute_response_quality(result["response"])
        ft_results.append(result)

    # Also compute quality for base results
    for r in base_results:
        r["quality"] = compute_response_quality(r["response"])

    # ── Finetuned perplexity ──
    console.print("\n  Computing perplexity on eval samples...")
    ft_ppl = compute_perplexity(ft_model, tokenizer, eval_texts)
    console.print(f"  Fine-tuned perplexity: {ft_ppl:.2f}")

    # ── Compute aggregate metrics ──
    base_tps = sum(r["tokens_per_sec"] for r in base_results) / len(base_results)
    ft_tps = sum(r["tokens_per_sec"] for r in ft_results) / len(ft_results)
    base_len = sum(r["tokens"] for r in base_results) / len(base_results)
    ft_len = sum(r["tokens"] for r in ft_results) / len(ft_results)
    base_rep = sum(compute_repetition_ratio(r["response"]) for r in base_results) / len(base_results)
    ft_rep = sum(compute_repetition_ratio(r["response"]) for r in ft_results) / len(ft_results)
    ft_quality = sum(r["quality"] for r in ft_results) / len(ft_results)

    base_metrics = {
        "perplexity": base_ppl,
        "avg_tokens_per_sec": base_tps,
        "avg_response_tokens": base_len,
        "avg_repetition": base_rep,
    }
    ft_metrics = {
        "perplexity": ft_ppl,
        "avg_tokens_per_sec": ft_tps,
        "avg_response_tokens": ft_len,
        "avg_repetition": ft_rep,
        "avg_quality": ft_quality,
    }

    # ── Per-category scores ──
    categories = sorted(set(r["category"] for r in ft_results))
    category_scores = {}
    for cat in categories:
        cat_results = [r for r in ft_results if r["category"] == cat]
        cat_rep = sum(compute_repetition_ratio(r["response"]) for r in cat_results) / len(cat_results)
        cat_quality = sum(r["quality"] for r in cat_results) / len(cat_results)
        # Category score: weighted combo of quality and coherence (0-1 scale)
        category_scores[cat] = round(cat_quality * 0.6 + (1 - cat_rep) * 0.4, 3)

    # ── Compute composite score ──
    score, breakdown = compute_composite_score(base_metrics, ft_metrics, category_scores)
    grade = get_score_grade(score)

    # ── Display results ──
    console.print()
    console.print(Rule("[bold green]Benchmark Results[/bold green]"))

    # Score card
    console.print()
    console.print(Panel(
        f"[bold]Score: {score}/100[/bold]  {grade}",
        title="[bold]Composite Score[/bold]",
        border_style="green" if score >= 60 else "yellow" if score >= 40 else "red",
        padding=(1, 4),
    ))

    # Breakdown
    breakdown_table = Table(box=box.ROUNDED, title="Score Breakdown", style="cyan")
    breakdown_table.add_column("Component", style="bold white", width=20)
    breakdown_table.add_column("Score", justify="center", width=10)
    breakdown_table.add_column("Max", justify="center", style="dim", width=6)
    breakdown_table.add_column("Bar", width=22)

    component_names = {
        "perplexity": "Perplexity",
        "coherence": "Coherence",
        "quality": "Response quality",
        "speed": "Speed",
        "consistency": "Consistency",
    }
    component_max = {"perplexity": 35, "coherence": 25, "quality": 20, "speed": 10, "consistency": 10}

    for key, name in component_names.items():
        val = breakdown.get(key, 0)
        mx = component_max[key]
        filled = int((val / mx) * 15) if mx > 0 else 0
        bar = "█" * filled + "░" * (15 - filled)
        color = "green" if val / mx >= 0.7 else "yellow" if val / mx >= 0.4 else "red"
        breakdown_table.add_row(name, f"[{color}]{val:.1f}[/{color}]", str(mx), f"[{color}]{bar}[/{color}]")

    console.print(breakdown_table)

    # Metrics comparison
    console.print()
    summary = Table(box=box.ROUNDED, title="Metrics Comparison", style="cyan")
    summary.add_column("Metric", style="bold white")
    summary.add_column("Base", style="yellow", justify="center")
    summary.add_column("Fine-tuned", style="green", justify="center")
    summary.add_column("Winner", justify="center")

    tps_winner = "[green]FT[/green]" if ft_tps >= base_tps else "[yellow]Base[/yellow]"
    summary.add_row("Tokens/sec", f"{base_tps:.1f}", f"{ft_tps:.1f}", tps_winner)
    summary.add_row("Avg tokens", f"{base_len:.0f}", f"{ft_len:.0f}", "—")
    ppl_winner = "[green]FT[/green]" if ft_ppl <= base_ppl else "[yellow]Base[/yellow]"
    summary.add_row("Perplexity", f"{base_ppl:.2f}", f"{ft_ppl:.2f}", ppl_winner)
    rep_winner = "[green]FT[/green]" if ft_rep <= base_rep else "[yellow]Base[/yellow]"
    summary.add_row("Repetition", f"{base_rep:.3f}", f"{ft_rep:.3f}", rep_winner)
    console.print(summary)

    # Category breakdown
    console.print()
    cat_table = Table(box=box.ROUNDED, title="Per-Category Scores", style="cyan")
    cat_table.add_column("Category", style="bold white")
    cat_table.add_column("Score", justify="center", width=8)
    cat_table.add_column("Bar", width=22)
    cat_table.add_column("Base rep.", style="yellow", justify="center")
    cat_table.add_column("FT rep.", style="green", justify="center")

    for cat in categories:
        val = category_scores[cat]
        filled = int(val * 15)
        bar = "█" * filled + "░" * (15 - filled)
        color = "green" if val >= 0.7 else "yellow" if val >= 0.4 else "red"

        b_items = [r for r in base_results if r["category"] == cat]
        f_items = [r for r in ft_results if r["category"] == cat]
        b_rep = sum(compute_repetition_ratio(r["response"]) for r in b_items) / len(b_items)
        f_rep = sum(compute_repetition_ratio(r["response"]) for r in f_items) / len(f_items)

        cat_table.add_row(cat, f"[{color}]{val:.2f}[/{color}]", f"[{color}]{bar}[/{color}]",
                          f"{b_rep:.3f}", f"{f_rep:.3f}")

    console.print(cat_table)

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
            subtitle=f"[dim]{ft_r['tokens']}tok | {ft_r['tokens_per_sec']:.1f}t/s | rep:{ft_rep_score:.2f} | q:{ft_r['quality']:.2f}[/dim]",
            border_style="green",
            width=console.width // 2 - 2,
        )

        console.print(Columns([base_panel, ft_panel], padding=1))

    # ── Save to history ──
    run = save_run(base_metrics, ft_metrics, score, breakdown, category_scores, tag=tag)
    console.print(f"\n[dim]Run #{run['id']} saved to {HISTORY_PATH}[/dim]")

    # ── Show trend if history exists ──
    history = load_history()
    if len(history["runs"]) > 1:
        prev = history["runs"][-2]
        diff = score - prev["score"]
        console.print()
        if diff > 2:
            console.print(f"[green]  ↑ Score improved by {diff:.1f} since last run[/green]")
        elif diff < -2:
            console.print(f"[red]  ↓ Score dropped by {abs(diff):.1f} since last run[/red]")
        else:
            console.print(f"[yellow]  = Score roughly the same as last run ({diff:+.1f})[/yellow]")

    console.print()
    console.print("[dim]View full history: python3 benchmark.py --history[/dim]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark with scoring and historical tracking")
    parser.add_argument("--quick", action="store_true", help="Quick mode: fewer prompts")
    parser.add_argument("--prompts", type=int, default=None, help="Number of test prompts")
    parser.add_argument("--history", action="store_true", help="View historical scores and trends")
    parser.add_argument("--tag", type=str, default=None, help="Tag this run with a note (e.g., '50 steps lr=2e-5')")
    args = parser.parse_args()

    if args.history:
        show_history()
    else:
        num = args.prompts or (4 if args.quick else None)
        run_benchmark(num_prompts=num, tag=args.tag)