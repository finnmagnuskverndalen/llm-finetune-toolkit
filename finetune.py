"""
Fine-tune small language models with QLoRA.

Key fixes over the original:
  1. Learning rate reduced from 1e-4 to 2e-5 (the #1 reason models got dumber)
  2. max_length raised from 256 to 1024 (model couldn't learn real answers)
  3. Brevity filter relaxed from 300 to 1500 chars (was discarding all good data)
  4. LoRA rank increased from 8 to 16 (more capacity to learn without forgetting)
  5. Added eval split to detect overfitting
  6. Fixed data pipeline None-filter bug
  7. Added gradient clipping (max_grad_norm=0.3)
  8. Added NEFTune noise for better generalization
  9. Auto-detects LoRA target modules per model architecture
  10. Config-driven — edit config.yaml, not this file
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import logging
import threading
import time
import yaml
import torch
import psutil
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, concatenate_datasets, DatasetDict
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

logging.basicConfig(level=logging.WARNING)
console = Console()

# ── Load Config ──────────────────────────────────────────────
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

MODEL_ID = CFG["model"]["name"]
OUTPUT_DIR = CFG["model"]["output_dir"]

# ── Dashboard State ──────────────────────────────────────────

training_state = {
    "step": 0,
    "total_steps": 0,
    "loss": None,
    "eval_loss": None,
    "best_eval_loss": float("inf"),
    "epoch": 0.0,
    "phase": "Starting up...",
    "logs": [],
    "done": False,
}


def add_log(msg, style="white"):
    ts = time.strftime("%H:%M:%S")
    training_state["logs"].append((ts, msg, style))
    if len(training_state["logs"]) > 14:
        training_state["logs"].pop(0)


def make_bar(value, max_value, width=20, fill="█", empty="░"):
    filled = int((value / max_value) * width) if max_value > 0 else 0
    return fill * filled + empty * (width - filled)


def build_dashboard():
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory()
    ram_used = ram.used / (1024**3)
    ram_total = ram.total / (1024**3)

    gpu_available = False
    try:
        if torch.cuda.is_available():
            gpu_available = True
            gpu_mem = torch.cuda.memory_allocated() / (1024**3)
            gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_name = torch.cuda.get_device_properties(0).name
            gpu_util = f"{gpu_mem:.1f}/{gpu_total:.1f} GB"
    except Exception:
        pass

    layout = Table.grid(padding=1)
    layout.add_column(justify="left", min_width=50)
    layout.add_column(justify="left", min_width=42)

    # System monitor
    stats = Table(box=box.ROUNDED, title="⚙ System Monitor", style="cyan", min_width=48)
    stats.add_column("Resource", style="bold white", width=10)
    stats.add_column("Usage", width=22)
    stats.add_column("Info", style="dim white", width=14)

    cpu_color = "green" if cpu < 60 else "yellow" if cpu < 85 else "red"
    stats.add_row("CPU", f"[{cpu_color}]{make_bar(cpu, 100)}[/{cpu_color}]", f"{cpu:.1f}%")
    ram_color = "green" if ram.percent < 60 else "yellow" if ram.percent < 85 else "red"
    stats.add_row("RAM", f"[{ram_color}]{make_bar(ram_used, ram_total)}[/{ram_color}]", f"{ram_used:.1f}/{ram_total:.1f}GB")

    if gpu_available:
        gpu_bar = make_bar(gpu_mem, gpu_total)
        stats.add_row("GPU MEM", f"[magenta]{gpu_bar}[/magenta]", gpu_util)
        stats.add_row("GPU", f"[magenta]{gpu_name[:20]}[/magenta]", "active")
    else:
        stats.add_row("GPU", "[red]░░░░░░░░░░░░░░░░░░░░[/red]", "[red]not used[/red]")

    # Training metrics
    train = Table(box=box.ROUNDED, title="🧠 Training", style="magenta", min_width=40)
    train.add_column("Metric", style="bold white", width=12)
    train.add_column("Value", style="yellow", width=26)

    step = training_state["step"]
    total = training_state["total_steps"]
    loss = training_state["loss"]
    eval_loss = training_state["eval_loss"]
    epoch = training_state["epoch"]
    phase = training_state["phase"]

    if total > 0:
        pct = (step / total) * 100
        train_bar = make_bar(step, total, width=22)
        train.add_row("Progress", f"[green]{train_bar}[/green]")
        train.add_row("Steps", f"{step}/{total} ({pct:.1f}%)")
    else:
        train.add_row("Progress", "[dim]waiting...[/dim]")
        train.add_row("Steps", "[dim]—[/dim]")

    train.add_row("Train Loss", f"{loss:.4f}" if loss else "[dim]—[/dim]")
    train.add_row("Eval Loss", f"{eval_loss:.4f}" if eval_loss else "[dim]—[/dim]")
    best = training_state["best_eval_loss"]
    train.add_row("Best Eval", f"{best:.4f}" if best < float("inf") else "[dim]—[/dim]")
    train.add_row("Epoch", f"{epoch:.3f}")
    train.add_row("Phase", f"[cyan]{phase}[/cyan]")
    train.add_row("Device", "[red]CPU[/red]" if not gpu_available else "[green]GPU[/green]")

    layout.add_row(stats, train)

    # Logs
    log_lines = Text()
    for ts, msg, style in training_state["logs"]:
        log_lines.append(f"[{ts}] ", style="dim")
        log_lines.append(msg + "\n", style=style)

    log_panel = Panel(log_lines, title="📋 Logs", border_style="blue", height=18)

    grid = Table.grid()
    grid.add_column()
    grid.add_row(Panel(layout, title=f"[bold green]Fine-tuning: {MODEL_ID}[/bold green]", border_style="green"))
    grid.add_row(log_panel)
    return grid


# ── Callbacks ────────────────────────────────────────────────

class RichCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        training_state["total_steps"] = state.max_steps
        training_state["phase"] = "Training"
        add_log(f"Training started — {state.max_steps} steps", "green")

    def on_step_end(self, args, state, control, **kwargs):
        training_state["step"] = state.global_step
        training_state["epoch"] = state.epoch or 0.0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if "loss" in logs:
                training_state["loss"] = logs["loss"]
                add_log(f"Step {state.global_step} | loss={logs['loss']:.4f}", "yellow")
            if "eval_loss" in logs:
                eval_l = logs["eval_loss"]
                training_state["eval_loss"] = eval_l
                if eval_l < training_state["best_eval_loss"]:
                    training_state["best_eval_loss"] = eval_l
                    add_log(f"★ New best eval loss: {eval_l:.4f}", "bold green")
                else:
                    add_log(f"Eval loss: {eval_l:.4f}", "cyan")

    def on_save(self, args, state, control, **kwargs):
        add_log(f"Checkpoint saved at step {state.global_step}", "cyan")

    def on_train_end(self, args, state, control, **kwargs):
        training_state["phase"] = "Done!"
        training_state["done"] = True
        add_log("Training complete!", "bold green")


def monitor_loop(live):
    while not training_state["done"]:
        live.update(build_dashboard())
        time.sleep(1)
    live.update(build_dashboard())


# ── Model Architecture Detection ─────────────────────────────

# Maps model architecture names to their linear projection module names.
# These are the layers where LoRA adapters get injected.
ARCH_TARGET_MODULES = {
    "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "phi3": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    "phi": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
    "gemma": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gemma2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "smollm": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


def detect_target_modules(model):
    """Auto-detect LoRA target modules from model architecture."""
    arch = model.config.model_type.lower()
    for key, modules in ARCH_TARGET_MODULES.items():
        if key in arch:
            return modules
    # Fallback: find all Linear layers
    add_log(f"Unknown arch '{arch}', scanning for Linear layers...", "yellow")
    names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Get just the attribute name (e.g., "q_proj" from "model.layers.0.self_attn.q_proj")
            short = name.split(".")[-1]
            if short not in ("lm_head",):
                names.add(short)
    return list(names)


# ── Data Pipeline ────────────────────────────────────────────

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
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": output},
    ]
    return {"messages": messages}


def format_messages(example, system_prompt):
    """Handle datasets that already have a 'messages' column."""
    msgs = example.get("messages", None)
    if msgs and isinstance(msgs, list) and len(msgs) >= 2:
        # Ensure system prompt is present
        if msgs[0].get("role") != "system":
            msgs = [{"role": "system", "content": system_prompt}] + msgs
        return {"messages": msgs}
    return {"messages": None}


def detect_and_format(example, system_prompt):
    """Auto-detect dataset format and convert to messages."""
    # Already has messages
    if "messages" in example and example["messages"]:
        return format_messages(example, system_prompt)
    # Guanaco-style
    if "text" in example and "### Human:" in str(example.get("text", "")):
        return format_guanaco(example, system_prompt)
    # Alpaca-style
    if "instruction" in example and "output" in example:
        return format_alpaca(example, system_prompt)
    return {"messages": None}


def load_and_prepare_datasets(tokenizer, cfg):
    """Load, format, filter, split, and template all datasets."""
    system_prompt = cfg["system_prompt"]
    data_cfg = cfg["data"]
    all_datasets = []

    for ds_cfg in cfg["datasets"]:
        ds_name = ds_cfg["name"]
        ds_split = ds_cfg.get("split", "train")
        max_samples = ds_cfg.get("max_samples", None)

        add_log(f"Loading {ds_name}...", "cyan")
        ds = load_dataset(ds_name, split=ds_split)

        if max_samples and max_samples < len(ds):
            ds = ds.select(range(max_samples))
            add_log(f"  Selected {max_samples} samples", "dim white")

        # Format to messages
        ds = ds.map(
            lambda ex: detect_and_format(ex, system_prompt),
            desc=f"Formatting {ds_name.split('/')[-1]}",
        )

        # FIXED: Properly filter out None messages
        before = len(ds)
        ds = ds.filter(lambda x: x["messages"] is not None and len(x["messages"]) >= 3)
        after = len(ds)
        add_log(f"  {ds_name.split('/')[-1]}: {after}/{before} valid examples", "white")

        all_datasets.append(ds)

    combined = concatenate_datasets(all_datasets)

    # ── Quality filters ──
    min_chars = data_cfg.get("min_assistant_chars", 10)
    max_chars = data_cfg.get("max_assistant_chars", 1500)

    def quality_filter(example):
        for msg in example["messages"]:
            if msg["role"] == "assistant":
                length = len(msg["content"])
                if length < min_chars or length > max_chars:
                    return False
        return True

    before = len(combined)
    combined = combined.filter(quality_filter, desc="Quality filter")
    after = len(combined)
    add_log(f"Quality filter: {after}/{before} kept ({min_chars}-{max_chars} chars)", "white")

    combined = combined.shuffle(seed=42)

    # ── Compute token length statistics ──
    sample_size = min(500, len(combined))
    sample = combined.select(range(sample_size))
    token_lengths = []
    for ex in sample:
        text = tokenizer.apply_chat_template(ex["messages"], tokenize=False)
        tokens = tokenizer(text, truncation=False)["input_ids"]
        token_lengths.append(len(tokens))

    avg_len = sum(token_lengths) / len(token_lengths)
    max_len = max(token_lengths)
    p95_len = sorted(token_lengths)[int(0.95 * len(token_lengths))]
    add_log(f"Token stats — avg: {avg_len:.0f}, p95: {p95_len}, max: {max_len}", "white")

    effective_max = data_cfg.get("max_length", 1024)
    truncated = sum(1 for l in token_lengths if l > effective_max)
    if truncated > 0:
        pct = (truncated / len(token_lengths)) * 100
        add_log(f"  ⚠ ~{pct:.1f}% will be truncated at {effective_max} tokens", "yellow")

    # ── Apply chat template ──
    def apply_template(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return {"text": text}

    combined = combined.map(apply_template, remove_columns=combined.column_names, desc="Templating")

    # ── Train/eval split ──
    eval_ratio = data_cfg.get("eval_split", 0.05)
    if eval_ratio > 0:
        split = combined.train_test_split(test_size=eval_ratio, seed=42)
        add_log(f"Split: {len(split['train'])} train / {len(split['test'])} eval", "green")
        return split["train"], split["test"]
    else:
        add_log(f"Dataset: {len(combined)} examples (no eval split)", "green")
        return combined, None


# ── Main ─────────────────────────────────────────────────────

def main():
    with Live(build_dashboard(), refresh_per_second=1, screen=False) as live:

        # ── Tokenizer ──
        training_state["phase"] = "Loading tokenizer"
        add_log(f"Model: {MODEL_ID}", "bold cyan")
        live.update(build_dashboard())

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        add_log("Tokenizer loaded", "green")

        # ── Model ──
        training_state["phase"] = "Loading model"
        add_log("Loading model with 4-bit quantization...", "cyan")
        live.update(build_dashboard())

        quant_cfg = CFG["quantization"]
        # Auto-select best compute dtype
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
            use_bf16 = True
        elif torch.cuda.is_available():
            compute_dtype = torch.float16
            use_bf16 = False
        else:
            compute_dtype = torch.float32
            use_bf16 = False

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_cfg.get("load_in_4bit", True),
            bnb_4bit_use_double_quant=quant_cfg.get("use_double_quant", True),
            bnb_4bit_quant_type=quant_cfg.get("quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model.config.use_cache = False
        add_log(f"Model loaded — dtype: {compute_dtype}", "green")

        # ── LoRA ──
        training_state["phase"] = "Applying LoRA"
        add_log("Configuring LoRA adapters...", "cyan")
        live.update(build_dashboard())

        lora_cfg = CFG["lora"]
        target_modules = detect_target_modules(model)
        add_log(f"Target modules: {target_modules}", "dim white")

        lora_config = LoraConfig(
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("alpha", 32),
            target_modules=target_modules,
            lora_dropout=lora_cfg.get("dropout", 0.05),
            bias=lora_cfg.get("bias", "none"),
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

        # CRITICAL: Required for gradient checkpointing with PEFT
        model.enable_input_require_grads()

        trainable, total = model.get_nb_trainable_parameters()
        add_log(f"LoRA: {trainable/1e6:.2f}M trainable / {total/1e6:.1f}M total ({trainable/total*100:.2f}%)", "green")

        # ── Data ──
        training_state["phase"] = "Loading datasets"
        live.update(build_dashboard())

        train_dataset, eval_dataset = load_and_prepare_datasets(tokenizer, CFG)

        # ── Training config ──
        train_cfg = CFG["training"]
        data_cfg = CFG["data"]

        max_steps = train_cfg.get("max_steps", -1)  # -1 means use num_epochs

        sft_args = SFTConfig(
            output_dir=OUTPUT_DIR,
            max_steps=max_steps,
            num_train_epochs=train_cfg.get("num_epochs", 2),
            per_device_train_batch_size=train_cfg.get("batch_size", 2),
            per_device_eval_batch_size=train_cfg.get("batch_size", 2),
            gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
            gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim=train_cfg.get("optim", "paged_adamw_8bit"),
            learning_rate=train_cfg.get("learning_rate", 2e-5),
            warmup_ratio=train_cfg.get("warmup_ratio", 0.06),
            lr_scheduler_type=train_cfg.get("lr_scheduler", "cosine"),
            max_grad_norm=train_cfg.get("max_grad_norm", 0.3),
            weight_decay=train_cfg.get("weight_decay", 0.01),
            fp16=(not use_bf16 and torch.cuda.is_available()),
            bf16=use_bf16,
            logging_steps=train_cfg.get("logging_steps", 5),
            logging_strategy="steps",
            save_steps=train_cfg.get("save_steps", 200),
            save_total_limit=2,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=train_cfg.get("eval_steps", 50) if eval_dataset else None,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False if eval_dataset else None,
            report_to=train_cfg.get("report_to", "none"),
            max_length=data_cfg.get("max_length", 1024),
            dataset_text_field="text",
            neftune_noise_alpha=train_cfg.get("neftune_noise_alpha", None),
        )

        # ── Start training ──
        monitor_thread = threading.Thread(target=monitor_loop, args=(live,), daemon=True)
        monitor_thread.start()

        trainer = SFTTrainer(
            model=model,
            args=sft_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[RichCallback()],
        )

        add_log("Starting training...", "bold cyan")
        trainer.train()

        # ── Save ──
        training_state["phase"] = "Saving model"
        add_log("Saving LoRA adapters...", "cyan")
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

        # Save config alongside model for reproducibility
        import shutil
        shutil.copy2(CONFIG_PATH, Path(OUTPUT_DIR) / "training_config.yaml")

        add_log(f"Saved to {OUTPUT_DIR}", "bold green")
        training_state["done"] = True
        time.sleep(2)

    console.print()
    console.print(f"[bold green]✓ Fine-tuning complete![/bold green]")
    console.print(f"[cyan]  Adapters saved to: {OUTPUT_DIR}[/cyan]")
    console.print(f"[cyan]  Run [bold]python merge.py[/bold] to export a merged model[/cyan]")
    console.print(f"[cyan]  Run [bold]python chat.py[/bold] to test[/cyan]")


if __name__ == "__main__":
    main()