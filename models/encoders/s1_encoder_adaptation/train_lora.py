import argparse
import inspect
import math
import time
import os
import random
import re
import shutil
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from datasets import Dataset

try:
    from datasets import set_seed as set_datasets_seed
except ImportError:
    set_datasets_seed = None

from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
import matplotlib.pyplot as plt

# LoRA / PEFT
try:
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError as e:
    raise ImportError("Missing dependency: peft. Install with `pip install -U peft`.") from e

from model_metadata import MODEL_METADATA

ALLOWED_MODEL_NAMES = [
    "bert-base-multilingual-cased",
    "xlm-roberta-base",
    "xlm-roberta-large",
    "mdeberta-v3-base",
    "deberta-v3-base",
]


def _fs_safe_model_name(name: str) -> str:
    """Convert a model identifier into a filesystem-safe slug."""
    normalized = (
        name.strip()
        .replace(os.sep, "-")
        .replace("/", "-")
        .replace("\\", "-")
    )
    safe = re.sub(r"[^0-9A-Za-z._-]+", "-", normalized).strip("-._")
    return safe or "model"


# -----------------------------------------------------------------------------
# 1) Setup
# -----------------------------------------------------------------------------
SEED = 42

USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 5
MAX_LENGTH = 512

TRAIN_CSV_PATH = os.path.join("datasets", "train_dataset.csv")
TEST_CSV_PATH = os.path.join("datasets", "test_dataset.csv")

TASK_ID = "t1"
TUNE_STRATEGY = "lora"
PLOTS_DIR = Path("results") / "plots" / "encoder"

FIXED_CONFIG = {
    "num_train_epochs": 20,
    "batch_size": 16,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
}

# -----------------
# LoRA configuration
# -----------------
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

TRAIN_CLASSIFIER_HEAD = True
TRAIN_POOLER = False 


def _linear_module_names(model) -> list[str]:
    return [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]


def _linear_leaf_names(model) -> set[str]:
    return {n.split(".")[-1] for n in _linear_module_names(model)}


def _count_leaf_matches(model, leaf: str) -> int:
    # PEFT matches by module name; leaf-name matching is a good proxy
    cnt = 0
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and name.split(".")[-1] == leaf:
            cnt += 1
    return cnt


def _choose_lora_targets(model) -> list[str]:
    """
    Robust target_modules selection:
    - Tries known naming conventions first
    - Verifies they exist in this specific installed model
    - Picks the first set that yields >0 matches
    """
    leafs = _linear_leaf_names(model)
    model_type = (getattr(getattr(model, "config", None), "model_type", "") or "").lower()

    # Candidate sets in preference order per family (but verified at runtime)
    if "deberta" in model_type:
        candidate_sets = [
            ["query_proj", "value_proj"],
            ["q_proj", "v_proj"],
            ["query", "value"],
            ["in_proj"],  # packed projections in some variants
        ]
    else:
        candidate_sets = [
            ["query", "value"],
            ["q_proj", "v_proj"],
            ["query_proj", "value_proj"],
            ["in_proj"],
        ]

    # Try sets that fully exist and match at least one Linear module
    for targets in candidate_sets:
        if all(t in leafs for t in targets):
            total_matches = sum(_count_leaf_matches(model, t) for t in targets)
            if total_matches > 0:
                return targets

    # Last resort: pick anything attention-ish that exists
    attentionish = [t for t in ["query", "value", "q_proj", "v_proj", "query_proj", "value_proj", "in_proj"] if t in leafs]
    if attentionish:
        total_matches = sum(_count_leaf_matches(model, t) for t in attentionish)
        if total_matches > 0:
            return attentionish

    # If we get here, we truly couldn't find anything reasonable
    sample_leafs = sorted(list(leafs))[:80]
    raise ValueError(
        "Could not find any LoRA target modules in this model. "
        f"model_type='{model_type}', linear leaf names sample={sample_leafs}"
    )


def _detect_modules_to_save(base_model) -> list[str] | None:
    """
    Ensure the classification head is trained + SAVED with the adapter.
    (Without this, the head is newly initialized and would remain random or be lost on save.)
    """
    mods: list[str] = []
    if hasattr(base_model, "classifier"):
        mods.append("classifier")
    if hasattr(base_model, "score"):
        mods.append("score")
    # Add others if your backbone uses different head attr names.
    return mods or None


def _assert_lora_injected(model) -> None:
    lora_params = [n for n, p in model.named_parameters() if "lora_" in n]
    if not lora_params:
        raise RuntimeError("LoRA injection failed: no parameters containing 'lora_' were found.")
    # Small debug sample (avoid printing huge lists)
    print(f"[Info] LoRA injection OK. Example LoRA params: {lora_params[:5]}")


def _set_trainables(model, train_head: bool = TRAIN_CLASSIFIER_HEAD, train_pooler: bool = TRAIN_POOLER) -> None:
    """
    Train LoRA params + optionally classifier head/pooler.
    NOTE: if you train the head, you must also SAVE it (we do via modules_to_save).
    """
    for n, p in model.named_parameters():
        is_lora = ("lora_" in n)
        is_head = False
        is_pool = False

        if train_head:
            # common head parameter paths
            if ".classifier." in n or n.startswith("classifier."):
                is_head = True
            if ".score." in n or n.startswith("score."):
                is_head = True

        if train_pooler:
            if ".pooler." in n or n.startswith("pooler."):
                is_pool = True

        p.requires_grad = (is_lora or is_head or is_pool)


def _build_output_slug(model_name: str, param_mode: str) -> str:
    if model_name not in MODEL_METADATA:
        raise KeyError(
            f"Model '{model_name}' missing from MODEL_METADATA. "
            "Add it to models/encoders/model_metadata.py."
        )
    meta = MODEL_METADATA[model_name]
    head_type = "defaultHead"
    # Encode the LoRA rank into the slug (e.g., lora8, lora16)
    lora_tag = f"lora{LORA_R}"
    slug = (
        f"{TASK_ID}_{meta['arch']}_{meta['lang']}_{meta['size']}"
        f"_{lora_tag}_{param_mode}_{head_type}"
    )
    return _fs_safe_model_name(slug)


TARGET_COLUMN = "clarity_label"
ARG1_KEY = "question"
ARG2_KEY = "interview_answer"


def _resolve_model_name(user_name: str) -> str:
    """Map shorthand model keys to full HF IDs and infer size."""
    name = user_name.strip()
    lower = name.lower()
    is_large = "large" in lower

    if name in MODEL_METADATA:
        return name

    if "xlm-roberta" in lower or "xlmr" in lower:
        return "FacebookAI/xlm-roberta-large" if is_large else "FacebookAI/xlm-roberta-base"
    if "roberta" in lower:
        return "roberta-large" if is_large else "roberta-base"
    if "mdeberta" in lower:
        return "microsoft/mdeberta-v3-base"
    if "deberta" in lower:
        return "microsoft/deberta-v3-large" if is_large else "microsoft/deberta-v3-base"
    if "mbert" in lower or "bert-base-multilingual" in lower:
        return "bert-base-multilingual-cased"
    if "bert" in lower:
        return "bert-base-uncased"

    return name


def _expand_models(user_name: str):
    """Return a list of model identifiers to train."""
    normalized = user_name.strip().lower()
    if normalized == "all":
        resolved = [_resolve_model_name(m) for m in ALLOWED_MODEL_NAMES]
        seen = set()
        return [m for m in resolved if m in MODEL_METADATA and not (m in seen or seen.add(m))]
    return [_resolve_model_name(user_name)]


def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)
    if set_datasets_seed is not None:
        set_datasets_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"[Info] Global seed set: {seed}")


set_global_seed(SEED)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_micro = f1_score(labels, preds, average="micro")
    f1_weighted = f1_score(labels, preds, average="weighted")
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "f1_weighted": f1_weighted,
    }


class WeightedTrainer(Trainer):
    """Trainer with class-weighted loss and a live epoch metrics table."""

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.float()
        self._epoch_cache: Dict[int, Dict[str, float]] = {}
        self._printed_header = False
        self._train_start_time: float | None = None
        self._table_header: str | None = None
        self._table_lines: list[str] = []
        self._printed_epochs: set[int] = set()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def _print_header_if_needed(self) -> None:
        if self._printed_header:
            return
        header = (
            f"{'Epoch':>5}  {'Progress%':>9}  {'ETA':>10}  {'Training Loss':>13}  {'Validation Loss':>15}  "
            f"{'Accuracy':>9}  {'F1 Macro':>9}  {'F1 Micro':>9}"
        )
        print(header)
        print("-" * len(header))
        self._table_header = header
        self._printed_header = True

    def _maybe_print_epoch_row(self, epoch_idx: int) -> None:
        if epoch_idx in self._printed_epochs:
            return
        row = self._epoch_cache.get(epoch_idx)
        if not row:
            return
        if "val_loss" not in row:
            return

        train_loss = row.get("train_loss", float("nan"))
        val_loss = row.get("val_loss", float("nan"))
        acc = row.get("accuracy", float("nan"))
        f1_macro = row.get("f1_macro", float("nan"))
        f1_micro = row.get("f1_micro", float("nan"))

        total_epochs = float(getattr(self.args, "num_train_epochs", 0.0) or 0.0)
        progress_pct = (epoch_idx / total_epochs * 100.0) if total_epochs > 0.0 else float("nan")

        eta_str = "N/A"
        if self._train_start_time is not None and total_epochs > 0.0 and epoch_idx > 0:
            elapsed = max(0.0, time.time() - self._train_start_time)
            epochs_done = float(epoch_idx)
            remaining = max(0.0, total_epochs - epochs_done)
            if epochs_done > 0.0:
                sec_per_epoch = elapsed / epochs_done
                eta_seconds = remaining * sec_per_epoch
                total_seconds = int(round(eta_seconds))
                minutes, seconds = divmod(total_seconds, 60)
                hours, minutes = divmod(minutes, 60)
                eta_str = f"{hours:d}h{minutes:02d}m" if hours > 0 else f"{minutes:02d}m{seconds:02d}s"

        self._print_header_if_needed()
        row_str = (
            f"{epoch_idx:5d}  "
            f"{progress_pct:9.2f}  "
            f"{eta_str:>10}  "
            f"{train_loss:13.6f}  "
            f"{val_loss:15.6f}  "
            f"{acc:9.6f}  "
            f"{f1_macro:9.6f}  "
            f"{f1_micro:9.6f}"
        )
        print(row_str)
        self._table_lines.append(row_str)
        self._printed_epochs.add(epoch_idx)

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        if self._train_start_time is None:
            self._train_start_time = time.time()
        if self.state.epoch is not None and "epoch" not in logs:
            logs["epoch"] = self.state.epoch
        if self.state.global_step is not None and "step" not in logs:
            logs["step"] = self.state.global_step
        self.state.log_history.append(dict(logs))

        epoch_val = logs.get("epoch")
        if epoch_val is None:
            return
        epoch_idx = _epoch_to_int(epoch_val)
        row = self._epoch_cache.setdefault(epoch_idx, {})

        if "loss" in logs:
            row["train_loss"] = float(logs["loss"])
        if "eval_loss" in logs:
            row["val_loss"] = float(logs["eval_loss"])
        if "eval_accuracy" in logs:
            row["accuracy"] = float(logs["eval_accuracy"])
        if "eval_f1_macro" in logs:
            row["f1_macro"] = float(logs["eval_f1_macro"])
        if "eval_f1_micro" in logs:
            row["f1_micro"] = float(logs["eval_f1_micro"])

        self._maybe_print_epoch_row(epoch_idx)


def _save_training_progress(trainer: Trainer, final_model_dir: str, eval_metrics: Dict[str, float] | None = None) -> None:
    if not isinstance(trainer, WeightedTrainer):
        return
    header = trainer._table_header
    lines = trainer._table_lines
    if not lines:
        return

    if eval_metrics is None:
        # Prefer the best epoch according to eval_f1_macro (or accuracy),
        # aligned with metric_for_best_model used during training.
        best_metric = getattr(getattr(trainer, "state", None), "best_metric", None)
        best_epoch_idx = None
        best_row = None

        # First try to find the epoch whose f1_macro matches best_metric.
        if best_metric is not None and isinstance(best_metric, (int, float)):
            for epoch_idx, row in trainer._epoch_cache.items():
                f1_macro = row.get("f1_macro")
                if f1_macro is None:
                    continue
                if abs(f1_macro - float(best_metric)) < 1e-8:
                    best_epoch_idx = epoch_idx
                    best_row = row
                    break

        # Fallback: pick the epoch with the highest f1_macro (or accuracy).
        if best_row is None:
            best_score = float("-inf")
            for epoch_idx, row in trainer._epoch_cache.items():
                score = row.get("f1_macro")
                if score is None:
                    score = row.get("accuracy")
                if score is None:
                    continue
                if score > best_score:
                    best_score = score
                    best_epoch_idx = epoch_idx
                    best_row = row

        if best_row is not None and best_epoch_idx is not None:
            eval_metrics = {
                "eval_loss": best_row.get("val_loss", float("nan")),
                "eval_accuracy": best_row.get("accuracy", float("nan")),
                "eval_f1_macro": best_row.get("f1_macro", float("nan")),
                "eval_f1_micro": best_row.get("f1_micro", float("nan")),
                "eval_f1_weighted": best_row.get("f1_weighted", float("nan")),
                "epoch": float(best_epoch_idx),
            }
        else:
            # Absolute fallback: use the last eval record if present.
            history = getattr(trainer.state, "log_history", None) or getattr(trainer, "log_history", [])
            if history:
                for record in reversed(history):
                    if "eval_loss" in record:
                        eval_metrics = record
                        break

    os.makedirs(final_model_dir, exist_ok=True)
    slug = os.path.basename(os.path.normpath(final_model_dir))
    out_path = os.path.join(final_model_dir, f"{slug}_training-progress.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        if header is not None:
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
        for line in lines:
            f.write(line + "\n")

        if eval_metrics:
            summary_keys = [
                "eval_loss",
                "eval_accuracy",
                "eval_f1_macro",
                "eval_f1_micro",
                "eval_f1_weighted",
                "eval_runtime",
                "eval_samples_per_second",
                "eval_steps_per_second",
            ]
            metrics_block: Dict[str, float] = {}
            for key in summary_keys:
                if key in eval_metrics:
                    metrics_block[key] = eval_metrics[key]

            # Use the epoch stored in eval_metrics if present, otherwise the trainer state.
            epoch_val = eval_metrics.get("epoch") if isinstance(eval_metrics, dict) else None
            if epoch_val is None:
                epoch_val = getattr(trainer.state, "epoch", None)
            step_val = getattr(trainer.state, "global_step", None)
            if epoch_val is not None:
                metrics_block["epoch"] = epoch_val
            if step_val is not None:
                metrics_block["step"] = step_val

            if metrics_block:
                f.write("\n")
                for key, value in metrics_block.items():
                    if isinstance(value, float):
                        f.write(f"  {key}: {value:.4f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")


def _epoch_to_int(epoch_value) -> int:
    epoch_float = float(epoch_value)
    epoch_idx = int(math.floor(epoch_float + 0.5))
    return max(1, epoch_idx)


def _remove_checkpoint_dirs(path: str) -> None:
    if not os.path.isdir(path):
        return
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path) and entry.startswith("checkpoint-"):
            shutil.rmtree(full_path, ignore_errors=True)


def _plot_loss_curves(trainer: Trainer, run_name: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    history = getattr(trainer, "log_history", None) or getattr(trainer.state, "log_history", [])
    if not history:
        return

    train_epochs, train_losses = [], []
    eval_epochs, eval_losses = [], []

    for record in history:
        epoch = record.get("epoch")
        if epoch is None:
            continue
        epoch_idx = _epoch_to_int(epoch)

        if "loss" in record:
            train_epochs.append(epoch_idx)
            train_losses.append(record["loss"])
        if "eval_loss" in record:
            eval_epochs.append(epoch_idx)
            eval_losses.append(record["eval_loss"])

    if not train_epochs and not eval_epochs:
        return

    run_name_safe = _fs_safe_model_name(run_name)

    plt.figure()
    if train_epochs:
        plt.plot(train_epochs, train_losses, marker="o", label="Train loss")
    if eval_epochs:
        plt.plot(eval_epochs, eval_losses, marker="o", label="Eval (test) loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss vs Epoch – {run_name_safe}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    all_epochs = train_epochs + eval_epochs
    if all_epochs:
        min_epoch = int(min(all_epochs))
        max_epoch = int(max(all_epochs))
        plt.xticks(range(min_epoch, max_epoch + 1))

    plt.tight_layout()
    loss_path = PLOTS_DIR / f"{run_name_safe}_loss.png"
    plt.savefig(loss_path)
    plt.close()
    print(f"[Info] Saved loss curves to {loss_path}")


# -----------------------------------------------------------------------------
# Core training function
# -----------------------------------------------------------------------------
def train_model(
    model_name: str,
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    arg1_key: str,
    arg2_key: str,
    label_col: str,
    *,
    output_dir: str,
    final_model_dir: str,
    batch_size: int,
    learning_rate: float,
    num_epochs: int,
    weight_decay: float,
):
    # Encode labels
    unique_labels = sorted(train_df[label_col].dropna().unique())
    label2id: Dict[str, int] = {lab: i for i, lab in enumerate(unique_labels)}
    id2label: Dict[int, str] = {i: lab for lab, i in label2id.items()}

    # Cleaning
    train_df = train_df.dropna(subset=[arg1_key, arg2_key, label_col]).copy()
    dev_df = dev_df.dropna(subset=[arg1_key, arg2_key, label_col]).copy()
    train_df["label"] = train_df[label_col].map(label2id)
    dev_df["label"] = dev_df[label_col].map(label2id)

    # Class weights from TRAIN set only
    label_counts = train_df["label"].value_counts().sort_index()
    class_weights = (len(train_df) / (len(label_counts) * label_counts)).sort_index()
    class_weights_tensor = torch.tensor(class_weights.to_numpy(), dtype=torch.float)

    # Tokenizer & tokenization
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    def tokenize(batch):
        return tokenizer(
            batch[arg1_key],
            batch[arg2_key],
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        )

    train_ds = Dataset.from_pandas(train_df, preserve_index=False).map(tokenize, batched=True)
    dev_ds = Dataset.from_pandas(dev_df, preserve_index=False).map(tokenize, batched=True)

    # -----------------
    # Model (LoRA)
    # -----------------
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label2id,
    )

    # Freeze everything first; then we'll re-enable LoRA + head as desired
    for p in base_model.parameters():
        p.requires_grad = False

    # Choose target_modules that ACTUALLY exist in this model install
    targets = _choose_lora_targets(base_model)
    modules_to_save = _detect_modules_to_save(base_model) if TRAIN_CLASSIFIER_HEAD else None

    print(f"[Info] LoRA target_modules chosen = {targets}")
    if modules_to_save:
        print(f"[Info] modules_to_save (trained+saved with adapter) = {modules_to_save}")

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=targets,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        modules_to_save=modules_to_save,
    )

    # Inject LoRA; if PEFT still complains, fall back to PEFT defaults
    try:
        model = get_peft_model(base_model, lora_cfg)
    except ValueError as e:
        if "No modules were targeted for adaptation" not in str(e):
            raise
        print("[Warn] target_modules did not match any layers; falling back to PEFT defaults for this backbone...")
        lora_cfg = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type=TaskType.SEQ_CLS,
            modules_to_save=modules_to_save,
        )
        model = get_peft_model(base_model, lora_cfg)

    _assert_lora_injected(model)

    # Ensure we actually train what we intend (LoRA + optional head/pooler)
    _set_trainables(model, train_head=TRAIN_CLASSIFIER_HEAD, train_pooler=TRAIN_POOLER)

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[Info] Trainable params: {trainable} / {total} ({100 * trainable / total:.4f}%)")

    # TrainingArguments
    training_kwargs = dict(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        disable_tqdm=True,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        save_total_limit=EARLY_STOPPING_PATIENCE + 2,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=weight_decay,
        seed=SEED,
        data_seed=SEED,
        fp16=torch.cuda.is_available(),
        use_safetensors=True,
    )

    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" not in ta_params and "eval_strategy" in ta_params:
        training_kwargs["eval_strategy"] = training_kwargs.pop("evaluation_strategy")
    filtered_kwargs = {k: v for k, v in training_kwargs.items() if k in ta_params}
    training_args = TrainingArguments(**filtered_kwargs)

    callbacks = (
        [EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]
        if USE_EARLY_STOPPING
        else []
    )

    trainer = WeightedTrainer(
        class_weights=class_weights_tensor,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    print("[Info] Starting LoRA training…")
    trainer.train()

    print("[Info] best ckpt:", trainer.state.best_model_checkpoint)
    print("[Info] best metric:", trainer.state.best_metric)
    print("[Info] best ckpt exists?:", os.path.exists(trainer.state.best_model_checkpoint or ""))

    _save_training_progress(trainer, final_model_dir)

    run_name = os.path.basename(os.path.normpath(output_dir)) or model_name
    _plot_loss_curves(trainer, run_name=run_name)

    print("[Info] Evaluating (using test set as eval_dataset)…")
    eval_metrics = trainer.evaluate(eval_dataset=dev_ds)
    for k, v in eval_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save adapter + (if enabled) classifier head via modules_to_save
    os.makedirs(final_model_dir, exist_ok=True)
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print(f"[Info] Saved LoRA adapter (+head if enabled) & tokenizer to {final_model_dir}")

    _remove_checkpoint_dirs(output_dir)
    return trainer, id2label, label2id, tokenizer


# -----------------------------------------------------------------------------
# Pipeline: load CSVs, train on ALL train, validate on TEST
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for encoder models (base frozen).")
    parser.add_argument(
        "--model_name",
        required=True,
        help="Base HF model key (e.g., xlmr, roberta, mbert, deberta) or 'all' to train every listed model.",
    )
    parser.add_argument("--param_mode", required=True, choices=("fixed",), help="Parameter mode to use.")
    args = parser.parse_args()

    model_names = _expand_models(args.model_name)
    config = FIXED_CONFIG

    for resolved_model in model_names:
        print(f"\n[Run] Training {resolved_model} with LoRA, mode={args.param_mode}")

        train_full_df = pd.read_csv(TRAIN_CSV_PATH)
        train_full_df = train_full_df.dropna(subset=[ARG1_KEY, ARG2_KEY, TARGET_COLUMN]).copy()

        if not os.path.exists(TEST_CSV_PATH):
            raise FileNotFoundError(
                f"[Fatal] Test CSV not found at {TEST_CSV_PATH}. "
                "We now require it for eval_loss / early stopping."
            )
        test_full_df = pd.read_csv(TEST_CSV_PATH)
        test_full_df = test_full_df.dropna(subset=[ARG1_KEY, ARG2_KEY, TARGET_COLUMN]).copy()

        slug = _build_output_slug(resolved_model, args.param_mode)
        output_dir = os.path.join("results", "models", slug)
        final_model_dir = os.path.join("models", "encoders", "trained_models", slug)

        trainer, _, _, _ = train_model(
            model_name=resolved_model,
            train_df=train_full_df,
            dev_df=test_full_df,
            arg1_key=ARG1_KEY,
            arg2_key=ARG2_KEY,
            label_col=TARGET_COLUMN,
            output_dir=output_dir,
            final_model_dir=final_model_dir,
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            num_epochs=config["num_train_epochs"],
            weight_decay=config["weight_decay"],
        )

        eval_metrics = trainer.evaluate()
        acc = float(eval_metrics.get("eval_accuracy", 0.0))
        f1_macro = float(eval_metrics.get("eval_f1_macro", 0.0))
        f1_micro = float(eval_metrics.get("eval_f1_micro", 0.0))
        f1_weighted = float(eval_metrics.get("eval_f1_weighted", 0.0))

        print(f"\n{resolved_model} on DEV RESULTS:")
        print(f"Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
        print(f"F1-Macro: {f1_macro:.4f}")
        print(f"F1-Micro: {f1_micro:.4f}")
        print(f"F1-Weighted: {f1_weighted:.4f}")
        print(f"[Info] Training run complete for {resolved_model}. Skipping post-training prediction/report generation.")


if __name__ == "__main__":
    main()
