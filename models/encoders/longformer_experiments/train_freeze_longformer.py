# stage4_train_freeze_longformer.py
# Train 3 fixed runs with different Longformer attention windows: 612, 768, 1024
# - Fixed hyperparams: lr=2e-5, wd=0.01, warmup_ratio=0.06
# - Fixed ablation choices (BEST): multilayer head, CTX+Q+A, CLS-only global attention
# - Saves ONLY best model (by eval_f1_macro) for each window
# - Writes *_training-progress.txt that includes best snapshot + best-restored metrics
# - Produces a summary CSV across windows

import sys, os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

try:
    from models.encoders.model_metadata import MODEL_METADATA
except Exception:
    MODEL_METADATA = {}

import argparse
import inspect
import math
import time
import random
import re
import shutil
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from datasets import Dataset

try:
    from datasets import set_seed as set_datasets_seed
except ImportError:
    set_datasets_seed = None

from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)
from transformers.utils import logging as hf_logging
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Fixed config (single source of truth)
# -----------------------------------------------------------------------------
FIXED_GRID_CONFIG = {
    # Model / tokenizer
    "tokenizer": "allenai/longformer-base-4096",
    "max_sequence_length": 2048,

    # ===== Fixed ablation choices (BEST) =====
    "classification_head": "multilayer",
    "input_order": "context+question+answer",
    "global_attention": "cls_only",
    # ========================================

    # Optimization
    "optimizer": "AdamW",
    "learning_rate": 2e-5,
    "weight_decay": 0.01,

    # Warmup (fixed)
    "lr_scheduler_type": "linear",
    "warmup_ratio": 0.06,

    # Regularization & loss
    "dropout": 0.1,
    "loss": "WCE",

    # Batch & training control
    "per_device_batch_size": 1,
    "gradient_accumulation_steps": 16,  # effective batch size = 16
    "early_stopping_patience": 7,

    # Epochs (fixed)
    "num_train_epochs": 20,

    # Fine-tuning strategy (partial unfreeze)
    "unfreeze_ratio": 0.25,
}

ATTENTION_WINDOWS_TO_RUN = [612, 768, 1024]  # requested


# -----------------------------------------------------------------------------
# Dataset paths / constants
# -----------------------------------------------------------------------------
SEED = 42
USE_EARLY_STOPPING = True

TRAIN_CSV_PATH = os.path.join("datasets", "train_dataset.csv")
TEST_CSV_PATH = os.path.join("datasets", "test_dataset.csv")
AUGMENTED_TRAIN_CSV_PATH = os.path.join(
    "datasets",
    "augmented_dataset",
    "train_dataset_augmented_filtered.csv",
)

TARGET_COLUMN = "clarity_label"
Q_COL = "question"
CTX_COL = "interview_question"
A_COL = "interview_answer"

TASK_ID = "t1"
TUNE_STRATEGY = "partial"

hf_logging.set_verbosity_error()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _fs_safe_model_name(name: str) -> str:
    normalized = (
        name.strip()
        .replace(os.sep, "-")
        .replace("/", "-")
        .replace("\\", "-")
    )
    safe = re.sub(r"[^0-9A-Za-z._-]+", "-", normalized).strip("-._")
    return safe or "model"


def _get_meta_fallback(model_name: str) -> Dict[str, str]:
    if model_name in MODEL_METADATA:
        return MODEL_METADATA[model_name]
    return {"arch": "longformer", "lang": "en", "size": "base"}


def _build_output_slug(
    model_name: str,
    truncation: str,
    dataset: str,
    max_length: int,
    attention_window: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    dropout: float,
    classification_head: str,
    input_order: str,
    global_attention: str,
) -> str:
    meta = _get_meta_fallback(model_name)

    raw_trunc = (truncation or "").strip()
    if raw_trunc == "head_tail":
        trunc_tag = "head-tail"
    elif raw_trunc:
        trunc_tag = raw_trunc
    else:
        trunc_tag = "notrunc"

    head_tag = "defaultHead" if classification_head == "default" else "multilayerHead"

    if input_order == "context+question+answer":
        order_tag = "CTX-Q-A"
    elif input_order == "question+context+answer":
        order_tag = "Q-CTX-A"
    else:
        order_tag = "Q-CTX-A-Q"

    gtag = "cls" if global_attention == "cls_only" else "clsQ"
    dataset_tag = "originalData" if dataset == "original" else "augmentedData"

    lr_tag = f"{learning_rate:.0e}".replace("+", "")
    wd_tag = f"{weight_decay:.3f}".rstrip("0").rstrip(".")
    wu_tag = f"{warmup_ratio:.3f}".rstrip("0").rstrip(".")

    slug = (
        f"{TASK_ID}_{meta['arch']}_{meta['lang']}_{meta['size']}"
        f"_{TUNE_STRATEGY}_fixed_{head_tag}_{trunc_tag}_{dataset_tag}"
        f"_{order_tag}_gattn{gtag}_L{int(max_length)}_aw{int(attention_window)}"
        f"_lr{lr_tag}_wd{wd_tag}_wu{wu_tag}_do{dropout:.2f}"
    )
    return _fs_safe_model_name(slug)


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


def _epoch_to_int(epoch_value) -> int:
    try:
        epoch_float = float(epoch_value)
    except (TypeError, ValueError):
        return 1
    epoch_idx = int(math.floor(epoch_float + 1e-9))
    return max(1, epoch_idx)


def _format_eta_mmss(total_seconds: float) -> str:
    try:
        total_seconds = float(total_seconds)
    except (TypeError, ValueError):
        return "N/A"
    total_seconds = max(0.0, total_seconds)
    sec = int(round(total_seconds))
    minutes, seconds = divmod(sec, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    return f"{minutes:02d}m{seconds:02d}s"


def _print_metrics_block(title: str, metrics: Optional[Dict[str, float]], *, step: Optional[int] = None) -> None:
    print(f"\n[{title}]")
    if not metrics:
        print("  (no metrics)")
        return

    ordered_keys = [
        "eval_loss",
        "eval_accuracy",
        "eval_f1_macro",
        "eval_f1_micro",
        "eval_f1_weighted",
        "eval_runtime",
        "eval_samples_per_second",
        "eval_steps_per_second",
        "epoch",
    ]
    for key in ordered_keys:
        if key not in metrics:
            continue
        val = metrics[key]
        if isinstance(val, float):
            print(f"  {key}: {val:.6f}")
        else:
            print(f"  {key}: {val}")

    if step is not None:
        print(f"  step: {step}")


def _write_metrics_block(f, title: str, metrics: Optional[Dict[str, float]], *, step: Optional[int] = None) -> None:
    f.write(f"\n[{title}]\n")
    if not metrics:
        f.write("  (no metrics)\n")
        return

    ordered_keys = [
        "eval_loss",
        "eval_accuracy",
        "eval_f1_macro",
        "eval_f1_micro",
        "eval_f1_weighted",
        "eval_runtime",
        "eval_samples_per_second",
        "eval_steps_per_second",
        "epoch",
    ]
    for key in ordered_keys:
        if key not in metrics:
            continue
        val = metrics[key]
        if isinstance(val, float):
            f.write(f"  {key}: {val:.6f}\n")
        else:
            f.write(f"  {key}: {val}\n")

    if step is not None:
        f.write(f"  step: {step}\n")


def _effective_len(max_length: int, attention_window: int) -> int:
    rem = max_length % attention_window
    if rem == 0:
        return max_length
    return max_length + (attention_window - rem)


# -----------------------------------------------------------------------------
# MLP head (multilayer)
# -----------------------------------------------------------------------------
class MLPHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int = 3, dropout: float = 0.1):
        super().__init__()
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_labels)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(pooled)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout2(x)
        return self.fc2(x)


class SequenceMLPHead(nn.Module):
    """
    Supports both pooled_output [B,H] and sequence_output [B,L,H].
    """
    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = MLPHead(hidden_size=hidden_size, num_labels=num_labels, dropout=dropout)

    def forward(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        if features.dim() == 3:
            pooled = features[:, 0, :]
        else:
            pooled = features
        return self.mlp(pooled)


# -----------------------------------------------------------------------------
# Early-stopping + best-restore callback
# -----------------------------------------------------------------------------
class BestModelInMemoryCallback(TrainerCallback):
    def __init__(self, metric_name: str = "eval_f1_macro", patience: int = 5, greater_is_better: bool = True):
        self.metric_name = metric_name
        self.patience = int(patience)
        self.greater_is_better = bool(greater_is_better)

        self.best_metric: Optional[float] = None
        self.best_epoch: Optional[float] = None
        self.best_step: Optional[int] = None
        self.best_metrics: Optional[Dict[str, float]] = None
        self.best_state_dict: Optional[Dict[str, torch.Tensor]] = None

        self._bad_epochs = 0

    def _is_better(self, value: float, best: Optional[float]) -> bool:
        if best is None:
            return True
        return value > best if self.greater_is_better else value < best

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or self.metric_name not in metrics:
            return control

        score = float(metrics[self.metric_name])

        if self._is_better(score, self.best_metric):
            self.best_metric = score
            self.best_epoch = state.epoch
            self.best_step = int(state.global_step) if state.global_step is not None else None
            self.best_metrics = dict(metrics)

            model = kwargs.get("model", None)
            if model is not None:
                sd = model.state_dict()
                self.best_state_dict = {k: v.detach().cpu().clone() for k, v in sd.items()}

            self._bad_epochs = 0
        else:
            self._bad_epochs += 1

        if self.patience > 0 and USE_EARLY_STOPPING and self._bad_epochs >= self.patience:
            print(
                f"[EarlyStop] No improvement in {self.metric_name} for "
                f"{self._bad_epochs} evals (patience={self.patience}). Stopping."
            )
            control.should_training_stop = True

        return control

    def restore_best(self, model: nn.Module) -> bool:
        if self.best_state_dict is None:
            return False
        model.load_state_dict(self.best_state_dict, strict=True)
        return True


# -----------------------------------------------------------------------------
# WeightedTrainer + epoch table
# -----------------------------------------------------------------------------
class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.float()
        self._epoch_cache: Dict[int, Dict[str, float]] = {}
        self._printed_header = False
        self._train_start_time: Optional[float] = None

        self._table_header: Optional[str] = None
        self._table_divider: Optional[str] = None
        self._table_lines: List[str] = []
        self._printed_epochs: set[int] = set()

        self.freeze_summary_lines: List[str] = []
        self.last_eval_metrics: Optional[Dict[str, float]] = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        if labels is None:
            labels = inputs.get("label")

        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def _print_header_if_needed(self) -> None:
        if self._printed_header:
            return
        header = (
            f"{'Epoch':>5}  {'Progress%':>9}  {'ETA':>10}  {'Training Loss':>13}  "
            f"{'Validation Loss':>15}  {'Accuracy':>9}  {'F1 Macro':>9}  {'F1 Micro':>9}"
        )
        divider = "-" * len(header)
        print(header)
        print(divider)
        self._table_header = header
        self._table_divider = divider
        self._printed_header = True

    def _maybe_print_epoch_row(self, epoch_idx: int) -> None:
        if epoch_idx in self._printed_epochs:
            return

        row = self._epoch_cache.get(epoch_idx)
        if not row or "val_loss" not in row:
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
                eta_str = _format_eta_mmss(remaining * (elapsed / epochs_done))

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


# -----------------------------------------------------------------------------
# Freezing / partial unfreezing logic + summary
# -----------------------------------------------------------------------------
def _get_base_model(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "base_model") and getattr(model, "base_model") is not None:
        return getattr(model, "base_model")
    for attr in ("longformer", "bert", "roberta", "deberta", "xlm_roberta"):
        if hasattr(model, attr):
            return getattr(model, attr)
    return model


def apply_partial_unfreezing(model: torch.nn.Module, unfreeze_ratio: float) -> List[str]:
    lines: List[str] = []

    try:
        unfreeze_ratio = float(unfreeze_ratio)
    except (TypeError, ValueError):
        unfreeze_ratio = 1.0
    unfreeze_ratio = max(0.0, min(1.0, unfreeze_ratio))

    base = _get_base_model(model)

    if hasattr(base, "embeddings"):
        for param in base.embeddings.parameters():
            param.requires_grad = False
        lines.append("[Freeze] Embedding layer parameters frozen.")
    else:
        lines.append("⚠️ Could not apply embedding freezing. No 'embeddings' on base model.")

    encoder_layers = None
    if hasattr(base, "encoder") and hasattr(base.encoder, "layer"):
        encoder_layers = list(base.encoder.layer)

    if encoder_layers is None:
        lines.append("⚠️ Could not apply encoder layer freezing. Unexpected model structure.")
    else:
        total_layers = len(encoder_layers)
        if total_layers == 0:
            lines.append("⚠️ Encoder has no layers; skipping encoder freezing.")
        else:
            unfreeze_start = int((1.0 - unfreeze_ratio) * total_layers)
            unfreeze_start = max(0, min(unfreeze_start, total_layers))

            for i, layer in enumerate(encoder_layers):
                requires_grad = (i >= unfreeze_start) and (unfreeze_ratio > 0.0)
                for param in layer.parameters():
                    param.requires_grad = requires_grad

            lines.append(f"*_Total encoder layers: {total_layers}")
            if unfreeze_ratio <= 0.0:
                lines.append("*_All layers frozen")
            elif unfreeze_start == 0:
                lines.append(f"*_Unfreezing from layer 0 to {total_layers - 1}")
            else:
                lines.append(f"*_Unfreezing from layer {unfreeze_start} to {total_layers - 1}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = 100.0 * trainable / total if total > 0 else 0.0

    lines.append(f"*_Total parameters:       {total:,}")
    lines.append(f"*__Trainable parameters:  {trainable:,}")
    lines.append(f"*__Trainable ratio:       {ratio:.2f}%")

    grouped: Dict[str, List[str]] = defaultdict(list)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if ".encoder.layer." in name:
            parts = name.split(".")
            if "layer" in parts:
                idx = parts.index("layer")
                top_group = ".".join(parts[:idx + 2])
            else:
                top_group = ".".join(name.split(".")[:2])
        else:
            top_group = ".".join(name.split(".")[:2])
        grouped[top_group].append(name)

    for group, names in sorted(grouped.items()):
        lines.append(f"✅ {group}: ({len(names)} params)")

    return lines


# -----------------------------------------------------------------------------
# Progress saving + plots
# -----------------------------------------------------------------------------
def _save_training_progress(
    trainer: Trainer,
    final_model_dir: str,
    best_info: Optional[Dict[str, Any]] = None,
    last_eval_metrics: Optional[Dict[str, float]] = None,
    best_eval_metrics: Optional[Dict[str, float]] = None,
) -> None:
    if not isinstance(trainer, WeightedTrainer):
        return

    header = trainer._table_header
    divider = trainer._table_divider
    lines = trainer._table_lines
    if not lines:
        return

    os.makedirs(final_model_dir, exist_ok=True)
    slug = os.path.basename(os.path.normpath(final_model_dir))
    out_path = os.path.join(final_model_dir, f"{slug}_training-progress.txt")

    step_val = getattr(trainer.state, "global_step", None)
    try:
        step_int = int(step_val) if step_val is not None else None
    except (TypeError, ValueError):
        step_int = None

    with open(out_path, "w", encoding="utf-8") as f:
        if header is not None:
            f.write(header + "\n")
        if divider is not None:
            f.write(divider + "\n")
        else:
            f.write("-" * (len(header) if header else 80) + "\n")

        for line in lines:
            f.write(line + "\n")

        if best_info:
            f.write("\n[Best model snapshot (selected by eval_f1_macro)]\n")
            f.write(f"  best_epoch: {best_info.get('best_epoch')}\n")
            f.write(f"  best_step: {best_info.get('best_step')}\n")
            f.write(f"  best_eval_f1_macro: {best_info.get('best_eval_f1_macro')}\n")

        if last_eval_metrics:
            _write_metrics_block(f, "Last epoch evaluation metrics (before best-restore)", last_eval_metrics, step=step_int)

        if best_eval_metrics:
            _write_metrics_block(f, "Best restored evaluation metrics", best_eval_metrics, step=step_int)

        freeze_summary = getattr(trainer, "freeze_summary_lines", None)
        if freeze_summary:
            f.write("\n[Layer freezing / trainable parameter summary]\n")
            for line in freeze_summary:
                f.write(line + "\n")


def _remove_checkpoint_dirs(path: str) -> None:
    if not os.path.isdir(path):
        return
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path) and entry.startswith("checkpoint-"):
            shutil.rmtree(full_path, ignore_errors=True)


def _plot_loss_curves(trainer: Trainer, run_name: str, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    history = getattr(trainer.state, "log_history", None) or []
    if not history:
        return

    train_by_epoch: Dict[int, float] = {}
    eval_by_epoch: Dict[int, float] = {}

    for record in history:
        epoch = record.get("epoch")
        if epoch is None:
            continue
        epoch_idx = _epoch_to_int(epoch)

        if "loss" in record:
            train_by_epoch[epoch_idx] = float(record["loss"])
        if "eval_loss" in record:
            eval_by_epoch[epoch_idx] = float(record["eval_loss"])

    if not train_by_epoch and not eval_by_epoch:
        return

    train_epochs = sorted(train_by_epoch.keys())
    eval_epochs = sorted(eval_by_epoch.keys())
    train_losses = [train_by_epoch[e] for e in train_epochs]
    eval_losses = [eval_by_epoch[e] for e in eval_epochs]

    run_name_safe = _fs_safe_model_name(run_name)

    plt.figure()
    if train_epochs:
        plt.plot(train_epochs, train_losses, marker="o", label="Train loss")
    if eval_epochs:
        plt.plot(eval_epochs, eval_losses, marker="o", label="Eval loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss vs Epoch – {run_name_safe}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = plots_dir / f"{run_name_safe}_loss.png"
    plt.savefig(loss_path)
    plt.close()
    print(f"[Info] Saved loss curves to {loss_path}")


def _save_run_config(final_model_dir: str, config: Dict[str, Any], slug: str) -> None:
    os.makedirs(final_model_dir, exist_ok=True)
    out = dict(config)
    out["slug"] = slug
    out_path = os.path.join(final_model_dir, "run_config.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[Info] Saved run_config.json to {out_path}")


# -----------------------------------------------------------------------------
# Tags + global attention mask
# -----------------------------------------------------------------------------
SPECIAL_TOKENS = ["[Q]", "[CTX]", "[A]"]


def _compose_text(q: str, ctx: str, a: str, input_order: str) -> str:
    q = "" if q is None else str(q)
    ctx = "" if ctx is None else str(ctx)
    a = "" if a is None else str(a)

    if input_order == "context+question+answer":
        return f"[CTX] {ctx}\n\n[Q] {q}\n\n[A] {a}"
    if input_order == "question+context+answer":
        return f"[Q] {q}\n\n[CTX] {ctx}\n\n[A] {a}"
    return f"[Q] {q}\n\n[CTX] {ctx}\n\n[A] {a}\n\n[Q] {q}"


def build_global_attention_mask_from_tags(
    input_ids: List[List[int]],
    *,
    mode: str,
    q_token_id: int,
    ctx_token_id: int,
    a_token_id: int,
) -> List[List[int]]:
    masks: List[List[int]] = []
    tag_set = {q_token_id, ctx_token_id, a_token_id}

    for ids in input_ids:
        m = [0] * len(ids)
        if not m:
            masks.append(m)
            continue

        m[0] = 1  # CLS always global

        if mode == "cls_plus_question":
            i = 0
            n = len(ids)
            while i < n:
                if ids[i] == q_token_id:
                    m[i] = 1
                    j = i + 1
                    while j < n and ids[j] not in tag_set:
                        m[j] = 1
                        j += 1
                    i = j
                else:
                    i += 1

        masks.append(m)

    return masks


def _apply_dropout_to_config(cfg: AutoConfig, dropout: float) -> AutoConfig:
    if hasattr(cfg, "hidden_dropout_prob"):
        cfg.hidden_dropout_prob = float(dropout)
    if hasattr(cfg, "attention_probs_dropout_prob"):
        cfg.attention_probs_dropout_prob = float(dropout)
    if hasattr(cfg, "classifier_dropout"):
        cfg.classifier_dropout = float(dropout)
    if hasattr(cfg, "dropout"):
        cfg.dropout = float(dropout)
    return cfg


# -----------------------------------------------------------------------------
# Core training function (single run)
# -----------------------------------------------------------------------------
def train_one(
    *,
    model_name: str,
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    label_col: str,
    output_dir: str,
    final_model_dir: str,
    plots_dir: Path,
    truncation: str,
    cfg_fixed: Dict[str, Any],
    save_model: bool = True,
    return_model: bool = False,
) -> Dict[str, Any]:
    max_length = int(cfg_fixed["max_sequence_length"])
    attention_window = int(cfg_fixed["attention_window"])
    learning_rate = float(cfg_fixed["learning_rate"])
    weight_decay = float(cfg_fixed["weight_decay"])
    warmup_ratio = float(cfg_fixed["warmup_ratio"])
    dropout = float(cfg_fixed["dropout"])
    batch_size = int(cfg_fixed["per_device_batch_size"])
    grad_accum = int(cfg_fixed["gradient_accumulation_steps"])
    num_epochs = int(cfg_fixed["num_train_epochs"])
    unfreeze_ratio = float(cfg_fixed["unfreeze_ratio"])
    classification_head = str(cfg_fixed["classification_head"])
    input_order = str(cfg_fixed["input_order"])
    global_attention = str(cfg_fixed["global_attention"])
    scheduler_type = str(cfg_fixed["lr_scheduler_type"])

    if attention_window % 2 != 0:
        raise ValueError(f"attention_window must be even, got {attention_window}")

    eff = _effective_len(max_length, attention_window)
    if eff != max_length:
        print(f"⚠️ [Note] max_length={max_length} will pad internally to {eff} for attention_window={attention_window}.")
    else:
        print(f"[Info] max_length={max_length} is divisible by attention_window={attention_window} (no internal pad).")

    # Labels
    unique_labels = sorted(train_df[label_col].dropna().unique())
    label2id: Dict[str, int] = {lab: i for i, lab in enumerate(unique_labels)}
    id2label: Dict[int, str] = {i: lab for lab, i in label2id.items()}

    # Clean
    for c in (Q_COL, CTX_COL, A_COL, label_col):
        if c not in train_df.columns:
            raise KeyError(f"Missing column in train_df: {c}")
        if c not in dev_df.columns:
            raise KeyError(f"Missing column in dev_df: {c}")

    train_df = train_df.dropna(subset=[Q_COL, CTX_COL, A_COL, label_col]).copy()
    dev_df = dev_df.dropna(subset=[Q_COL, CTX_COL, A_COL, label_col]).copy()
    train_df["label"] = train_df[label_col].map(label2id)
    dev_df["label"] = dev_df[label_col].map(label2id)

    label_counts = train_df["label"].value_counts().sort_index()
    class_weights = (len(train_df) / (len(label_counts) * label_counts)).sort_index()
    class_weights_tensor = torch.tensor(class_weights.to_numpy(), dtype=torch.float)

    # Tokenizer (+ tags)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    q_token_id = tokenizer.convert_tokens_to_ids("[Q]")
    ctx_token_id = tokenizer.convert_tokens_to_ids("[CTX]")
    a_token_id = tokenizer.convert_tokens_to_ids("[A]")

    def tokenize(batch):
        texts = [
            _compose_text(q, ctx, a, input_order)
            for q, ctx, a in zip(batch[Q_COL], batch[CTX_COL], batch[A_COL])
        ]

        if truncation != "head_tail":
            enc = tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_token_type_ids=False,
            )
            enc["global_attention_mask"] = build_global_attention_mask_from_tags(
                enc["input_ids"],
                mode=global_attention,
                q_token_id=q_token_id,
                ctx_token_id=ctx_token_id,
                a_token_id=a_token_id,
            )
            return enc

        # head_tail manual truncation
        inputs = tokenizer(
            texts,
            truncation=False,
            padding=False,
            return_attention_mask=True,
            return_token_type_ids=False,
        )
        input_ids_list = inputs["input_ids"]
        attn_list = inputs["attention_mask"]

        head_len = max_length // 2
        tail_len = max_length - head_len

        new_input_ids = []
        new_attn = []
        for ids, mask in zip(input_ids_list, attn_list):
            if len(ids) <= max_length:
                new_input_ids.append(ids)
                new_attn.append(mask)
                continue
            head_ids = ids[:head_len]
            tail_ids = ids[-tail_len:]
            head_mask = mask[:head_len]
            tail_mask = mask[-tail_len:]
            new_input_ids.append(head_ids + tail_ids)
            new_attn.append(head_mask + tail_mask)

        encoded = {"input_ids": new_input_ids, "attention_mask": new_attn}
        padded = tokenizer.pad(encoded, padding="max_length", max_length=max_length, return_tensors=None)
        padded["global_attention_mask"] = build_global_attention_mask_from_tags(
            padded["input_ids"],
            mode=global_attention,
            q_token_id=q_token_id,
            ctx_token_id=ctx_token_id,
            a_token_id=a_token_id,
        )
        return padded

    train_ds = Dataset.from_pandas(train_df, preserve_index=False).map(tokenize, batched=True)
    dev_ds = Dataset.from_pandas(dev_df, preserve_index=False).map(tokenize, batched=True)

    if "label" in train_ds.column_names and "labels" not in train_ds.column_names:
        train_ds = train_ds.rename_column("label", "labels")
    if "label" in dev_ds.column_names and "labels" not in dev_ds.column_names:
        dev_ds = dev_ds.rename_column("label", "labels")

    keep = {"input_ids", "attention_mask", "global_attention_mask", "labels"}
    train_remove = [c for c in train_ds.column_names if c not in keep]
    dev_remove = [c for c in dev_ds.column_names if c not in keep]
    if train_remove:
        train_ds = train_ds.remove_columns(train_remove)
    if dev_remove:
        dev_ds = dev_ds.remove_columns(dev_remove)

    # Config
    cfg = AutoConfig.from_pretrained(model_name)
    cfg.num_labels = len(unique_labels)
    cfg.id2label = dict(id2label)
    cfg.label2id = dict(label2id)
    cfg = _apply_dropout_to_config(cfg, dropout=float(dropout))

    # Apply attention window
    if hasattr(cfg, "num_hidden_layers"):
        layers = int(getattr(cfg, "num_hidden_layers"))
        cfg.attention_window = [int(attention_window)] * layers
    else:
        cfg.attention_window = int(attention_window)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=cfg)

    if added > 0:
        model.resize_token_embeddings(len(tokenizer))

    # Multilayer head
    if classification_head == "multilayer":
        hidden = getattr(cfg, "hidden_size", None)
        if hidden is None:
            raise ValueError("Could not find cfg.hidden_size for multilayer head.")
        model.classifier = SequenceMLPHead(hidden_size=int(hidden), num_labels=int(cfg.num_labels), dropout=float(dropout))

    # Freeze/unfreeze
    freeze_summary_lines = apply_partial_unfreezing(model, unfreeze_ratio)

    supports_gc = hasattr(model, "gradient_checkpointing_enable")
    if not supports_gc:
        print("[Warn] Model does not support gradient checkpointing; disabling.")

    # TrainingArguments (handle warmup_ratio fallback)
    training_kwargs = dict(
        output_dir=str(output_dir),
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        disable_tqdm=True,
        report_to="none",

        load_best_model_at_end=False,  # we restore best ourselves
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,

        num_train_epochs=int(num_epochs),
        learning_rate=float(learning_rate),
        per_device_train_batch_size=int(batch_size),
        per_device_eval_batch_size=int(batch_size),
        gradient_accumulation_steps=max(1, int(grad_accum)),
        weight_decay=float(weight_decay),

        seed=SEED,
        data_seed=SEED,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,

        save_safetensors=False,
        use_safetensors=False,

        gradient_checkpointing=bool(supports_gc),
        gradient_checkpointing_kwargs={"use_reentrant": False},

        lr_scheduler_type=str(scheduler_type),
        warmup_ratio=float(warmup_ratio),
    )
    if not supports_gc:
        training_kwargs.pop("gradient_checkpointing_kwargs", None)

    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" not in ta_params and "eval_strategy" in ta_params:
        training_kwargs["eval_strategy"] = training_kwargs.pop("evaluation_strategy")

    if "warmup_ratio" not in ta_params and "warmup_steps" in ta_params:
        steps_per_epoch = math.ceil(len(train_ds) / max(1, batch_size) / max(1, grad_accum))
        total_steps = steps_per_epoch * max(1, int(num_epochs))
        warmup_steps = int(round(total_steps * float(warmup_ratio)))
        training_kwargs.pop("warmup_ratio", None)
        training_kwargs["warmup_steps"] = max(0, warmup_steps)
        print(f"[Info] TrainingArguments lacks warmup_ratio; using warmup_steps={warmup_steps} instead.")

    filtered_kwargs = {k: v for k, v in training_kwargs.items() if k in ta_params}
    training_args = TrainingArguments(**filtered_kwargs)

    best_cb = BestModelInMemoryCallback(
        metric_name="eval_f1_macro",
        patience=int(cfg_fixed["early_stopping_patience"]),
        greater_is_better=True,
    )

    trainer_kwargs = dict(
        class_weights=class_weights_tensor,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=compute_metrics,
        callbacks=[best_cb],
    )
    trainer_sig = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_sig:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = WeightedTrainer(**trainer_kwargs)
    trainer.freeze_summary_lines = freeze_summary_lines

    print(f"[Info] Training (attention_window={attention_window})…")
    trainer.train()

    run_name = os.path.basename(os.path.normpath(str(output_dir))) or "run"
    _plot_loss_curves(trainer, run_name=run_name, plots_dir=plots_dir)

    print("[Info] Evaluating LAST epoch weights…")
    last_eval_metrics = trainer.evaluate(eval_dataset=dev_ds)
    trainer.last_eval_metrics = last_eval_metrics

    restored = best_cb.restore_best(trainer.model)
    best_info = None
    if restored:
        best_info = {
            "best_epoch": best_cb.best_epoch,
            "best_step": best_cb.best_step,
            "best_eval_f1_macro": best_cb.best_metric,
        }
        print(
            f"\n[Best model snapshot] eval_f1_macro={best_cb.best_metric:.6f} "
            f"(epoch={best_cb.best_epoch}, step={best_cb.best_step})"
        )
    else:
        print("\n⚠️ [Best model snapshot] Could not restore best model (no best_state_dict captured).")

    print("[Info] Evaluating BEST restored weights…")
    best_eval_metrics = trainer.evaluate(eval_dataset=dev_ds)

    step_val = getattr(trainer.state, "global_step", None)
    try:
        step_int = int(step_val) if step_val is not None else None
    except (TypeError, ValueError):
        step_int = None

    _print_metrics_block("Last epoch evaluation metrics (before best-restore)", last_eval_metrics, step=step_int)
    _print_metrics_block("Best restored evaluation metrics", best_eval_metrics, step=step_int)

    _save_training_progress(
        trainer,
        str(final_model_dir),
        best_info=best_info,
        last_eval_metrics=last_eval_metrics,
        best_eval_metrics=best_eval_metrics,
    )

    # Save BEST ONLY (optional)
    if save_model:
        os.makedirs(final_model_dir, exist_ok=True)
        trainer.save_model(str(final_model_dir))
        tokenizer.save_pretrained(str(final_model_dir))
        print(f"[Info] Saved BEST model & tokenizer to {final_model_dir}")

    _remove_checkpoint_dirs(str(output_dir))

    payload: Dict[str, Any] = {
        "best_info": best_info,
        "last_eval_metrics": last_eval_metrics,
        "best_eval_metrics": best_eval_metrics,
    }
    if return_model:
        payload["model"] = model
        payload["tokenizer"] = tokenizer
        payload["id2label"] = id2label
    return payload


# -----------------------------------------------------------------------------
# Runner (3 windows)
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Stage4 Longformer fixed runs over attention windows.")
    parser.add_argument("--dataset", choices=("original", "augmented"), default="original")
    parser.add_argument("--truncation", choices=("head", "head_tail"), default="head")
    args = parser.parse_args()

    model_name = str(FIXED_GRID_CONFIG["tokenizer"])

    # base folder = THIS FILE LOCATION
    STAGE_DIR = Path(__file__).resolve().parent
    TMP_DIR = STAGE_DIR / "tmp_checkpoints"
    TRAINED_DIR = STAGE_DIR / "stage4_trained_models"
    PLOTS_DIR = STAGE_DIR / "plots"
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    TRAINED_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    train_path = TRAIN_CSV_PATH if args.dataset == "original" else AUGMENTED_TRAIN_CSV_PATH
    train_full_df = pd.read_csv(train_path)

    if not os.path.exists(TEST_CSV_PATH):
        raise FileNotFoundError(f"[Fatal] Test CSV not found at {TEST_CSV_PATH}.")
    dev_full_df = pd.read_csv(TEST_CSV_PATH)

    required_cols = {Q_COL, CTX_COL, A_COL, TARGET_COLUMN}
    for c in required_cols:
        if c not in train_full_df.columns:
            raise KeyError(f"[Fatal] Missing {c} in train CSV")
        if c not in dev_full_df.columns:
            raise KeyError(f"[Fatal] Missing {c} in test/dev CSV")

    print("\n[FixedConfig] Base (except attention_window):")
    for k, v in FIXED_GRID_CONFIG.items():
        if k == "attention_window":
            continue
        print(f"  {k}: {v}")

    results: List[Dict[str, Any]] = []
    raw_csv = STAGE_DIR / "stage4_attention_window_summary_raw.csv"
    sorted_csv = STAGE_DIR / "stage4_attention_window_summary.csv"

    for i, aw in enumerate(ATTENTION_WINDOWS_TO_RUN, start=1):
        cfg_run = dict(FIXED_GRID_CONFIG)
        cfg_run["attention_window"] = int(aw)

        slug = _build_output_slug(
            model_name=model_name,
            truncation=args.truncation,
            dataset=args.dataset,
            max_length=int(cfg_run["max_sequence_length"]),
            attention_window=int(cfg_run["attention_window"]),
            learning_rate=float(cfg_run["learning_rate"]),
            weight_decay=float(cfg_run["weight_decay"]),
            warmup_ratio=float(cfg_run["warmup_ratio"]),
            dropout=float(cfg_run["dropout"]),
            classification_head=str(cfg_run["classification_head"]),
            input_order=str(cfg_run["input_order"]),
            global_attention=str(cfg_run["global_attention"]),
        )

        print(f"\n[Run {i}/{len(ATTENTION_WINDOWS_TO_RUN)}] attention_window={aw}")
        print(f"[Run] slug={slug}")

        output_dir = TMP_DIR / slug
        final_model_dir = TRAINED_DIR / slug

        meta = train_one(
            model_name=model_name,
            train_df=train_full_df,
            dev_df=dev_full_df,
            label_col=TARGET_COLUMN,
            output_dir=str(output_dir),
            final_model_dir=str(final_model_dir),
            plots_dir=PLOTS_DIR,
            truncation=args.truncation,
            cfg_fixed=cfg_run,
        )

        _save_run_config(str(final_model_dir), cfg_run, slug)

        best_eval = meta.get("best_eval_metrics") or {}
        row = {
            "run_index": i,
            "attention_window": int(aw),
            "effective_seq_len": int(_effective_len(int(cfg_run["max_sequence_length"]), int(aw))),
            "slug": slug,
            "best_eval_f1_macro": float(best_eval.get("eval_f1_macro", float("nan"))),
            "best_eval_accuracy": float(best_eval.get("eval_accuracy", float("nan"))),
            "best_eval_loss": float(best_eval.get("eval_loss", float("nan"))),
            "final_model_dir": str(final_model_dir),
        }
        results.append(row)

        pd.DataFrame(results).to_csv(raw_csv, index=False)

    out_df = pd.DataFrame(results).sort_values(by="best_eval_f1_macro", ascending=False)
    out_df.to_csv(sorted_csv, index=False)
    print(f"\n[Done] Saved summary -> {sorted_csv}")
    print(f"[Done] Saved raw -> {raw_csv}")
    print(f"[Saved] Models in: {TRAINED_DIR}")

if __name__ == "__main__":
    main()
