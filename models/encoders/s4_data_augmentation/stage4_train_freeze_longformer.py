# models/encoders/longformer_experiments/ablations/stage3_train_longformer_ablations.py
#
# Longformer Ablation Study (Stage 3 placement)
# - Fixed training configuration (no hyperparameter grid)
# - Ablations over:
#     (1) classification_head: default vs multilayer (MLPHead)
#     (2) input_order: 3 tagged layouts using [Q], [CTX], [A]
#     (3) global_attention: cls_only vs cls_plus_question (tag-based [Q] spans)
#
# Saves everything UNDER THIS FOLDER (stage3):
#   longformer_experiments/ablations/
#       tmp_checkpoints/<slug>/
#       stage3_trained_models/<slug>/
#
# Examples:
#   Run full ablations (12 runs):
#     python stage3_train_longformer_ablations.py --param_mode fixed --grid_search --dataset original --truncation head
#
#   Run single ablation:
#     python stage3_train_longformer_ablations.py --param_mode fixed --run_one \
#       --classification_head default \
#       --input_order question+context+answer \
#       --global_attention cls_plus_question \
#       --dataset original \
#       --truncation head_tail


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
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
from itertools import product
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
# 0) Fixed config + ablation space
# -----------------------------------------------------------------------------

LONGFORMER_MODEL = "allenai/longformer-base-4096"

# Tags used to construct inputs & locate question spans for global attention
SPECIAL_TOKENS = ["[Q]", "[CTX]", "[A]"]

# Fixed training configuration (Nawar spec)
FIXED_GRID_CONFIG = {
    # Model / tokenizer
    "tokenizer": LONGFORMER_MODEL,
    "max_sequence_length": 2048,
    "attention_window": 512,

    # Optimization
    "optimizer": "AdamW",
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "scheduler": "linear_with_warmup",

    # Regularization & loss
    "dropout": 0.1,
    "loss": "WCE",

    # Batch & training control
    "batch_size": 1,      # per-device batch size
    "grad_accum": 16,     # effective batch = batch_size * grad_accum
    "num_train_epochs": 20,
    "early_stopping_patience": 7,

    # Unfreezing strategy (keep same behavior as your other stages)
    "unfreeze_ratio": 0.25,
}

# Ablation dimensions (2 x 3 x 2 = 12)
ABLATION_SPACE = {
    "classification_head": ["default", "multilayer"],
    "input_order": [
        "context+question+answer",
        "question+context+answer",
        "question+context+answer+question_repeat",
    ],
    "global_attention": ["cls_only", "cls_plus_question"],
}


# -----------------------------------------------------------------------------
# 1) Setup
# -----------------------------------------------------------------------------
SEED = 42
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = int(FIXED_GRID_CONFIG.get("early_stopping_patience", 7))

TRAIN_CSV_PATH = os.path.join("datasets", "train_dataset.csv")
TEST_CSV_PATH = os.path.join("datasets", "test_dataset.csv")
AUGMENTED_TRAIN_CSV_PATH = os.path.join(
    "datasets",
    "augmented_dataset",
    "train_dataset_augmented_filtered.csv",
)

TASK_ID = "t1"
TUNE_STRATEGY = "partial"

# Plots + summary (kept under results/, like your other stages)
PLOTS_DIR = Path("results") / "plots" / "encoder"
SUMMARY_CSV_PATH = Path("results") / "eval_logs" / "detailed" / "encoder" / "stage3_longformer_ablation_summary.csv"

# Defaults for CLI (remain overridable)
DEFAULT_CONFIG = {
    "num_train_epochs": int(FIXED_GRID_CONFIG["num_train_epochs"]),
    "batch_size": int(FIXED_GRID_CONFIG["batch_size"]),
    "learning_rate": float(FIXED_GRID_CONFIG["learning_rate"]),
    "weight_decay": float(FIXED_GRID_CONFIG["weight_decay"]),
    "grad_accum": int(FIXED_GRID_CONFIG["grad_accum"]),
    "dropout": float(FIXED_GRID_CONFIG["dropout"]),
    "max_length": int(FIXED_GRID_CONFIG["max_sequence_length"]),
    "attention_window": int(FIXED_GRID_CONFIG["attention_window"]),
    "unfreeze_ratio": float(FIXED_GRID_CONFIG["unfreeze_ratio"]),
}

TARGET_COLUMN = "clarity_label"
Q_KEY = "question"
A_KEY = "interview_answer"
CTX_KEY = "interview_question"  # context

hf_logging.set_verbosity_error()


# -----------------------------------------------------------------------------
# 2) Utilities
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


def _resolve_model_name(_user_name: str) -> str:
    return LONGFORMER_MODEL


def _normalize_global_attention(mode: str) -> str:
    m = (mode or "").strip().lower()
    if m in ("cls", "cls_only"):
        return "cls_only"
    if m in ("cls_question", "cls_plus_question", "cls_plus_q", "clsq"):
        return "cls_plus_question"
    raise ValueError(f"Unknown global_attention mode: {mode}")


def _short_input_order_tag(order: str) -> str:
    mapping = {
        "context+question+answer": "CTX-Q-A",
        "question+context+answer": "Q-CTX-A",
        "question+context+answer+question_repeat": "Q-CTX-A-Q",
    }
    return mapping.get(order, _fs_safe_model_name(order))


def _build_output_slug(
    model_name: str,
    param_mode: str,
    *,
    unfreeze_ratio: Optional[float] = None,
    truncation: Optional[str] = None,
    dataset: Optional[str] = None,
    classification_head: Optional[str] = None,
    input_order: Optional[str] = None,
    global_attention: Optional[str] = None,
    max_length: Optional[int] = None,
    learning_rate: Optional[float] = None,
    dropout: Optional[float] = None,
    attention_window: Optional[int] = None,
) -> str:
    meta = _get_meta_fallback(model_name)

    raw_trunc = (truncation or "").strip()
    if raw_trunc == "head_tail":
        trunc_tag = "head-tail"
    elif raw_trunc:
        trunc_tag = raw_trunc
    else:
        trunc_tag = "notrunc"

    pct = int(round(100 * float(unfreeze_ratio))) if unfreeze_ratio is not None else None
    strategy_tag = f"unfreezing{pct}" if pct is not None else TUNE_STRATEGY

    head_tag = (classification_head or "default").strip().lower()
    if head_tag == "default":
        head_tag = "defaultHead"
    elif head_tag == "multilayer":
        head_tag = "multilayerHead"
    else:
        head_tag = _fs_safe_model_name(head_tag)

    slug = (
        f"{TASK_ID}_{meta['arch']}_{meta['lang']}_{meta['size']}"
        f"_{strategy_tag}_{param_mode}_{head_tag}_{trunc_tag}"
    )

    if dataset:
        ds = dataset
        if ds == "original":
            ds = "originalData"
        elif ds == "augmented":
            ds = "augmentedData"
        slug = f"{slug}_{ds}"

    if input_order:
        slug = f"{slug}_ord{_short_input_order_tag(input_order)}"

    if global_attention:
        g = _normalize_global_attention(global_attention)
        slug = f"{slug}_gattn{g}"

    if max_length:
        slug = f"{slug}_L{int(max_length)}"

    if attention_window:
        slug = f"{slug}_aw{int(attention_window)}"

    if learning_rate is not None:
        lr_tag = f"{learning_rate:.0e}".replace("+", "")
        slug = f"{slug}_lr{lr_tag}"

    if dropout is not None:
        slug = f"{slug}_do{dropout:.2f}"

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
    epoch_float = float(epoch_value)
    epoch_idx = int(math.floor(epoch_float + 0.5))
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
            print(f"  {key}: {val:.4f}")
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
            f.write(f"  {key}: {val:.4f}\n")
        else:
            f.write(f"  {key}: {val}\n")

    if step is not None:
        f.write(f"  step: {step}\n")


# -----------------------------------------------------------------------------
# 3) Best in-memory restore callback
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
        if not metrics:
            return control
        if self.metric_name not in metrics:
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
                f"[EarlyStop] No improvement in {self.metric_name} "
                f"for {self._bad_epochs} evals (patience={self.patience}). Stopping."
            )
            control.should_training_stop = True

        return control

    def restore_best(self, model: nn.Module) -> bool:
        if self.best_state_dict is None:
            return False
        model.load_state_dict(self.best_state_dict, strict=True)
        return True


# -----------------------------------------------------------------------------
# 4) WeightedTrainer (kept style, just safer log via super().log)
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
        logits = outputs.get("logits") if isinstance(outputs, dict) else getattr(outputs, "logits", None)
        if logits is None:
            raise ValueError("Model outputs did not contain 'logits'.")

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

        super().log(logs)

        record = self.state.log_history[-1] if self.state.log_history else {}
        epoch_val = record.get("epoch")
        if epoch_val is None:
            return

        epoch_idx = _epoch_to_int(epoch_val)
        row = self._epoch_cache.setdefault(epoch_idx, {})

        if "loss" in record:
            row["train_loss"] = float(record["loss"])
        if "eval_loss" in record:
            row["val_loss"] = float(record["eval_loss"])
        if "eval_accuracy" in record:
            row["accuracy"] = float(record["eval_accuracy"])
        if "eval_f1_macro" in record:
            row["f1_macro"] = float(record["eval_f1_macro"])
        if "eval_f1_micro" in record:
            row["f1_micro"] = float(record["eval_f1_micro"])

        self._maybe_print_epoch_row(epoch_idx)


# -----------------------------------------------------------------------------
# 5) Freezing / partial unfreezing logic + summary (kept)
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

    # 1) Freeze embeddings
    if hasattr(base, "embeddings"):
        for param in base.embeddings.parameters():
            param.requires_grad = False
        lines.append("[Freeze] Embedding layer parameters frozen.")
    else:
        lines.append("⚠️ Could not apply embedding freezing. No 'embeddings' on base model.")

    # 2) Freeze encoder layers based on unfreeze_ratio
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

    # 3) Trainable summary
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
        if ".encoder.layer." in name and "layer" in name.split("."):
            parts = name.split(".")
            idx = parts.index("layer")
            top_group = ".".join(parts[:idx + 2])
        else:
            top_group = ".".join(name.split(".")[:2])
        grouped[top_group].append(name)

    for group, names in sorted(grouped.items()):
        lines.append(f"✅ {group}: ({len(names)} params)")

    return lines


# -----------------------------------------------------------------------------
# 6) Progress saving + plots (kept)
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


def _plot_loss_curves(trainer: Trainer, run_name: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    history = getattr(trainer.state, "log_history", None) or []
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
        plt.plot(eval_epochs, eval_losses, marker="o", label="Eval loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss vs Epoch – {run_name_safe}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = PLOTS_DIR / f"{run_name_safe}_loss.png"
    plt.savefig(loss_path)
    plt.close()
    print(f"[Info] Saved loss curves to {loss_path}")


# -----------------------------------------------------------------------------
# 7) Input building for ablations
# -----------------------------------------------------------------------------
def _build_input_text_row(row: pd.Series, input_order: str) -> str:
    q = "" if pd.isna(row.get(Q_KEY)) else str(row.get(Q_KEY))
    a = "" if pd.isna(row.get(A_KEY)) else str(row.get(A_KEY))
    ctx = "" if pd.isna(row.get(CTX_KEY)) else str(row.get(CTX_KEY))

    if input_order == "context+question+answer":
        return f"[CTX] {ctx} [Q] {q} [A] {a}"
    if input_order == "question+context+answer":
        return f"[Q] {q} [CTX] {ctx} [A] {a}"
    if input_order == "question+context+answer+question_repeat":
        return f"[Q] {q} [CTX] {ctx} [A] {a} [Q] {q}"

    raise ValueError(f"Unknown input_order: {input_order}")


def _add_input_text_column(df: pd.DataFrame, input_order: str) -> None:
    df["input_text"] = df.apply(lambda r: _build_input_text_row(r, input_order), axis=1)


# -----------------------------------------------------------------------------
# 8) Global attention mask builder (tag-based)
# -----------------------------------------------------------------------------
def build_global_attention_mask_from_tags(
    input_ids: List[List[int]],
    *,
    mode: str,
    tokenizer: AutoTokenizer,
) -> List[List[int]]:
    mode = _normalize_global_attention(mode)

    q_id = tokenizer.convert_tokens_to_ids("[Q]")
    ctx_id = tokenizer.convert_tokens_to_ids("[CTX]")
    a_id = tokenizer.convert_tokens_to_ids("[A]")
    tag_ids = {q_id, ctx_id, a_id}

    masks: List[List[int]] = []
    for ids in input_ids:
        m = [0] * len(ids)
        if not ids:
            masks.append(m)
            continue

        # always global on CLS / <s>
        m[0] = 1

        if mode == "cls_only":
            masks.append(m)
            continue

        # cls_plus_question: tokens after each [Q] until next tag token
        q_positions = [i for i, t in enumerate(ids) if t == q_id]
        for qpos in q_positions:
            start = qpos + 1
            end = len(ids)
            for j in range(start, len(ids)):
                if ids[j] in tag_ids:
                    end = j
                    break
            for j in range(start, end):
                m[j] = 1

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


def _assert_length_multiple_of_attention_window(max_length: int, attention_window: int = 512) -> None:
    if max_length % attention_window != 0:
        print(
            f"⚠️ [Warn] max_length={max_length} is not a multiple of attention_window={attention_window}. "
            "Longformer can still run, but you may see padding/multiple warnings or inefficiencies. "
            "Consider using 1024/2048/3072/4096."
        )


# -----------------------------------------------------------------------------
# 9) Multilayer head (use YOUR MLPHead + adapter)
# -----------------------------------------------------------------------------
class MLPHead(nn.Module):
    # CLS token → Dropout → Linear → GELU → Dropout → Linear → num_labels logits
    def __init__(self, hidden_size: int, num_labels: int = 3):
        super().__init__()
        self.dropout1 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.dropout2 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_size, num_labels)

    def forward(self, cls_embedding: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(cls_embedding)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout2(x)
        logits = self.fc2(x)
        return logits


class LongformerClassifierAdapter(nn.Module):
    """
    LongformerForSequenceClassification calls:
        logits = self.classifier(sequence_output)
    where sequence_output is (batch, seq_len, hidden).

    This adapter extracts CLS embedding and feeds it into MLPHead.
    """
    def __init__(self, mlp: MLPHead):
        super().__init__()
        self.mlp = mlp

    def forward(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        cls_embedding = features[:, 0, :]
        return self.mlp(cls_embedding)


def _apply_multilayer_head(model: nn.Module) -> None:
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise ValueError("Model has no .config; cannot attach multilayer head.")

    hidden_size = int(getattr(cfg, "hidden_size", 0) or 0)
    num_labels = int(getattr(cfg, "num_labels", 0) or 0)

    if hidden_size <= 0:
        raise ValueError("Could not determine hidden_size from config.")
    if num_labels <= 0:
        raise ValueError("Could not determine num_labels from config.")
    if not hasattr(model, "classifier"):
        raise ValueError("Model has no `.classifier` attribute; unexpected architecture.")

    mlp = MLPHead(hidden_size=hidden_size, num_labels=num_labels)
    model.classifier = LongformerClassifierAdapter(mlp)


# -----------------------------------------------------------------------------
# 10) Core training function
# -----------------------------------------------------------------------------
def train_model(
    model_name: str,
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    text_key: str,
    label_col: str,
    *,
    output_dir: str,
    final_model_dir: str,
    batch_size: int,
    grad_accum: int,
    learning_rate: float,
    num_epochs: int,
    weight_decay: float,
    unfreeze_ratio: float,
    truncation: str,
    max_length: int,
    global_attention: str,
    dropout: float,
    classification_head: str,
    attention_window: int,
) -> Tuple[Trainer, Dict[int, str], Dict[str, int], Any, Dict[str, Any]]:

    unique_labels = sorted(train_df[label_col].dropna().unique())
    label2id: Dict[str, int] = {lab: i for i, lab in enumerate(unique_labels)}
    id2label: Dict[int, str] = {i: lab for lab, i in label2id.items()}

    train_df = train_df.dropna(subset=[text_key, label_col]).copy()
    dev_df = dev_df.dropna(subset=[text_key, label_col]).copy()
    train_df["label"] = train_df[label_col].map(label2id)
    dev_df["label"] = dev_df[label_col].map(label2id)

    label_counts = train_df["label"].value_counts().sort_index()
    class_weights = (len(train_df) / (len(label_counts) * label_counts)).sort_index()
    class_weights_tensor = torch.tensor(class_weights.to_numpy(), dtype=torch.float)

    _assert_length_multiple_of_attention_window(max_length, int(attention_window))

    # Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # Add tags as special tokens so [Q]/[CTX]/[A] are stable token IDs
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    def tokenize(batch):
        if truncation != "head_tail":
            enc = tokenizer(
                batch[text_key],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_token_type_ids=False,
            )
            enc["global_attention_mask"] = build_global_attention_mask_from_tags(
                enc["input_ids"],
                mode=global_attention,
                tokenizer=tokenizer,
            )
            return enc

        # head_tail: tokenize without truncation then manually take head+tail
        inputs = tokenizer(
            batch[text_key],
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

        for i in range(len(input_ids_list)):
            ids = input_ids_list[i]
            mask = attn_list[i]

            if len(ids) <= max_length:
                new_input_ids.append(ids)
                new_attn.append(mask)
                continue

            head_ids = ids[:head_len]
            tail_ids = ids[-tail_len:]
            head_mask = mask[:head_len]
            tail_mask = mask[-tail_len:]

            truncated_ids = head_ids + tail_ids
            truncated_mask = head_mask + tail_mask

            new_input_ids.append(truncated_ids)
            new_attn.append(truncated_mask)

        encoded = {"input_ids": new_input_ids, "attention_mask": new_attn}

        padded = tokenizer.pad(
            encoded,
            padding="max_length",
            max_length=max_length,
            return_tensors=None,
        )

        padded["global_attention_mask"] = build_global_attention_mask_from_tags(
            padded["input_ids"],
            mode=global_attention,
            tokenizer=tokenizer,
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

    # Model (HF default) + dropout + attention_window
    cfg = AutoConfig.from_pretrained(model_name)
    cfg.num_labels = len(unique_labels)
    cfg.id2label = dict(id2label)
    cfg.label2id = dict(label2id)
    cfg = _apply_dropout_to_config(cfg, dropout=float(dropout))

    if hasattr(cfg, "attention_window"):
        cfg.attention_window = int(attention_window)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=cfg)

    # Must resize embeddings after adding special tokens
    model.resize_token_embeddings(len(tokenizer))

    # Apply multilayer head if needed
    head = (classification_head or "default").strip().lower()
    if head == "multilayer":
        _apply_multilayer_head(model)

    # Freezing
    freeze_summary_lines = apply_partial_unfreezing(model, float(unfreeze_ratio))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable_params / max(1, total_params)
    print(f"[Info] Trainable params: {trainable_params:,} / {total_params:,} ({pct:.2f}%)")
    print(f"[Info] Dropout set to: {dropout}")
    print(f"[Info] Head type: {head}")

    # TrainingArguments (kept)
    training_kwargs = dict(
        output_dir=str(output_dir),
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        disable_tqdm=True,
        report_to="none",

        load_best_model_at_end=False,
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

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" not in ta_params and "eval_strategy" in ta_params:
        training_kwargs["eval_strategy"] = training_kwargs.pop("evaluation_strategy")

    filtered_kwargs = {k: v for k, v in training_kwargs.items() if k in ta_params}
    training_args = TrainingArguments(**filtered_kwargs)

    best_cb = BestModelInMemoryCallback(
        metric_name="eval_f1_macro",
        patience=EARLY_STOPPING_PATIENCE,
        greater_is_better=True,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights_tensor,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[best_cb],
    )
    trainer.freeze_summary_lines = freeze_summary_lines

    print("[Info] Starting Longformer training…")
    trainer.train()

    run_name = os.path.basename(os.path.normpath(str(output_dir))) or model_name
    _plot_loss_curves(trainer, run_name=run_name)

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

    print("\n[Layer freezing / trainable parameter summary]")
    for line in freeze_summary_lines:
        print(line)

    _save_training_progress(
        trainer,
        str(final_model_dir),
        best_info=best_info,
        last_eval_metrics=last_eval_metrics,
        best_eval_metrics=best_eval_metrics,
    )

    # SAVE BEST weights (because we restored them)
    os.makedirs(final_model_dir, exist_ok=True)
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    print(f"[Info] Saved BEST fine-tuned model & tokenizer to {final_model_dir}")

    _remove_checkpoint_dirs(str(output_dir))

    meta_out = {
        "best_info": best_info,
        "last_eval_metrics": last_eval_metrics,
        "best_eval_metrics": best_eval_metrics,
    }
    return trainer, id2label, label2id, tokenizer, meta_out


# -----------------------------------------------------------------------------
# 11) Run helpers
# -----------------------------------------------------------------------------
def _run_one(
    *,
    args,
    resolved_model: str,
    train_full_df: pd.DataFrame,
    test_full_df: pd.DataFrame,
    classification_head: str,
    input_order: str,
    global_attention: str,
) -> Dict[str, Any]:

    max_length = int(args.max_length)
    learning_rate = float(args.learning_rate)
    dropout = float(args.dropout)
    attention_window = int(args.attention_window)

    slug = _build_output_slug(
        resolved_model,
        args.param_mode,
        unfreeze_ratio=float(args.unfreeze_ratio),
        truncation=args.truncation,
        dataset=args.dataset,
        classification_head=classification_head,
        input_order=input_order,
        global_attention=global_attention,
        max_length=max_length,
        attention_window=attention_window,
        learning_rate=learning_rate,
        dropout=dropout,
    )

    # SAVE EVERYTHING UNDER THIS FOLDER (stage3 requested)
    STAGE_DIR = Path(__file__).resolve().parent  # longformer_experiments/ablations
    output_dir = STAGE_DIR / "tmp_checkpoints" / slug
    final_model_dir = STAGE_DIR / "stage3_trained_models" / slug

    print(
        f"\n[Run] Longformer ablation "
        f"(head={classification_head}, input_order={input_order}, gattn={_normalize_global_attention(global_attention)}, "
        f"truncation={args.truncation}, unfreeze_ratio={float(args.unfreeze_ratio):.2f}, "
        f"dataset={args.dataset}, max_length={max_length}, attention_window={attention_window}, "
        f"batch_size={args.batch_size}, grad_accum={args.grad_accum}, lr={learning_rate}, dropout={dropout}), "
        f"mode={args.param_mode}"
    )

    # Build tagged input_text per ablation
    train_df = train_full_df.copy()
    dev_df = test_full_df.copy()
    _add_input_text_column(train_df, input_order)
    _add_input_text_column(dev_df, input_order)

    train_df = train_df.dropna(subset=["input_text", TARGET_COLUMN]).copy()
    dev_df = dev_df.dropna(subset=["input_text", TARGET_COLUMN]).copy()

    _, _, _, _, meta = train_model(
        model_name=resolved_model,
        train_df=train_df,
        dev_df=dev_df,
        text_key="input_text",
        label_col=TARGET_COLUMN,
        output_dir=str(output_dir),
        final_model_dir=str(final_model_dir),
        batch_size=int(args.batch_size),
        grad_accum=int(args.grad_accum),
        learning_rate=float(learning_rate),
        num_epochs=int(args.num_epochs),
        weight_decay=float(args.weight_decay),
        unfreeze_ratio=float(args.unfreeze_ratio),
        truncation=args.truncation,
        max_length=int(max_length),
        global_attention=global_attention,
        dropout=float(dropout),
        classification_head=classification_head,
        attention_window=int(attention_window),
    )

    best_eval = meta.get("best_eval_metrics") or {}
    out = {
        "slug": slug,
        "classification_head": classification_head,
        "input_order": input_order,
        "global_attention": _normalize_global_attention(global_attention),
        "learning_rate": float(learning_rate),
        "max_length": int(max_length),
        "attention_window": int(attention_window),
        "dropout": float(dropout),
        "best_eval_f1_macro": float(best_eval.get("eval_f1_macro", float("nan"))),
        "best_eval_accuracy": float(best_eval.get("eval_accuracy", float("nan"))),
        "best_eval_loss": float(best_eval.get("eval_loss", float("nan"))),
        "final_model_dir": str(final_model_dir),
    }
    return out


# -----------------------------------------------------------------------------
# 12) Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Longformer ablation study (fixed config) over head/input_order/global_attention."
    )
    parser.add_argument(
        "--dataset",
        choices=("original", "augmented"),
        default="original",
        help="Which training dataset to use.",
    )
    parser.add_argument(
        "--model_name",
        default="longformer",
        help="Ignored (Longformer-only). Keep for drop-in compatibility.",
    )
    parser.add_argument(
        "--param_mode",
        required=True,
        choices=("fixed",),
        help="Parameter mode to use.",
    )
    parser.add_argument(
        "--truncation",
        default="head",
        choices=("head", "head_tail"),
        help="Truncation strategy.",
    )

    # Ablation params (single run)
    parser.add_argument(
        "--classification_head",
        choices=("default", "multilayer"),
        default="default",
        help="Head ablation dimension.",
    )
    parser.add_argument(
        "--input_order",
        choices=tuple(ABLATION_SPACE["input_order"]),
        default="question+context+answer",
        help="Tagged input layout.",
    )
    parser.add_argument(
        "--global_attention",
        choices=("cls_only", "cls_plus_question", "cls", "cls_question"),
        default="cls_only",
        help="Global attention pattern (cls_only vs cls_plus_question).",
    )

    # Training params (defaults from FIXED_GRID_CONFIG)
    parser.add_argument("--batch_size", type=int, default=int(DEFAULT_CONFIG["batch_size"]))
    parser.add_argument("--grad_accum", type=int, default=int(DEFAULT_CONFIG["grad_accum"]))
    parser.add_argument("--num_epochs", type=int, default=int(DEFAULT_CONFIG["num_train_epochs"]))
    parser.add_argument("--weight_decay", type=float, default=float(DEFAULT_CONFIG["weight_decay"]))
    parser.add_argument("--unfreeze_ratio", type=float, default=float(DEFAULT_CONFIG["unfreeze_ratio"]))

    parser.add_argument("--learning_rate", type=float, default=float(DEFAULT_CONFIG["learning_rate"]))
    parser.add_argument("--max_length", type=int, default=int(DEFAULT_CONFIG["max_length"]))
    parser.add_argument("--dropout", type=float, default=float(DEFAULT_CONFIG["dropout"]))
    parser.add_argument("--attention_window", type=int, default=int(DEFAULT_CONFIG["attention_window"]))

    # Runner flags
    parser.add_argument(
        "--grid_search",
        action="store_true",
        help="Runs the full ablation grid (12 runs). Kept name for compatibility.",
    )
    parser.add_argument(
        "--run_one",
        action="store_true",
        help="Runs a single ablation setting (classification_head/input_order/global_attention).",
    )

    args = parser.parse_args()

    if not args.grid_search and not args.run_one:
        # Default: run full ablations (what Nawar asked for)
        args.grid_search = True

    resolved_model = _resolve_model_name(args.model_name)
    print(f"[Info] Using model: {resolved_model}")

    train_path = TRAIN_CSV_PATH if args.dataset == "original" else AUGMENTED_TRAIN_CSV_PATH
    train_full_df = pd.read_csv(train_path)

    if not os.path.exists(TEST_CSV_PATH):
        raise FileNotFoundError(f"[Fatal] Test CSV not found at {TEST_CSV_PATH}.")
    test_full_df = pd.read_csv(TEST_CSV_PATH)

    # Basic sanity dropna for core columns (input_text is built per-run)
    train_full_df = train_full_df.dropna(subset=[Q_KEY, A_KEY, TARGET_COLUMN]).copy()
    test_full_df = test_full_df.dropna(subset=[Q_KEY, A_KEY, TARGET_COLUMN]).copy()

    # -------------------------------------------------------------------------
    # FULL ABLATION GRID (12 runs)
    # -------------------------------------------------------------------------
    if args.grid_search and not args.run_one:
        combos = list(product(
            ABLATION_SPACE["classification_head"],
            ABLATION_SPACE["input_order"],
            ABLATION_SPACE["global_attention"],
        ))
        print(f"\n[Ablations] Running {len(combos)} configurations (2 x 3 x 2).")

        results: List[Dict[str, Any]] = []
        best_row: Optional[Dict[str, Any]] = None

        for i, (head, order, gattn) in enumerate(combos, start=1):
            print(f"\n[Ablations] ({i}/{len(combos)}) head={head}, order={order}, gattn={gattn}")
            row = _run_one(
                args=args,
                resolved_model=resolved_model,
                train_full_df=train_full_df,
                test_full_df=test_full_df,
                classification_head=head,
                input_order=order,
                global_attention=gattn,
            )
            results.append(row)

            if best_row is None or (row["best_eval_f1_macro"] > best_row["best_eval_f1_macro"]):
                best_row = row

        out_df = pd.DataFrame(results).sort_values(by="best_eval_f1_macro", ascending=False)
        SUMMARY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(SUMMARY_CSV_PATH, index=False)

        print(f"\n[Ablations] Saved summary CSV -> {SUMMARY_CSV_PATH}")
        if best_row:
            print("\n[Ablations] BEST CONFIG:")
            print(f"  best_eval_f1_macro: {best_row['best_eval_f1_macro']:.6f}")
            print(f"  classification_head: {best_row['classification_head']}")
            print(f"  input_order: {best_row['input_order']}")
            print(f"  global_attention: {best_row['global_attention']}")
            print(f"  max_length: {best_row['max_length']}")
            print(f"  attention_window: {best_row['attention_window']}")
            print(f"  lr: {best_row['learning_rate']}")
            print(f"  dropout: {best_row['dropout']}")
            print(f"  slug: {best_row['slug']}")
            print(f"  final_model_dir: {best_row['final_model_dir']}")
        return

    # -------------------------------------------------------------------------
    # SINGLE RUN
    # -------------------------------------------------------------------------
    row = _run_one(
        args=args,
        resolved_model=resolved_model,
        train_full_df=train_full_df,
        test_full_df=test_full_df,
        classification_head=args.classification_head,
        input_order=args.input_order,
        global_attention=args.global_attention,
    )

    print("\n[Final] Longformer DEV RESULTS (BEST restored):")
    print(f"F1-Macro: {row['best_eval_f1_macro']:.4f}")
    print(f"Accuracy: {row['best_eval_accuracy']:.4f} ({row['best_eval_accuracy'] * 100:.2f}%)")
    print(f"Eval Loss: {row['best_eval_loss']:.4f}")
    print(f"Saved to: {row['final_model_dir']}")
    print("[Info] Training run complete.")


if __name__ == "__main__":
    main()
