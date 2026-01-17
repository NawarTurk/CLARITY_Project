
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
from itertools import product

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
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.utils import WEIGHTS_NAME, SAFE_WEIGHTS_NAME
from transformers.utils import logging as hf_logging

import matplotlib.pyplot as plt

SEARCH_SPACE = {
    "learning_rate": [1e-5, 2e-5],
    "max_length": [1024, 2048],
    "dropout": [0.1, 0.2],
}

FIXED_GRID_CONFIG = {
    "tokenizer": "Longformer",
    "attention_window": 512,
    "classification_head": "default",
    "unfreeze_ratio": 0.25,
    "loss": "WCE",
    "optimizer": "AdamW",
    "weight_decay": 0.01,
    "scheduler": "linear with warmup",
    "batch_size": 8,
    "early_stopping_patience": 7,
}

# -----------------------------------------------------------------------------
# 1) Setup
# -----------------------------------------------------------------------------
LONGFORMER_MODEL = "allenai/longformer-base-4096"  # supports sequences up to 4096

SEED = 42
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = int(FIXED_GRID_CONFIG["early_stopping_patience"])

TRAIN_CSV_PATH = os.path.join("datasets", "train_dataset.csv")
TEST_CSV_PATH = os.path.join("datasets", "test_dataset.csv")
AUGMENTED_TRAIN_CSV_PATH = os.path.join(
    "datasets",
    "augmented_dataset",
    "train_dataset_augmented_filtered.csv",
)

TASK_ID = "t1"
TUNE_STRATEGY = "partial"
PLOTS_DIR = Path("results") / "plots" / "encoder"

# Baseline defaults for a single run (overridden by grid if --grid_search)
DEFAULT_CONFIG = {
    "num_train_epochs": 20,
    "batch_size": int(FIXED_GRID_CONFIG["batch_size"]),
    "learning_rate": 2e-5,
    "weight_decay": float(FIXED_GRID_CONFIG["weight_decay"]),
    "grad_accum": 16,
    "dropout": 0.1,
    "max_length": 2048,
}

TARGET_COLUMN = "clarity_label"
ARG1_KEY = "question"
ARG2_KEY = "interview_answer"

hf_logging.set_verbosity_error()


def _fs_safe_model_name(name: str) -> str:
    normalized = (
        name.strip()
        .replace(os.sep, "-")
        .replace("/", "-")
        .replace("\\", "-")
    )
    safe = re.sub(r"[^0-9A-Za-z._-]+", "-", normalized).strip("-._")
    return safe or "model"


def _ensure_enriched_column(df: pd.DataFrame) -> None:
    if "enriched_input" in df.columns:
        return
    if "q" in df.columns:
        df["enriched_input"] = df["q"]
        return
    if "question" not in df.columns or "interview_question" not in df.columns:
        raise KeyError("Missing columns needed for enriched input: question/interview_question.")
    df["enriched_input"] = df.apply(
        lambda row: (
            f"Target question: {row['question']}\n\n"
            f"Full interviewer turn (context): {row['interview_question']}"
        ),
        axis=1,
    )


def _get_meta_fallback(model_name: str) -> Dict[str, str]:
    if model_name in MODEL_METADATA:
        return MODEL_METADATA[model_name]
    return {"arch": "longformer", "lang": "en", "size": "base"}


def _build_output_slug(
    model_name: str,
    param_mode: str,
    unfreeze_ratio: Optional[float] = None,
    truncation: Optional[str] = None,
    dataset: Optional[str] = None,
    input_mode: Optional[str] = None,
    global_attention: Optional[str] = None,
    max_length: Optional[int] = None,
    learning_rate: Optional[float] = None,
    dropout: Optional[float] = None,
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

    slug = (
        f"{TASK_ID}_{meta['arch']}_{meta['lang']}_{meta['size']}"
        f"_{strategy_tag}_{param_mode}_defaultHead_{trunc_tag}"
    )

    if dataset:
        if dataset == "original":
            dataset = "originalData"
        elif dataset == "augmented":
            dataset = "augmentedData"
        slug = f"{slug}_{dataset}"

    if input_mode:
        slug = f"{slug}_{input_mode}"

    if global_attention:
        slug = f"{slug}_gattn{global_attention}"

    if max_length:
        slug = f"{slug}_L{int(max_length)}"

    if learning_rate is not None:
        lr_tag = f"{learning_rate:.0e}".replace("+", "")
        slug = f"{slug}_lr{lr_tag}"

    if dropout is not None:
        slug = f"{slug}_do{dropout:.2f}"

    return _fs_safe_model_name(slug)


def _resolve_model_name(_user_name: str) -> str:
    return LONGFORMER_MODEL


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


# Trainer with weighted loss + live epoch table
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


# Freezing / partial unfreezing logic + summary
from collections import defaultdict


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

    # Longformer uses base.encoder.layer typically (Roberta-like)
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
        if ".encoder.layer." in name:
            parts = name.split(".")
            idx = parts.index("layer")
            top_group = ".".join(parts[:idx + 2])
        else:
            top_group = ".".join(name.split(".")[:2])
        grouped[top_group].append(name)

    for group, names in sorted(grouped.items()):
        lines.append(f"✅ {group}: ({len(names)} params)")

    return lines


# Progress saving + plots
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
# Longformer global attention mask builder
# -----------------------------------------------------------------------------
def build_global_attention_mask(
    input_ids: List[List[int]],
    *,
    mode: str,
    sep_token_id: Optional[int],
) -> List[List[int]]:
    masks: List[List[int]] = []
    for ids in input_ids:
        m = [0] * len(ids)
        if len(m) == 0:
            masks.append(m)
            continue

        # always global on <s>
        m[0] = 1

        if mode == "cls_question" and sep_token_id is not None:
            try:
                first_sep = ids.index(sep_token_id)
                for j in range(0, min(first_sep + 1, len(m))):
                    m[j] = 1
            except ValueError:
                pass

        masks.append(m)

    return masks


def _apply_dropout_to_config(cfg: AutoConfig, dropout: float) -> AutoConfig:
    # Common names
    if hasattr(cfg, "hidden_dropout_prob"):
        cfg.hidden_dropout_prob = float(dropout)
    if hasattr(cfg, "attention_probs_dropout_prob"):
        cfg.attention_probs_dropout_prob = float(dropout)

    # Newer HF configs sometimes use classifier_dropout
    if hasattr(cfg, "classifier_dropout"):
        cfg.classifier_dropout = float(dropout)

    # Some Longformer configs may have dropout fields
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
# Core training function (default head only)
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
    grad_accum: int,
    learning_rate: float,
    num_epochs: int,
    weight_decay: float,
    unfreeze_ratio: float,
    truncation: str,
    max_length: int,
    global_attention: str,
    dropout: float,
) -> Tuple[Trainer, Dict[int, str], Dict[str, int], Any, Dict[str, Any]]:
    unique_labels = sorted(train_df[label_col].dropna().unique())
    label2id: Dict[str, int] = {lab: i for i, lab in enumerate(unique_labels)}
    id2label: Dict[int, str] = {i: lab for lab, i in label2id.items()}

    train_df = train_df.dropna(subset=[arg1_key, arg2_key, label_col]).copy()
    dev_df = dev_df.dropna(subset=[arg1_key, arg2_key, label_col]).copy()
    train_df["label"] = train_df[label_col].map(label2id)
    dev_df["label"] = dev_df[label_col].map(label2id)

    label_counts = train_df["label"].value_counts().sort_index()
    class_weights = (len(train_df) / (len(label_counts) * label_counts)).sort_index()
    class_weights_tensor = torch.tensor(class_weights.to_numpy(), dtype=torch.float)

    _assert_length_multiple_of_attention_window(max_length, int(FIXED_GRID_CONFIG["attention_window"]))

    # Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    sep_id = tokenizer.sep_token_id

    def tokenize(batch):
        if truncation != "head_tail":
            enc = tokenizer(
                batch[arg1_key],
                batch[arg2_key],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_token_type_ids=False,
            )
            enc["global_attention_mask"] = build_global_attention_mask(
                enc["input_ids"],
                mode=global_attention,
                sep_token_id=sep_id,
            )
            return enc

        inputs = tokenizer(
            batch[arg1_key],
            batch[arg2_key],
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

        padded["global_attention_mask"] = build_global_attention_mask(
            padded["input_ids"],
            mode=global_attention,
            sep_token_id=sep_id,
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

    # Model (DEFAULT classification head ONLY) + apply dropout via config
    cfg = AutoConfig.from_pretrained(model_name)
    cfg.num_labels = len(unique_labels)
    cfg.id2label = dict(id2label)
    cfg.label2id = dict(label2id)
    cfg = _apply_dropout_to_config(cfg, dropout=float(dropout))

    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=cfg)

    freeze_summary_lines = apply_partial_unfreezing(model, unfreeze_ratio)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable_params / max(1, total_params)
    print(f"[Info] Trainable params: {trainable_params:,} / {total_params:,} ({pct:.2f}%)")
    print(f"[Info] Dropout set to: {dropout}")

    # TrainingArguments
    training_kwargs = dict(
        output_dir=str(output_dir),
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        disable_tqdm=True,
        report_to="none",

        # We manually restore best weights in-memory
        load_best_model_at_end=False,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,

        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=max(1, int(grad_accum)),
        weight_decay=weight_decay,
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
# Main pipeline + grid search runner
# -----------------------------------------------------------------------------
def _run_one(
    *,
    args,
    resolved_model: str,
    train_full_df: pd.DataFrame,
    test_full_df: pd.DataFrame,
    arg1_key: str,
    learning_rate: float,
    max_length: int,
    dropout: float,
) -> Dict[str, Any]:
    slug = _build_output_slug(
        resolved_model,
        args.param_mode,
        float(FIXED_GRID_CONFIG["unfreeze_ratio"]),
        args.truncation,
        args.dataset,
        args.input_mode,
        global_attention=args.global_attention,
        max_length=max_length,
        learning_rate=learning_rate,
        dropout=dropout,
    )

    STAGE4_DIR = Path(__file__).resolve().parent
    output_dir = STAGE4_DIR / "tmp_checkpoints" / slug
    final_model_dir = STAGE4_DIR / "stage4_trained_models" / slug

    print(
        f"\n[Run] Training Longformer (DEFAULT head only) "
        f"(truncation={args.truncation}, unfreeze_ratio={float(FIXED_GRID_CONFIG['unfreeze_ratio']):.2f}, "
        f"dataset={args.dataset}, input_mode={args.input_mode}, max_length={max_length}, "
        f"batch_size={args.batch_size}, grad_accum={args.grad_accum}, global_attention={args.global_attention}, "
        f"lr={learning_rate}, dropout={dropout}), mode={args.param_mode}"
    )

    trainer, _, _, _, meta = train_model(
        model_name=resolved_model,
        train_df=train_full_df,
        dev_df=test_full_df,
        arg1_key=arg1_key,
        arg2_key=ARG2_KEY,
        label_col=TARGET_COLUMN,
        output_dir=str(output_dir),
        final_model_dir=str(final_model_dir),
        batch_size=int(args.batch_size),
        grad_accum=int(args.grad_accum),
        learning_rate=float(learning_rate),
        num_epochs=int(args.num_epochs),
        weight_decay=float(args.weight_decay),
        unfreeze_ratio=float(FIXED_GRID_CONFIG["unfreeze_ratio"]),
        truncation=args.truncation,
        max_length=int(max_length),
        global_attention=args.global_attention,
        dropout=float(dropout),
    )

    best_eval = meta.get("best_eval_metrics") or {}
    out = {
        "slug": slug,
        "learning_rate": float(learning_rate),
        "max_length": int(max_length),
        "dropout": float(dropout),
        "best_eval_f1_macro": float(best_eval.get("eval_f1_macro", float("nan"))),
        "best_eval_accuracy": float(best_eval.get("eval_accuracy", float("nan"))),
        "best_eval_loss": float(best_eval.get("eval_loss", float("nan"))),
        "final_model_dir": str(final_model_dir),
    }
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Longformer-only encoder fine-tuning (DEFAULT head only) + optional grid search."
    )
    parser.add_argument(
        "--dataset",
        choices=("original", "augmented"),
        default="original",
        help="Which training dataset to use.",
    )
    parser.add_argument(
        "--input_mode",
        choices=("atomic", "enriched"),
        default="atomic",
        help="atomic uses question; enriched uses question+interview_question.",
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
    parser.add_argument(
        "--global_attention",
        choices=("cls", "cls_question"),
        default="cls",
        help="Global attention pattern.",
    )

    # Common training params
    parser.add_argument("--batch_size", type=int, default=int(DEFAULT_CONFIG["batch_size"]))
    parser.add_argument("--grad_accum", type=int, default=int(DEFAULT_CONFIG["grad_accum"]))
    parser.add_argument("--num_epochs", type=int, default=int(DEFAULT_CONFIG["num_train_epochs"]))
    parser.add_argument("--weight_decay", type=float, default=float(DEFAULT_CONFIG["weight_decay"]))

    # Single-run hyperparams (ignored if --grid_search)
    parser.add_argument("--learning_rate", type=float, default=float(DEFAULT_CONFIG["learning_rate"]))
    parser.add_argument("--max_length", type=int, default=int(DEFAULT_CONFIG["max_length"]))
    parser.add_argument("--dropout", type=float, default=float(DEFAULT_CONFIG["dropout"]))

    # Grid search flag
    parser.add_argument(
        "--grid_search",
        action="store_true",
        help="If set, runs exhaustive grid search over SEARCH_SPACE (lr, max_length, dropout).",
    )

    args = parser.parse_args()

    resolved_model = _resolve_model_name(args.model_name)
    print(f"[Info] Using model: {resolved_model}")

    train_path = TRAIN_CSV_PATH if args.dataset == "original" else AUGMENTED_TRAIN_CSV_PATH
    train_full_df = pd.read_csv(train_path)

    if args.input_mode == "enriched":
        _ensure_enriched_column(train_full_df)
        train_full_df.to_csv("temporary.csv", index=False)
        arg1_key = "enriched_input"
    else:
        arg1_key = ARG1_KEY

    train_full_df = train_full_df.dropna(subset=[arg1_key, ARG2_KEY, TARGET_COLUMN]).copy()

    if not os.path.exists(TEST_CSV_PATH):
        raise FileNotFoundError(f"[Fatal] Test CSV not found at {TEST_CSV_PATH}.")

    test_full_df = pd.read_csv(TEST_CSV_PATH)
    if args.input_mode == "enriched":
        _ensure_enriched_column(test_full_df)
    test_full_df = test_full_df.dropna(subset=[arg1_key, ARG2_KEY, TARGET_COLUMN]).copy()

    # -------------------------------------------------------------------------
    # GRID SEARCH
    # -------------------------------------------------------------------------
    if args.grid_search:
        combos = list(product(
            SEARCH_SPACE["learning_rate"],
            SEARCH_SPACE["max_length"],
            SEARCH_SPACE["dropout"],
        ))
        print(f"\n[GridSearch] Running {len(combos)} configurations (exhaustive grid).")

        results: List[Dict[str, Any]] = []
        best_row: Optional[Dict[str, Any]] = None

        for i, (lr, ml, do) in enumerate(combos, start=1):
            print(f"\n[GridSearch] ({i}/{len(combos)}) lr={lr}, max_length={ml}, dropout={do}")
            row = _run_one(
                args=args,
                resolved_model=resolved_model,
                train_full_df=train_full_df,
                test_full_df=test_full_df,
                arg1_key=arg1_key,
                learning_rate=float(lr),
                max_length=int(ml),
                dropout=float(do),
            )
            results.append(row)

            if best_row is None or (row["best_eval_f1_macro"] > best_row["best_eval_f1_macro"]):
                best_row = row

        out_df = pd.DataFrame(results).sort_values(by="best_eval_f1_macro", ascending=False)
        out_path = Path("results") / "eval_logs" / "detailed" / "encoder" / "stage4_longformer_grid_summary.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)

        print(f"\n[GridSearch] Saved summary CSV -> {out_path}")
        if best_row:
            print("\n[GridSearch] BEST CONFIG:")
            print(f"  best_eval_f1_macro: {best_row['best_eval_f1_macro']:.6f}")
            print(f"  learning_rate: {best_row['learning_rate']}")
            print(f"  max_length: {best_row['max_length']}")
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
        arg1_key=arg1_key,
        learning_rate=float(args.learning_rate),
        max_length=int(args.max_length),
        dropout=float(args.dropout),
    )

    print("\n[Final] Longformer DEV RESULTS (BEST restored):")
    print(f"F1-Macro: {row['best_eval_f1_macro']:.4f}")
    print(f"Accuracy: {row['best_eval_accuracy']:.4f} ({row['best_eval_accuracy'] * 100:.2f}%)")
    print(f"Eval Loss: {row['best_eval_loss']:.4f}")
    print(f"Saved to: {row['final_model_dir']}")
    print("[Info] Training run complete.")


if __name__ == "__main__":
    main()
