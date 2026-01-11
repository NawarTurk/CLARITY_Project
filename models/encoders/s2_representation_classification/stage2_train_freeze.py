import sys, os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

from models.encoders.model_metadata import MODEL_METADATA

import argparse
import inspect
import math
import time
import os
import random
import re
import shutil
import json
from pathlib import Path
from typing import Dict, Optional

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
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.utils import WEIGHTS_NAME, SAFE_WEIGHTS_NAME

import matplotlib.pyplot as plt


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
TUNE_STRATEGY = "partial"  # partial unfreezing of encoder layers; embeddings frozen
PLOTS_DIR = Path("results") / "plots" / "encoder"

FIXED_CONFIG = {
    "num_train_epochs": 20,
    "batch_size": 16,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
}



TARGET_COLUMN = "clarity_label"
ARG1_KEY = "question"
ARG2_KEY = "interview_answer"

HEAD_TAGS = {
    "mlp": "multiLayerHead",     # CLS → Dropout → Linear → GELU → Dropout → Linear
    "avgpool": "avgPoolHead",    # all tokens → masked average pooling → Linear
    "default": "defaultHead"    # Truncations
}


def _build_output_slug(
    model_name: str,
    param_mode: str,
    head_type: str,
    unfreeze_ratio: Optional[float] = None,
    truncation: Optional[str] = None,
) -> str:
    if model_name not in MODEL_METADATA:
        raise KeyError(
            f"Model '{model_name}' missing from MODEL_METADATA. "
            "Add it to models/encoders/model_metadata.py."
        )
    meta = MODEL_METADATA[model_name]
    head_tag = HEAD_TAGS.get(head_type, head_type)
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
        f"_{strategy_tag}_{param_mode}_{head_tag}_{trunc_tag}"
    )
    return _fs_safe_model_name(slug)


# Extend head tags to include Hugging Face's default classification head.
HEAD_TAGS = {**HEAD_TAGS, "default": "defaultHead"}


def _resolve_model_name(user_name: str) -> str:
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
    """Compute accuracy plus macro/micro/weighted F1 for evaluation."""
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
    """Format ETA like 00m22s (and include hours if needed)."""
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
# Heads
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


class AvgPoolHead(nn.Module):
    # all token embeddings → masked average pooling → Linear → num_labels logits
    def __init__(self, hidden_size: int, num_labels: int = 3):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, pooled_embedding: torch.Tensor) -> torch.Tensor:
        return self.classifier(pooled_embedding)


def _filter_forward_args(module: nn.Module, kwargs: Dict[str, object]) -> Dict[str, object]:
    """Pass only kwargs that exist in the target module.forward signature."""
    try:
        sig = inspect.signature(module.forward)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return kwargs


class CLSHeadSequenceClassifier(nn.Module):
    """
    Backbone: AutoModel(...)
    Head:
      - mlp: CLS token → Dropout → Linear → GELU → Dropout → Linear → logits
      - avgpool: all tokens → masked average pooling → Linear → logits

    Important:
      - forward has explicit HF-style arguments so Trainer does NOT drop token columns.
      - save_pretrained writes weights in HF-compatible filenames so Trainer can checkpoint + reload best.
    """

    def __init__(
        self,
        model_name: str,
        *,
        num_labels: int,
        id2label: Dict[int, str],
        label2id: Dict[str, int],
        head_type: str = "mlp",
    ):
        super().__init__()

        head_type = (head_type or "mlp").strip().lower()
        if head_type not in ("mlp", "avgpool"):
            raise ValueError(f"Unsupported head_type='{head_type}'. Use 'mlp' or 'avgpool'.")

        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = int(num_labels)
        config.id2label = dict(id2label)
        config.label2id = dict(label2id)

        self.config = config
        self.model_name = model_name
        self.head_type = head_type

        self.base_model = AutoModel.from_pretrained(model_name, config=config)

        hidden_size = int(getattr(config, "hidden_size", None) or getattr(config, "d_model", 0) or 0)
        if hidden_size <= 0:
            raise ValueError(f"Could not determine hidden size from config for model: {model_name}")

        if head_type == "mlp":
            self.classifier = MLPHead(hidden_size=hidden_size, num_labels=num_labels)
        else:
            self.classifier = AvgPoolHead(hidden_size=hidden_size, num_labels=num_labels)

    @staticmethod
    def _masked_mean_pool(last_hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if attention_mask is None:
            return last_hidden.mean(dim=1)
        mask = attention_mask.to(dtype=last_hidden.dtype).unsqueeze(-1)  # [B, S, 1]
        summed = (last_hidden * mask).sum(dim=1)                          # [B, H]
        denom = mask.sum(dim=1).clamp(min=1e-6)                           # [B, 1]
        return summed / denom

    # ✅ Explicit args: prevents Trainer from stripping token columns
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,  # ignored (WeightedTrainer computes weighted loss)
        **kwargs,
    ):
        payload = dict(kwargs)
        payload["input_ids"] = input_ids
        payload["attention_mask"] = attention_mask
        payload["token_type_ids"] = token_type_ids
        payload.setdefault("return_dict", True)

        filtered = _filter_forward_args(self.base_model, payload)
        outputs = self.base_model(**filtered)

        last_hidden = getattr(outputs, "last_hidden_state", None)
        if last_hidden is None:
            raise RuntimeError("Backbone did not return last_hidden_state; cannot pool.")

        if self.head_type == "mlp":
            embedding = last_hidden[:, 0, :]  # CLS / <s>
        else:
            embedding = self._masked_mean_pool(last_hidden, attention_mask)

        logits = self.classifier(embedding)
        return SequenceClassifierOutput(logits=logits)

    # ✅ HF-like save_pretrained so Trainer checkpoints + best-model reload work
    def save_pretrained(
        self,
        save_directory: str,
        is_main_process: bool = True,
        state_dict=None,
        save_function=torch.save,
        safe_serialization: bool = False,
        **kwargs,
    ) -> None:
        if not is_main_process:
            return

        os.makedirs(save_directory, exist_ok=True)

        # Save config
        try:
            self.config.save_pretrained(save_directory)
        except Exception:
            pass

        # Save metadata for your own bookkeeping
        meta = {
            "model_name": self.model_name,
            "head_type": self.head_type,
            "num_labels": int(self.config.num_labels),
            "id2label": dict(getattr(self.config, "id2label", {}) or {}),
            "label2id": dict(getattr(self.config, "label2id", {}) or {}),
        }
        with open(os.path.join(save_directory, "head_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        # Save weights in the exact filenames Trainer expects
        sd = state_dict if state_dict is not None else self.state_dict()

        if safe_serialization:
            try:
                from safetensors.torch import save_file  # type: ignore
                save_file(sd, os.path.join(save_directory, SAFE_WEIGHTS_NAME))
                return
            except Exception:
                # fall back to torch save below
                pass

        save_function(sd, os.path.join(save_directory, WEIGHTS_NAME))


# -----------------------------------------------------------------------------
# Trainer with weighted loss + live epoch table
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
        self._table_lines: list[str] = []
        self._printed_epochs: set[int] = set()

        self.freeze_summary_lines: list[str] = []
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

        # Ensure epoch/step present
        if self.state.epoch is not None and "epoch" not in logs:
            logs["epoch"] = self.state.epoch
        if self.state.global_step is not None and "step" not in logs:
            logs["step"] = self.state.global_step

        # Do our own bookkeeping without printing Hugging Face's default `{...}` log lines
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
from collections import defaultdict


def _get_base_model(model: torch.nn.Module) -> torch.nn.Module:
    base = getattr(model, "base_model", None)
    return base if base is not None else model


def apply_partial_unfreezing(model: torch.nn.Module, unfreeze_ratio: float) -> list[str]:
    lines: list[str] = []

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
    if hasattr(base, "encoder") and hasattr(base.encoder, "layer"):
        encoder_layers = list(base.encoder.layer)
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
    else:
        lines.append("⚠️ Could not apply encoder layer freezing. Unexpected model structure.")

    # 3) Trainable parameter summary (includes classifier head, not frozen)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = 100.0 * trainable / total if total > 0 else 0.0

    lines.append(f"*_Total parameters:     {total:,}")
    lines.append(f"*__Trainable parameters: {trainable:,}")
    lines.append(f"*__Trainable ratio:      {ratio:.2f}%")

    grouped: Dict[str, list[str]] = defaultdict(list)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # group encoder layers nicely when present
        if ".encoder.layer." in name:
            parts = name.split(".")
            idx = parts.index("layer")  # ... encoder layer {i} ...
            top_group = ".".join(parts[:idx + 2])  # up to layer index
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
    eval_metrics: Optional[Dict[str, float]] = None,
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

        _write_metrics_block(f, "Final evaluation metrics", eval_metrics, step=step_int)

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
        plt.plot(eval_epochs, eval_losses, marker="o", label="Eval (test) loss")
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
    unfreeze_ratio: float,
    head_type: str,
    truncation: str = "head",
) -> None:
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

    # Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    def tokenize(batch):
        # Default: standard truncation
        if truncation != "head_tail":
            return tokenizer(
                batch[arg1_key],
                batch[arg2_key],
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
            )

        # head_tail: keep first 256 and last 256 tokens for long sequences
        inputs = tokenizer(
            batch[arg1_key],
            batch[arg2_key],
            truncation=False,
            padding=False,
            return_attention_mask=True,
        )

        input_ids_list = inputs["input_ids"]
        attn_list = inputs["attention_mask"]
        type_ids_list = inputs.get("token_type_ids")

        head_len = MAX_LENGTH // 2
        tail_len = MAX_LENGTH - head_len

        new_input_ids = []
        new_attn = []
        new_type_ids = [] if type_ids_list is not None else None

        for i in range(len(input_ids_list)):
            ids = input_ids_list[i]
            mask = attn_list[i]

            # Short sequences: keep as-is
            if len(ids) <= MAX_LENGTH:
                new_input_ids.append(ids)
                new_attn.append(mask)
                if new_type_ids is not None:
                    new_type_ids.append(type_ids_list[i])
                continue

            # Long sequences: take head and tail
            head_ids = ids[:head_len]
            tail_ids = ids[-tail_len:]
            head_mask = mask[:head_len]
            tail_mask = mask[-tail_len:]

            truncated_ids = head_ids + tail_ids
            truncated_mask = head_mask + tail_mask

            new_input_ids.append(truncated_ids)
            new_attn.append(truncated_mask)

            if new_type_ids is not None:
                t_ids = type_ids_list[i]
                new_type_ids.append(t_ids[:head_len] + t_ids[-tail_len:])

        encoded = {"input_ids": new_input_ids, "attention_mask": new_attn}
        if new_type_ids is not None:
            encoded["token_type_ids"] = new_type_ids

        padded = tokenizer.pad(
            encoded,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors=None,
        )
        return padded

    train_ds = Dataset.from_pandas(train_df, preserve_index=False).map(tokenize, batched=True)
    dev_ds = Dataset.from_pandas(dev_df, preserve_index=False).map(tokenize, batched=True)

    # ✅ Ensure label column is exactly "labels"
    if "label" in train_ds.column_names and "labels" not in train_ds.column_names:
        train_ds = train_ds.rename_column("label", "labels")
    if "label" in dev_ds.column_names and "labels" not in dev_ds.column_names:
        dev_ds = dev_ds.rename_column("label", "labels")

    # ✅ Remove raw text columns so the collator only sees model inputs + labels
    keep = {"input_ids", "attention_mask", "labels"}
    if "token_type_ids" in train_ds.column_names:
        keep.add("token_type_ids")
    train_remove = [c for c in train_ds.column_names if c not in keep]
    dev_remove = [c for c in dev_ds.column_names if c not in keep]
    if train_remove:
        train_ds = train_ds.remove_columns(train_remove)
    if dev_remove:
        dev_ds = dev_ds.remove_columns(dev_remove)

    # Model (either HF default head or custom CLS head)
    if head_type == "default":
        # Use Hugging Face's standard classification head, matching Stage 1
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(unique_labels),
            id2label=id2label,
            label2id=label2id,
        )
    else:
        # Custom CLS architecture (mlp / avgpool)
        model = CLSHeadSequenceClassifier(
            model_name=model_name,
            num_labels=len(unique_labels),
            id2label=id2label,
            label2id=label2id,
            head_type=head_type,
        )

    # Apply partial unfreezing and capture summary lines
    freeze_summary_lines = apply_partial_unfreezing(model, unfreeze_ratio)

    # TrainingArguments
    training_kwargs = dict(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        disable_tqdm=True,
        report_to="none",
        load_best_model_at_end=False,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=weight_decay,
        seed=SEED,
        data_seed=SEED,
        fp16=torch.cuda.is_available(),
        # ✅ prevent Trainer from stripping token columns due to signature mismatch
        remove_unused_columns=False,
        # try to avoid safetensors surprises; still works if your env uses it anyway
        save_safetensors=False,
        use_safetensors=False,
    )

    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    # Map evaluation_strategy → eval_strategy if using a newer transformers API
    if "evaluation_strategy" not in ta_params and "eval_strategy" in ta_params:
        training_kwargs["eval_strategy"] = training_kwargs.pop("evaluation_strategy")

    # Drop any kwargs not supported by this install
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
        eval_dataset=dev_ds,  # test acts as eval set
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    trainer.freeze_summary_lines = freeze_summary_lines

    print("[Info] Starting training with partial unfreezing…")
    trainer.train()

    # Plot loss curves from the full training history (all epochs actually run)
    run_name = os.path.basename(os.path.normpath(output_dir)) or model_name
    _plot_loss_curves(trainer, run_name=run_name)

    print("[Info] Evaluating (using dev/test set as eval_dataset)…")
    eval_metrics = trainer.evaluate(eval_dataset=dev_ds)
    trainer.last_eval_metrics = eval_metrics

    step_val = getattr(trainer.state, "global_step", None)
    try:
        step_int = int(step_val) if step_val is not None else None
    except (TypeError, ValueError):
        step_int = None

    _print_metrics_block("Final evaluation metrics", eval_metrics, step=step_int)

    print("\n[Layer freezing / trainable parameter summary]")
    for line in freeze_summary_lines:
        print(line)

    _save_training_progress(trainer, final_model_dir, eval_metrics=eval_metrics)

    # Save model + tokenizer (trainer.model is best if load_best_model_at_end=True)
    os.makedirs(final_model_dir, exist_ok=True)
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print(f"[Info] Saved fine-tuned model & tokenizer to {final_model_dir}")

    _remove_checkpoint_dirs(output_dir)

    return trainer, id2label, label2id, tokenizer


# -----------------------------------------------------------------------------
# 4) Pipeline
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Encoder fine-tuning with partial unfreezing + selectable head (mlp/avgpool)."
    )
    parser.add_argument(
        "--model_name",
        required=True,
        help="Base HF model key (e.g., xlmr, roberta, mbert, deberta) or 'all' to train every listed model.",
    )
    parser.add_argument(
        "--param_mode",
        required=True,
        choices=("fixed",),
        help="Parameter mode to use.",
    )
    parser.add_argument(
        "--head_type",
        default="mlp",
        choices=("default", "mlp", "avgpool"),
        help="mlp = CLS→Dropout→Linear→GELU→Dropout→Linear | avgpool = masked mean pool→Linear",
    )
    parser.add_argument(
        "--unfreeze_ratio",
        type=float,
        default=None,
        help="Run only one ratio (e.g., 0.25 / 0.50 / 0.75). If omitted, runs (0.25, 0.50, 0.75).",
    )
    parser.add_argument(
        "--truncation",
        default="head",
        choices=("head", "head_tail"),
        help="Truncation strategy label used for naming/checkpointing.",
    )
    args = parser.parse_args()

    model_names = _expand_models(args.model_name)
    config = FIXED_CONFIG

    if args.unfreeze_ratio is None:
        unfreeze_ratios = (0.25, 0.50, 0.75)
    else:
        unfreeze_ratios = (float(args.unfreeze_ratio),)

    for resolved_model in model_names:
        for unfreeze_ratio in unfreeze_ratios:
            print(
                f"\n[Run] Training {resolved_model} "
                f"(head_type={args.head_type}, truncation={args.truncation}, "
                f"unfreeze_ratio={unfreeze_ratio:.2f}), mode={args.param_mode}"
            )

            train_full_df = pd.read_csv(TRAIN_CSV_PATH)
            train_full_df = train_full_df.dropna(subset=[ARG1_KEY, ARG2_KEY, TARGET_COLUMN]).copy()

            if not os.path.exists(TEST_CSV_PATH):
                raise FileNotFoundError(
                    f"[Fatal] Test CSV not found at {TEST_CSV_PATH}. "
                    "We now require it for eval_loss / early stopping."
                )
            test_full_df = pd.read_csv(TEST_CSV_PATH)
            test_full_df = test_full_df.dropna(subset=[ARG1_KEY, ARG2_KEY, TARGET_COLUMN]).copy()

            slug = _build_output_slug(
                resolved_model,
                args.param_mode,
                args.head_type,
                unfreeze_ratio,
                args.truncation,
            )
            STAGE2_DIR = Path(__file__).resolve().parent
            output_dir = STAGE2_DIR / "tmp_checkpoints" / slug
            final_model_dir = STAGE2_DIR / "stage2_trained_models" / slug

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
                unfreeze_ratio=unfreeze_ratio,
                head_type=args.head_type,
                truncation=args.truncation,
            )

            eval_metrics = getattr(trainer, "last_eval_metrics", None) or trainer.evaluate()
            acc = float(eval_metrics.get("eval_accuracy", 0.0))
            f1_macro = float(eval_metrics.get("eval_f1_macro", 0.0))
            f1_micro = float(eval_metrics.get("eval_f1_micro", 0.0))
            f1_weighted = float(eval_metrics.get("eval_f1_weighted", 0.0))

            print(f"[Info] Eval accuracy (partial): {acc}")
            print(
                f"\n{resolved_model} on DEV RESULTS "
                f"(head_type={args.head_type}, unfreeze_ratio={unfreeze_ratio:.2f}):"
            )
            print(f"Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
            print(f"F1-Macro: {f1_macro:.4f}")
            print(f"F1-Micro: {f1_micro:.4f}")
            print(f"F1-Weighted: {f1_weighted:.4f}")
            print("[Info] Training run complete for this configuration.")


if __name__ == "__main__":
    main()
