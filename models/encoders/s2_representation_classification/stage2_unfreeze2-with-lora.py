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
import types
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
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.modeling_outputs import SequenceClassifierOutput
import matplotlib.pyplot as plt

try:
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError as e:
    raise ImportError("Missing dependency: peft. Install with `pip install -U peft`.") from e




ALLOWED_MODEL_NAMES = [
    "roberta-base",
    "roberta-large",
    "bert-base-multilingual-cased",
    "bert-base-uncased",
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
TUNE_STRATEGY = "top2_lora"
PLOTS_DIR = Path("results") / "plots" / "encoder"

FIXED_CONFIG = {
    "num_train_epochs": 20,
    "batch_size": 16,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
}

# LoRA configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Constraints: Top-K encoder layers trainable + LoRA + head
TOP_K_TRAINABLE_LAYERS = 2
TRAIN_CLASSIFIER_HEAD = True
TRAIN_POOLER = False

TARGET_COLUMN = "clarity_label"
ARG1_KEY = "question"
ARG2_KEY = "interview_answer"

# Head tags for slug
HEAD_TAGS = {
    "default": "defaultHead",
    "mlp": "multiLayerHead",
    "avgpool": "avgPoolHead",
}


# -----------------------------------------------------------------------------
# Seeding / metrics
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Custom heads + forward patch
# -----------------------------------------------------------------------------
class MLPHead(nn.Module):
    # CLS token → Dropout → Linear → GELU → Dropout → Linear → logits
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
        return self.fc2(x)


class AvgPoolHead(nn.Module):
    # all tokens → masked mean pool → Linear → logits
    def __init__(self, hidden_size: int, num_labels: int = 3):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, pooled_embedding: torch.Tensor) -> torch.Tensor:
        return self.classifier(pooled_embedding)


def _filter_forward_args(module: nn.Module, kwargs: Dict[str, object]) -> Dict[str, object]:
    try:
        sig = inspect.signature(module.forward)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return kwargs


def _get_backbone_from_seqcls(model: nn.Module) -> nn.Module:
    for attr in (
        "bert",
        "roberta",
        "deberta",
        "xlm_roberta",
        "distilbert",
        "electra",
        "albert",
        "camembert",
        "mpnet",
        "xlnet",
        "flaubert",
    ):
        if hasattr(model, attr):
            return getattr(model, attr)
    base = getattr(model, "base_model", None)
    return base if base is not None else model


def _masked_mean_pool(last_hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if attention_mask is None:
        return last_hidden.mean(dim=1)
    mask = attention_mask.to(dtype=last_hidden.dtype).unsqueeze(-1)  # [B,S,1]
    summed = (last_hidden * mask).sum(dim=1)                          # [B,H]
    denom = mask.sum(dim=1).clamp(min=1e-6)                           # [B,1]
    return summed / denom


def _patch_seqcls_head(model: nn.Module, head_type: str) -> None:
    """
    Replace classification head and override forward to use either:
      - mlp: CLS embedding (last_hidden[:,0,:]) -> MLPHead
      - avgpool: masked mean pool -> AvgPoolHead

    head_type == 'default' => no patch
    """
    head_type = (head_type or "default").strip().lower()
    if head_type == "default":
        return
    if head_type not in ("mlp", "avgpool"):
        raise ValueError(f"Unsupported --head_type '{head_type}'. Use default|mlp|avgpool.")

    hidden_size = int(getattr(getattr(model, "config", None), "hidden_size", 0) or 0)
    if hidden_size <= 0:
        hidden_size = int(getattr(getattr(model, "config", None), "d_model", 0) or 0)
    if hidden_size <= 0:
        raise ValueError("Could not infer hidden size from model.config (hidden_size/d_model).")

    num_labels = int(getattr(getattr(model, "config", None), "num_labels", 0) or 0)
    if num_labels <= 0:
        raise ValueError("Could not infer num_labels from model.config.")

    new_head = MLPHead(hidden_size, num_labels) if head_type == "mlp" else AvgPoolHead(hidden_size, num_labels)

    if hasattr(model, "classifier"):
        model.classifier = new_head
        head_attr = "classifier"
    elif hasattr(model, "score"):
        model.score = new_head
        head_attr = "score"
    else:
        model.classifier = new_head
        head_attr = "classifier"

    setattr(model, "head_type", head_type)
    setattr(model, "_head_attr_name", head_attr)

    def _forward_patched(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,  # WeightedTrainer computes weighted loss
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        return_dict = True if return_dict is None else return_dict

        backbone = _get_backbone_from_seqcls(self)

        payload = dict(kwargs)
        payload.update(
            dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
        )
        payload = _filter_forward_args(backbone, payload)
        outputs = backbone(**payload)

        if isinstance(outputs, (tuple, list)):
            last_hidden = outputs[0]
            hidden_states = outputs[2] if len(outputs) > 2 else None
            attentions = outputs[3] if len(outputs) > 3 else None
        else:
            last_hidden = getattr(outputs, "last_hidden_state", None)
            hidden_states = getattr(outputs, "hidden_states", None)
            attentions = getattr(outputs, "attentions", None)

        if last_hidden is None:
            raise RuntimeError("Backbone did not return last_hidden_state.")

        if getattr(self, "head_type", "mlp") == "mlp":
            pooled = last_hidden[:, 0, :]
        else:
            pooled = _masked_mean_pool(last_hidden, attention_mask)

        drop = getattr(self, "dropout", None)
        if isinstance(drop, nn.Module):
            pooled = drop(pooled)

        head_name = getattr(self, "_head_attr_name", "classifier")
        head = getattr(self, head_name)
        logits = head(pooled)

        return SequenceClassifierOutput(
            logits=logits,
            hidden_states=hidden_states,
            attentions=attentions,
        )

    model.forward = types.MethodType(_forward_patched, model)


# -----------------------------------------------------------------------------
# Weighted Trainer + live epoch metrics table
# -----------------------------------------------------------------------------
def _epoch_to_int(epoch_value) -> int:
    epoch_float = float(epoch_value)
    epoch_idx = int(math.floor(epoch_float + 0.5))
    return max(1, epoch_idx)


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
        self.freeze_summary_lines: list[str] = []

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
        print(header)
        print("-" * len(header))
        self._table_header = header
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


class FullStateWeightedTrainer(WeightedTrainer):
    """
    Because we train base weights (top-K) + LoRA, force full checkpoints so
    load_best_model_at_end can restore everything correctly.
    """
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if getattr(self.model, "config", None) is not None:
            self.model.config.save_pretrained(output_dir)

        torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))


# -----------------------------------------------------------------------------
# Saving training progress + plots
# -----------------------------------------------------------------------------
def _save_training_progress(trainer: Trainer, final_model_dir: str, eval_metrics: Dict[str, float] | None = None) -> None:
    if not isinstance(trainer, WeightedTrainer):
        return
    header = trainer._table_header
    lines = trainer._table_lines
    if not lines:
        return

    # pick best epoch by best_metric if possible
    best_metric = getattr(getattr(trainer, "state", None), "best_metric", None)
    best_epoch_idx = None
    best_row = None

    if best_metric is not None and isinstance(best_metric, (int, float)):
        for epoch_idx, row in trainer._epoch_cache.items():
            f1_macro = row.get("f1_macro")
            if f1_macro is None:
                continue
            if abs(f1_macro - float(best_metric)) < 1e-8:
                best_epoch_idx = epoch_idx
                best_row = row
                break

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

    if eval_metrics is None:
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
            f.write("\n[Final evaluation metrics]\n")
            summary_keys = [
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
            for key in summary_keys:
                if key not in eval_metrics:
                    continue
                val = eval_metrics[key]
                if isinstance(val, float):
                    f.write(f"  {key}: {val:.4f}\n")
                else:
                    f.write(f"  {key}: {val}\n")

            step_val = getattr(trainer.state, "global_step", None)
            if step_val is not None:
                try:
                    f.write(f"  step: {int(step_val)}\n")
                except Exception:
                    f.write(f"  step: {step_val}\n")

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
# LoRA target module selection (robust)
# -----------------------------------------------------------------------------
def _linear_module_names(model) -> list[str]:
    return [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]


def _linear_leaf_names(model) -> set[str]:
    return {n.split(".")[-1] for n in _linear_module_names(model)}


def _count_leaf_matches(model, leaf: str) -> int:
    cnt = 0
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and name.split(".")[-1] == leaf:
            cnt += 1
    return cnt


def _choose_lora_targets(model) -> list[str]:
    leafs = _linear_leaf_names(model)
    model_type = (getattr(getattr(model, "config", None), "model_type", "") or "").lower()

    if "deberta" in model_type:
        candidate_sets = [
            ["query_proj", "value_proj"],
            ["q_proj", "v_proj"],
            ["query", "value"],
            ["in_proj"],
        ]
    else:
        candidate_sets = [
            ["query", "value"],
            ["q_proj", "v_proj"],
            ["query_proj", "value_proj"],
            ["in_proj"],
        ]

    for targets in candidate_sets:
        if all(t in leafs for t in targets):
            total_matches = sum(_count_leaf_matches(model, t) for t in targets)
            if total_matches > 0:
                return targets

    attentionish = [t for t in ["query", "value", "q_proj", "v_proj", "query_proj", "value_proj", "in_proj"] if t in leafs]
    if attentionish:
        total_matches = sum(_count_leaf_matches(model, t) for t in attentionish)
        if total_matches > 0:
            return attentionish

    sample_leafs = sorted(list(leafs))[:80]
    raise ValueError(
        "Could not find any LoRA target modules in this model. "
        f"model_type='{model_type}', linear leaf names sample={sample_leafs}"
    )


def _detect_modules_to_save(base_model) -> list[str] | None:
    mods: list[str] = []
    if hasattr(base_model, "classifier"):
        mods.append("classifier")
    if hasattr(base_model, "score"):
        mods.append("score")
    return mods or None


def _assert_lora_injected(model) -> None:
    lora_params = [n for n, p in model.named_parameters() if "lora_" in n]
    if not lora_params:
        raise RuntimeError("LoRA injection failed: no parameters containing 'lora_' were found.")
    print(f"[Info] LoRA injection OK. Example LoRA params: {lora_params[:5]}")


def _unwrap_backbone_from_peft(model: torch.nn.Module) -> torch.nn.Module:
    m = model
    bm = getattr(m, "base_model", None)
    if bm is not None:
        inner = getattr(bm, "model", None)
        m = inner if inner is not None else bm

    for attr in ("bert", "roberta", "deberta", "xlm_roberta", "distilbert"):
        if hasattr(m, attr):
            return getattr(m, attr)

    return m


def apply_topk_unfreezing(model: torch.nn.Module, top_k: int) -> list[str]:
    lines: list[str] = []
    base = _unwrap_backbone_from_peft(model)

    # Freeze embeddings
    if hasattr(base, "embeddings"):
        for p in base.embeddings.parameters():
            p.requires_grad = False
        msg = "[Freeze] Embedding layer parameters frozen."
        print(msg); lines.append(msg)
    else:
        msg = "⚠️ Could not apply embedding freezing. No 'embeddings' on base model."
        print(msg); lines.append(msg)

    # Freeze all encoder layers; unfreeze last top_k
    if hasattr(base, "encoder") and hasattr(base.encoder, "layer"):
        encoder_layers = list(base.encoder.layer)
        total_layers = len(encoder_layers)

        if total_layers == 0:
            msg = "⚠️ Encoder has no layers; skipping encoder freezing."
            print(msg); lines.append(msg)
        else:
            for layer in encoder_layers:
                for p in layer.parameters():
                    p.requires_grad = False

            top_k = int(top_k)
            top_k = max(0, min(top_k, total_layers))
            start = total_layers - top_k

            for i in range(start, total_layers):
                for p in encoder_layers[i].parameters():
                    p.requires_grad = True

            msg = f"*_Top-K unfreezing: top_k={top_k} (layers {start}..{total_layers - 1}), total_layers={total_layers}"
            print(msg); lines.append(msg)
    else:
        msg = "⚠️ Could not apply encoder layer freezing. Unexpected model structure."
        print(msg); lines.append(msg)

    return lines


def _set_trainables(
    model: torch.nn.Module,
    train_head: bool = TRAIN_CLASSIFIER_HEAD,
    train_pooler: bool = TRAIN_POOLER,
    keep_existing_trainables: bool = True,
) -> None:
    for n, p in model.named_parameters():
        is_lora = ("lora_" in n)

        is_head = False
        if train_head:
            if ".classifier." in n or n.startswith("classifier.") or ".classifier" in n:
                is_head = True
            if ".score." in n or n.startswith("score.") or ".score" in n:
                is_head = True

        is_pool = False
        if train_pooler:
            if ".pooler." in n or n.startswith("pooler.") or ".pooler" in n:
                is_pool = True

        want = is_lora or is_head or is_pool
        if keep_existing_trainables and p.requires_grad:
            want = True
        p.requires_grad = want


def summarize_trainables(model: torch.nn.Module) -> list[str]:
    from collections import defaultdict

    lines: list[str] = []

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = 100.0 * trainable / total if total > 0 else 0.0

    msg = f"*_Total parameters:     {total:,}"
    print(msg); lines.append(msg)
    msg = f"*__Trainable parameters: {trainable:,}"
    print(msg); lines.append(msg)
    msg = f"*__Trainable ratio:      {ratio:.4f}%"
    print(msg); lines.append(msg)

    grouped: dict[str, list[str]] = defaultdict(list)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        parts = name.split(".")
        if "encoder" in parts and "layer" in parts:
            li = parts.index("layer")
            top_group = ".".join(parts[: li + 2]) if li + 1 < len(parts) else ".".join(parts[: li + 1])
        else:
            top_group = ".".join(parts[:2])

        grouped[top_group].append(name)

    for group, names in sorted(grouped.items()):
        msg = f"✅ {group}: ({len(names)} params)"
        print(msg)
        lines.append(msg)

    return lines


def _build_output_slug(model_name: str, param_mode: str, head_type: str) -> str:
    if model_name not in MODEL_METADATA:
        raise KeyError(
            f"Model '{model_name}' missing from MODEL_METADATA. "
            "Add it to models/encoders/model_metadata.py."
        )
    meta = MODEL_METADATA[model_name]
    head_tag = HEAD_TAGS.get(head_type, head_type)
    lora_tag = f"top{TOP_K_TRAINABLE_LAYERS}_lora{LORA_R}"
    slug = (
        f"{TASK_ID}_{meta['arch']}_{meta['lang']}_{meta['size']}"
        f"_{lora_tag}_{param_mode}_{head_tag}"
    )
    return _fs_safe_model_name(slug)


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
    head_type: str,
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

    # ✅ fix collator + label naming
    if "label" in train_ds.column_names and "labels" not in train_ds.column_names:
        train_ds = train_ds.rename_column("label", "labels")
    if "label" in dev_ds.column_names and "labels" not in dev_ds.column_names:
        dev_ds = dev_ds.rename_column("label", "labels")

    keep = {"input_ids", "attention_mask", "labels"}
    if "token_type_ids" in train_ds.column_names:
        keep.add("token_type_ids")
    train_remove = [c for c in train_ds.column_names if c not in keep]
    dev_remove = [c for c in dev_ds.column_names if c not in keep]
    if train_remove:
        train_ds = train_ds.remove_columns(train_remove)
    if dev_remove:
        dev_ds = dev_ds.remove_columns(dev_remove)

    # Base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label2id,
    )

    # Patch head BEFORE LoRA injection (unless default)
    _patch_seqcls_head(base_model, head_type=head_type)

    # Freeze everything first (we'll re-enable top2 + LoRA + head)
    for p in base_model.parameters():
        p.requires_grad = False

    # Choose LoRA targets
    targets = _choose_lora_targets(base_model)
    modules_to_save = _detect_modules_to_save(base_model) if TRAIN_CLASSIFIER_HEAD else None

    print(f"[Info] head_type = {head_type}")
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

    # Inject LoRA
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

    # Apply top-2 unfreezing on backbone (base weights)
    freeze_summary_lines: list[str] = []
    freeze_summary_lines.append(f"*_Head type: {head_type} ({HEAD_TAGS.get(head_type, head_type)})")
    freeze_summary_lines += apply_topk_unfreezing(model, TOP_K_TRAINABLE_LAYERS)

    # Enable LoRA + head (preserve already-unfrozen top-2)
    _set_trainables(model, train_head=TRAIN_CLASSIFIER_HEAD, train_pooler=TRAIN_POOLER, keep_existing_trainables=True)

    # Summary of trainables
    freeze_summary_lines += summarize_trainables(model)

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

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
        # safest with patched forward + PEFT
        remove_unused_columns=False,
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

    trainer = FullStateWeightedTrainer(
        class_weights=class_weights_tensor,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    trainer.freeze_summary_lines = freeze_summary_lines

    print("[Info] Starting training (Top-2 + LoRA)…")
    trainer.train()

    # Plot loss curves from the full training history (all epochs actually run)
    run_name = os.path.basename(os.path.normpath(output_dir)) or model_name
    _plot_loss_curves(trainer, run_name=run_name)

    print("[Info] best ckpt:", trainer.state.best_model_checkpoint)
    print("[Info] best metric:", trainer.state.best_metric)
    print("[Info] best ckpt exists?:", os.path.exists(trainer.state.best_model_checkpoint or ""))

    print("[Info] Evaluating (using test set as eval_dataset)…")
    eval_metrics = trainer.evaluate(eval_dataset=dev_ds)
    for k, v in eval_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    _save_training_progress(trainer, final_model_dir, eval_metrics=eval_metrics)

    # Save a standard loadable model folder: merge LoRA into base
    os.makedirs(final_model_dir, exist_ok=True)
    if hasattr(trainer.model, "merge_and_unload"):
        merged = trainer.model.merge_and_unload()  # type: ignore[attr-defined]
        merged.save_pretrained(final_model_dir, safe_serialization=True)
        tokenizer.save_pretrained(final_model_dir)
        print(f"[Info] Saved MERGED full model & tokenizer to {final_model_dir}")
    else:
        trainer.save_model(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        print(f"[Info] Saved model & tokenizer to {final_model_dir}")

    _remove_checkpoint_dirs(output_dir)
    return trainer, id2label, label2id, tokenizer


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Top-2 encoder layers + LoRA fine-tuning with selectable CLS head.")
    parser.add_argument(
        "--model_name",
        required=True,
        help="Base HF model key (e.g., xlmr, roberta, mbert, deberta) or 'all' to train every listed model.",
    )
    parser.add_argument("--param_mode", required=True, choices=("fixed",), help="Parameter mode to use.")
    parser.add_argument(
        "--head_type",
        default="default",
        choices=("default", "mlp", "avgpool"),
        help="default = HF head | mlp = CLS→Dropout→Linear→GELU→Dropout→Linear | avgpool = masked mean pool→Linear",
    )
    args = parser.parse_args()

    model_names = _expand_models(args.model_name)
    config = FIXED_CONFIG

    for resolved_model in model_names:
        print(
            f"\n[Run] Training {resolved_model} with Top-{TOP_K_TRAINABLE_LAYERS} + LoRA(r={LORA_R}) "
            f"(head_type={args.head_type}), mode={args.param_mode}"
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

        slug = _build_output_slug(resolved_model, args.param_mode, args.head_type)
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
            head_type=args.head_type,
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
        print("[Info] Training run complete.")


if __name__ == "__main__":
    main()
