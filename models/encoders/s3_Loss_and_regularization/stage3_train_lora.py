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

# LoRA / PEFT
try:
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError as e:
    raise ImportError("Missing dependency: peft. Install with `pip install -U peft`.") from e


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

TARGET_COLUMN = "clarity_label"
ARG1_KEY = "question"
ARG2_KEY = "interview_answer"

# -----------------
# LoRA configuration
# -----------------
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

TRAIN_CLASSIFIER_HEAD = True
TRAIN_POOLER = False

# -----------------
# Head types
# -----------------
HEAD_TAGS = {
    "mlp": "multiLayerHead",     # CLS token → Dropout → Linear → GELU → Dropout → Linear
    "avgpool": "avgPoolHead",    # all tokens → masked average pooling → Dropout → Linear
}


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Dropout helpers (apply dropout to the entire backbone)  ✅
# -----------------------------------------------------------------------------
def apply_dropout_to_hf_config(config, p: float) -> None:
    if config is None:
        return
    p = float(p)
    for key in (
        "hidden_dropout_prob",
        "attention_probs_dropout_prob",
        "classifier_dropout",
        "dropout",
        "attention_dropout",
        "hidden_dropout",
        "pooler_dropout",
        "emb_dropout",
    ):
        if hasattr(config, key):
            try:
                setattr(config, key, p)
            except Exception:
                pass


def patch_model_dropout_modules(model: nn.Module, p: float) -> None:
    p = float(p)
    for m in model.modules():
        # Standard torch dropout
        if isinstance(m, nn.Dropout):
            m.p = p

        # DeBERTa "StableDropout" often uses drop_prob
        if hasattr(m, "drop_prob") and isinstance(getattr(m, "drop_prob", None), (float, int)):
            try:
                setattr(m, "drop_prob", p)
            except Exception:
                pass

        # Some modules store probability in 'p' but aren't nn.Dropout
        if hasattr(m, "p") and isinstance(getattr(m, "p", None), (float, int)) and not isinstance(m, nn.Dropout):
            try:
                setattr(m, "p", p)
            except Exception:
                pass


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
        return self.fc2(x)


class AvgPoolHead(nn.Module):
    # all token embeddings → masked average pooling → Dropout → Linear → num_labels logits
    def __init__(self, hidden_size: int, num_labels: int = 3, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_labels)  # ✅ renamed back to fc

    def forward(self, pooled_embedding: torch.Tensor) -> torch.Tensor:
        x = self.dropout(pooled_embedding)
        return self.fc(x)


def _filter_forward_args(module: nn.Module, kwargs: Dict[str, object]) -> Dict[str, object]:
    """Pass only kwargs that exist in the target module.forward signature."""
    try:
        sig = inspect.signature(module.forward)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return kwargs


def _get_backbone_from_seqcls(model: nn.Module) -> nn.Module:
    """
    For AutoModelForSequenceClassification models, the backbone is usually one of:
    bert, roberta, deberta, xlm_roberta, distilbert, etc.
    """
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
        "rembert",
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


def _patch_seqcls_head(model: nn.Module, head_type: str, dropout: float = 0.1) -> None:
    """
    Replace the classifier head AND patch forward to use:
      - mlp: CLS token embedding (last_hidden[:,0,:]) → MLPHead
      - avgpool: masked mean pool(last_hidden) → AvgPoolHead (with dropout)

    This keeps the model as a HF PreTrainedModel (important for PEFT),
    while ensuring behavior matches your desired CLS strategy.
    """
    head_type = (head_type or "mlp").strip().lower()
    if head_type == "default":
        # Keep the HF default classification head; just record type for metadata.
        setattr(model, "head_type", "default")
        return
    if head_type not in ("mlp", "avgpool"):
        raise ValueError(f"Unsupported --head_type '{head_type}'. Use 'mlp', 'avgpool', or 'default'.")

    hidden_size = int(getattr(getattr(model, "config", None), "hidden_size", 0) or 0)
    if hidden_size <= 0:
        # Some configs use d_model
        hidden_size = int(getattr(getattr(model, "config", None), "d_model", 0) or 0)
    if hidden_size <= 0:
        raise ValueError("Could not infer hidden size from model.config (hidden_size/d_model).")

    num_labels = int(getattr(getattr(model, "config", None), "num_labels", 0) or 0)
    if num_labels <= 0:
        raise ValueError("Could not infer num_labels from model.config.")

    # Replace classifier module
    if head_type == "mlp":
        new_head = MLPHead(hidden_size=hidden_size, num_labels=num_labels)
        if hasattr(new_head, "dropout1") and isinstance(new_head.dropout1, nn.Dropout):
            new_head.dropout1.p = float(dropout)
        if hasattr(new_head, "dropout2") and isinstance(new_head.dropout2, nn.Dropout):
            new_head.dropout2.p = float(dropout)
    else:
        new_head = AvgPoolHead(hidden_size=hidden_size, num_labels=num_labels, dropout=float(dropout))

    # Most seqcls models use .classifier; some use .score
    if hasattr(model, "classifier"):
        model.classifier = new_head
        head_attr = "classifier"
    elif hasattr(model, "score"):
        model.score = new_head
        head_attr = "score"
    else:
        # create classifier if neither exists
        model.classifier = new_head
        head_attr = "classifier"

    # Stash for debugging / saving meta
    setattr(model, "head_type", head_type)
    setattr(model, "_head_attr_name", head_attr)

    # Patch forward with explicit signature so Trainer doesn't drop columns
    def _forward_patched(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,  # ignored (WeightedTrainer computes weighted loss)
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

        # outputs can be tuple or ModelOutput
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
            pooled = last_hidden[:, 0, :]  # CLS/<s>
        else:
            pooled = _masked_mean_pool(last_hidden, attention_mask)

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
# LoRA helper functions (unchanged, but used)
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
    """
    Robust target_modules selection:
    - Tries known naming conventions first
    - Verifies they exist in this specific installed model
    - Picks the first set that yields >0 matches
    """
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

    attentionish = [
        t
        for t in ["query", "value", "q_proj", "v_proj", "query_proj", "value_proj", "in_proj"]
        if t in leafs
    ]
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
    """
    Ensure the classification head is trained + SAVED with the adapter.
    """
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


def _set_trainables(model, train_head: bool = TRAIN_CLASSIFIER_HEAD, train_pooler: bool = TRAIN_POOLER) -> None:
    """
    Train LoRA params + optionally classifier head/pooler.
    NOTE: if you train the head, you must also SAVE it (modules_to_save).
    """
    for n, p in model.named_parameters():
        is_lora = ("lora_" in n)
        is_head = False
        is_pool = False

        if train_head:
            if ".classifier." in n or n.startswith("classifier."):
                is_head = True
            if ".score." in n or n.startswith("score."):
                is_head = True

        if train_pooler:
            if ".pooler." in n or n.startswith("pooler."):
                is_pool = True

        p.requires_grad = (is_lora or is_head or is_pool)


def _trainable_summary_lines(model: nn.Module) -> list[str]:
    """
    Similar vibe to your freeze-summary block, but for LoRA:
    prints total/trainable/ratio + grouped trainables.
    """
    from collections import defaultdict

    lines: list[str] = []
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = 100.0 * trainable / total if total > 0 else 0.0

    lines.append(f"*_Total parameters:     {total:,}")
    lines.append(f"*__Trainable parameters: {trainable:,}")
    lines.append(f"*__Trainable ratio:      {ratio:.2f}%")

    grouped: dict[str, list[str]] = defaultdict(list)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # group attention layers nicely; otherwise by first 2 segments
        if ".encoder.layer." in name:
            parts = name.split(".")
            if "layer" in parts:
                idx = parts.index("layer")
                top_group = ".".join(parts[:idx + 2])
            else:
                top_group = ".".join(parts[:4])
        else:
            top_group = ".".join(name.split(".")[:2])
        grouped[top_group].append(name)

    for group, names in sorted(grouped.items()):
        lines.append(f"✅ {group}: ({len(names)} params)")
    return lines


# -----------------------------------------------------------------------------
# Output naming
# -----------------------------------------------------------------------------
def _build_output_slug(
    model_name: str,
    param_mode: str,
    head_type: str,
    truncation: str | None = None,
    loss_type: str | None = None,
    dropout: float | None = None,
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
    lora_tag = f"lora{LORA_R}"

    slug = (
        f"{TASK_ID}_{meta['arch']}_{meta['lang']}_{meta['size']}"
        f"_{lora_tag}_{param_mode}_{head_tag}_{trunc_tag}"
    )

    if loss_type is not None or dropout is not None:
        loss_tag = (loss_type or "WCE").upper()
        drop_tag = None
        if dropout is not None:
            try:
                drop_tag = int(round(100 * float(dropout)))
            except (TypeError, ValueError):
                drop_tag = None
        if drop_tag is not None:
            loss_tag = f"{loss_tag}_Dropout{drop_tag:02d}"
        slug = f"{slug}_{loss_tag}"

    # Data augmentation tag (Stage 4 will add variants; Stage 3 is NoAug)
    slug = f"{slug}_NoAug"

    return _fs_safe_model_name(slug)


# -----------------------------------------------------------------------------
# Trainer with class-weighted loss + live epoch metrics table
# -----------------------------------------------------------------------------
class WeightedTrainer(Trainer):
    """Trainer with class-weighted (WCE / Focal) loss and a live epoch metrics table."""

    def __init__(
        self,
        class_weights: torch.Tensor,
        *args,
        loss_type: str = "WCE",
        focal_gamma: float = 2.0,
        best_model_dir: Optional[str] = None,
        best_metric_name: str = "eval_f1_macro",
        best_tokenizer=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.float()
        loss_type = (loss_type or "WCE").upper()
        self.loss_type = "FOCAL" if loss_type == "FOCAL" else "WCE"
        self.focal_gamma = float(focal_gamma)
        self._epoch_cache: Dict[int, Dict[str, float]] = {}
        self._printed_header = False
        self._train_start_time: Optional[float] = None
        self._table_header: Optional[str] = None
        self._table_divider: Optional[str] = None
        self._table_lines: list[str] = []
        self._printed_epochs: set[int] = set()

        # saved into training-progress.txt
        self.trainable_summary_lines: list[str] = []
        self.best_metric_name = best_metric_name
        self.best_model_dir = best_model_dir
        self.best_eval_metric: Optional[float] = None
        self.best_eval_metrics: Optional[Dict[str, float]] = None
        self.best_eval_step: Optional[int] = None
        self.best_model_saved = False
        self._best_tokenizer = best_tokenizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        if labels is None:
            labels = inputs.get("label")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Flatten for loss computation
        logits = logits.view(-1, self.model.config.num_labels)
        labels = labels.view(-1)

        if self.loss_type == "FOCAL":
            # Ignore padding labels (if present)
            valid_mask = labels != -100
            if valid_mask.any():
                logits_valid = logits[valid_mask]
                labels_valid = labels[valid_mask]

                log_probs = torch.nn.functional.log_softmax(logits_valid, dim=-1)
                probs = log_probs.exp()

                labels_unsq = labels_valid.unsqueeze(-1)
                log_pt = log_probs.gather(1, labels_unsq).squeeze(1)
                pt = probs.gather(1, labels_unsq).squeeze(1)

                focal_factor = (1.0 - pt).pow(self.focal_gamma)
                class_weights = self.class_weights.to(logits.device)
                alpha = class_weights[labels_valid]

                loss = -alpha * focal_factor * log_pt
                loss = loss.mean()
            else:
                loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        else:
            loss_fct = CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss

    def _maybe_update_best(self, metrics: Dict[str, float]) -> None:
        if not metrics:
            return
        metric_val = metrics.get(self.best_metric_name)
        if metric_val is None:
            return
        try:
            metric_val = float(metric_val)
        except (TypeError, ValueError):
            return
        if self.best_eval_metric is not None and metric_val <= self.best_eval_metric:
            return

        self.best_eval_metric = metric_val
        self.best_eval_metrics = dict(metrics)
        if "epoch" not in self.best_eval_metrics and self.state.epoch is not None:
            self.best_eval_metrics["epoch"] = float(self.state.epoch)
        self.best_eval_step = getattr(self.state, "global_step", None)
        if self.best_model_dir:
            os.makedirs(self.best_model_dir, exist_ok=True)
            self.save_model(self.best_model_dir)
            if self._best_tokenizer is not None:
                self._best_tokenizer.save_pretrained(self.best_model_dir)
            self.best_model_saved = True

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        self._maybe_update_best(metrics)
        return metrics

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
                sec_per_epoch = elapsed / epochs_done
                eta_str = _format_eta_mmss(remaining * sec_per_epoch)

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

        # Keep log history for plotting, but avoid printing raw `{...}` lines
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

    step_val = getattr(trainer, "best_eval_step", None)
    if step_val is None:
        step_val = getattr(trainer.state, "global_step", None)
    try:
        step_int = int(step_val) if step_val is not None else None
    except (TypeError, ValueError):
        step_int = None

    with open(out_path, "w", encoding="utf-8") as f:
        if header is not None:
            f.write(header + "\n")
        f.write((divider or "-" * (len(header) if header else 80)) + "\n")
        for line in lines:
            f.write(line + "\n")

        # Final evaluation metrics block
        f.write("\n[Final evaluation metrics]\n")
        if eval_metrics:
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
                if key not in eval_metrics:
                    continue
                val = eval_metrics[key]
                if isinstance(val, float):
                    f.write(f"  {key}: {val:.4f}\n")
                else:
                    f.write(f"  {key}: {val}\n")
        else:
            f.write("  (no metrics)\n")

        if step_int is not None:
            f.write(f"  step: {step_int}\n")

        # Trainable summary block
        summary = getattr(trainer, "trainable_summary_lines", None)
        if summary:
            f.write("\n[Layer freezing / trainable parameter summary]\n")
            for line in summary:
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


def _write_head_meta(final_model_dir: str, *, head_type: str, lora_r: int, lora_alpha: int, lora_dropout: float, targets: list[str]) -> None:
    os.makedirs(final_model_dir, exist_ok=True)
    meta = {
        "head_type": head_type,
        "head_tag": HEAD_TAGS.get(head_type, head_type),
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_target_modules": targets,
        "train_classifier_head": bool(TRAIN_CLASSIFIER_HEAD),
        "train_pooler": bool(TRAIN_POOLER),
    }
    with open(os.path.join(final_model_dir, "head_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


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
    truncation: str = "head",
    dropout: float = 0.1,
    loss_type: str = "WCE",
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

    # Tokenizer & tokenization
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

    # ✅ Remove raw text columns so collator only sees model inputs + labels
    keep = {"input_ids", "attention_mask", "labels"}
    if "token_type_ids" in train_ds.column_names:
        keep.add("token_type_ids")
    train_remove = [c for c in train_ds.column_names if c not in keep]
    dev_remove = [c for c in dev_ds.column_names if c not in keep]
    if train_remove:
        train_ds = train_ds.remove_columns(train_remove)
    if dev_remove:
        dev_ds = dev_ds.remove_columns(dev_remove)

    # -----------------
    # Model (LoRA) + custom head_type
    # -----------------
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label2id,
    )

    # ✅ Apply dropout to entire backbone (config + instantiated modules) BEFORE LoRA injection
    apply_dropout_to_hf_config(getattr(base_model, "config", None), float(dropout))
    patch_model_dropout_modules(base_model, float(dropout))

    # Patch the head and forward BEFORE LoRA injection
    _patch_seqcls_head(base_model, head_type=head_type, dropout=dropout)

    # Freeze everything first; then re-enable LoRA + head/pooler as desired
    for p in base_model.parameters():
        p.requires_grad = False

    # Choose target_modules that ACTUALLY exist in this model install
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
        save_strategy="no",
        logging_strategy="epoch",
        disable_tqdm=True,
        report_to="none",
        load_best_model_at_end=False,
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
        # keeps columns stable (safe; especially when you patch forward)
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

    trainer = WeightedTrainer(
        class_weights=class_weights_tensor,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        loss_type=loss_type,
        best_model_dir=str(final_model_dir),
        best_metric_name="eval_f1_macro",
        best_tokenizer=tokenizer,
    )

    # Attach summary lines (LoRA trainables)
    trainer.trainable_summary_lines = _trainable_summary_lines(model)

    print("[Info] Starting LoRA training…")
    trainer.train()

    # Plot loss curves from the full training history (all epochs actually run)
    run_name = os.path.basename(os.path.normpath(output_dir)) or model_name
    _plot_loss_curves(trainer, run_name=run_name)

    # Evaluate and save progress with final eval block (best/chosen model)
    print("[Info] Evaluating (using test set as eval_dataset)…")
    eval_metrics = trainer.evaluate(eval_dataset=dev_ds)
    best_metrics = getattr(trainer, "best_eval_metrics", None) or eval_metrics
    for k, v in best_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    _save_training_progress(trainer, final_model_dir, eval_metrics=best_metrics)

    # Save adapter + (if enabled) classifier head via modules_to_save
    os.makedirs(final_model_dir, exist_ok=True)
    if not getattr(trainer, "best_model_saved", False):
        trainer.save_model(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
    _write_head_meta(
        final_model_dir,
        head_type=head_type,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        targets=targets,
    )

    print(f"[Info] Saved LoRA adapter (+head if enabled) & tokenizer to {final_model_dir}")

    _remove_checkpoint_dirs(output_dir)
    return trainer, id2label, label2id, tokenizer


# -----------------------------------------------------------------------------
# Pipeline: load CSVs, train on ALL train, validate on TEST
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning with selectable CLS strategy (mlp / avgpool).")
    parser.add_argument(
        "--truncation",
        default="head",
        choices=("head", "head_tail"),
        help="Truncation strategy label used for naming/checkpointing.",
    )
    parser.add_argument(
        "--model_name",
        required=True,
        help="Base HF model key (e.g., xlmr, roberta, mbert, deberta) or 'all' to train every listed model.",
    )
    parser.add_argument(
        "--param_mode",
        choices=("fixed",),
        default="fixed",
        help="Parameter mode to use (Stage 3 uses fixed).",
    )
    parser.add_argument(
        "--head_type",
        default="mlp",
        choices=("default", "mlp", "avgpool"),
        help="mlp = CLS→Dropout→Linear→GELU→Dropout→Linear | avgpool = masked mean pool→Dropout→Linear",
    )
    parser.add_argument(
        "--loss_fn",
        choices=("WCE", "Focal"),
        default="WCE",
        help="Loss function name (e.g., WCE, Focal).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability (applied to backbone + head in this script).",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=LORA_R,
        help="LoRA rank (currently logged but Stage 3 uses a fixed config).",
    )
    parser.add_argument(
        "--lora_top_layers",
        type=int,
        default=None,
        help="Reserved for future use (top encoder layers to apply LoRA).",
    )
    args = parser.parse_args()

    model_names = _expand_models(args.model_name)
    config = FIXED_CONFIG

    for resolved_model in model_names:
        print(
            f"\n[Run] Training {resolved_model} with LoRA "
            f"(head_type={args.head_type}, truncation={args.truncation}, "
            f"loss_fn={args.loss_fn}, dropout={args.dropout}), mode={args.param_mode}"
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
            args.truncation,
            args.loss_fn,
            args.dropout,
        )
        STAGE3_DIR = Path(__file__).resolve().parent
        output_dir = STAGE3_DIR / "tmp_checkpoints" / slug
        final_model_dir = STAGE3_DIR / "stage3_trained_models" / slug

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
            truncation=args.truncation,
            dropout=float(args.dropout),
            loss_type=args.loss_fn,
        )

        eval_metrics = trainer.evaluate()
        best_metrics = getattr(trainer, "best_eval_metrics", None) or eval_metrics
        acc = float(best_metrics.get("eval_accuracy", 0.0))
        f1_macro = float(best_metrics.get("eval_f1_macro", 0.0))
        f1_micro = float(best_metrics.get("eval_f1_micro", 0.0))
        f1_weighted = float(best_metrics.get("eval_f1_weighted", 0.0))

        print(f"\n{resolved_model} on DEV RESULTS:")
        print(f"Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
        print(f"F1-Macro: {f1_macro:.4f}")
        print(f"F1-Micro: {f1_micro:.4f}")
        print(f"F1-Weighted: {f1_weighted:.4f}")
        print(f"[Info] Training run complete for {resolved_model}.")


if __name__ == "__main__":
    main()
