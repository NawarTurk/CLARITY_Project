import argparse
import inspect
import math
import time
import os
import random
import re
import shutil
from pathlib import Path
from typing import Dict, Tuple, Optional

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
TUNE_STRATEGY = "partial"  # partial unfreezing of encoder layers; embeddings frozen
PLOTS_DIR = Path("results") / "plots" / "encoder"

FIXED_CONFIG = {
    "num_train_epochs": 20,
    "batch_size": 16,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
}


def _build_output_slug(model_name: str, param_mode: str, unfreeze_ratio: Optional[float] = None) -> str:
    if model_name not in MODEL_METADATA:
        raise KeyError(
            f"Model '{model_name}' missing from MODEL_METADATA. "
            "Add it to models/encoders/model_metadata.py."
        )
    meta = MODEL_METADATA[model_name]
    head_type = "defaultHead"  # current default classification head
    pct = int(round(100 * float(unfreeze_ratio))) if unfreeze_ratio is not None else None
    strategy_tag = f"unfreezing{pct}" if pct is not None else TUNE_STRATEGY

    slug = (
        f"{TASK_ID}_{meta['arch']}_{meta['lang']}_{meta['size']}"
        f"_{strategy_tag}_{param_mode}_{head_type}"
    )

    return _fs_safe_model_name(slug)


TARGET_COLUMN = "clarity_label"
ARG1_KEY = "question"
ARG2_KEY = "interview_answer"


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

    # Fall back to user-provided name
    return name


def _expand_models(user_name: str):
    normalized = user_name.strip().lower()
    if normalized == "all":
        resolved = [_resolve_model_name(m) for m in ALLOWED_MODEL_NAMES]
        seen = set()
        # Add new models to ALLOWED_MODEL_NAMES to include them when using --model_name all
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
        # will be filled from freezing logic; used when saving progress
        self.freeze_summary_lines: list[str] = []

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

        # ETA logic
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
                if hours > 0:
                    eta_str = f"{hours:d}h{minutes:02d}m"
                else:
                    eta_str = f"{minutes:02d}m{seconds:02d}s"

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
# Freezing / partial unfreezing logic + summary (console + txt file)
# -----------------------------------------------------------------------------
from collections import defaultdict


def _get_base_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Try to get the underlying backbone (bert/roberta/deberta/xlm_roberta/distilbert).
    Mirrors the mentor's mBERT script behaviour.
    """
    base = getattr(model, "base_model", None)
    if base is not None:
        return base

    for attr in ("bert", "roberta", "deberta", "xlm_roberta", "distilbert"):
        if hasattr(model, attr):
            return getattr(model, attr)

    return model  # last resort


def apply_partial_unfreezing(model: torch.nn.Module, unfreeze_ratio: float) -> list[str]:
    lines: list[str] = []

    #Clamp ratio to [0, 1]
    try:
        unfreeze_ratio = float(unfreeze_ratio)
    except (TypeError, ValueError):
        unfreeze_ratio = 1.0
    unfreeze_ratio = max(0.0, min(1.0, unfreeze_ratio))

    base = _get_base_model(model)

    #1) Freeze embeddings
    if hasattr(base, "embeddings"):
        for param in base.embeddings.parameters():
            param.requires_grad = False
        msg = "[Freeze] Embedding layer parameters frozen."
        print(msg)
        lines.append(msg)
    else:
        msg = "⚠️ Could not apply embedding freezing. No 'embeddings' on base model."
        print(msg)
        lines.append(msg)

    #2) Freeze encoder layers based on unfreeze_ratio
    if hasattr(base, "encoder") and hasattr(base.encoder, "layer"):
        encoder_layers = list(base.encoder.layer)
        total_layers = len(encoder_layers)

        if total_layers == 0:
            msg = "⚠️ Encoder has no layers; skipping encoder freezing."
            print(msg)
            lines.append(msg)
        else:
            unfreeze_start = int((1.0 - unfreeze_ratio) * total_layers)
            unfreeze_start = max(0, min(unfreeze_start, total_layers))

            for i, layer in enumerate(encoder_layers):
                requires_grad = (i >= unfreeze_start) and (unfreeze_ratio > 0.0)
                for param in layer.parameters():
                    param.requires_grad = requires_grad

            pct_unfrozen = int(round(100.0 * unfreeze_ratio))
            msg = f"*_Unfreezing ratio: {pct_unfrozen}% of encoder layers"
            print(msg); lines.append(msg)

            msg = f"*_Total encoder layers: {total_layers}"
            print(msg); lines.append(msg)

            if unfreeze_ratio <= 0.0:
                msg = "*_All layers frozen"
                print(msg); lines.append(msg)
            elif unfreeze_start == 0:
                msg = f"*_Unfreezing from layer 0 to {total_layers - 1}"
                print(msg); lines.append(msg)
            else:
                msg = f"*_Unfreezing from layer {unfreeze_start} to {total_layers - 1}"
                print(msg); lines.append(msg)
    else:
        msg = "⚠️ Could not apply encoder layer freezing. Unexpected model structure."
        print(msg)
        lines.append(msg)

    #3) Trainable parameter summary
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = 100.0 * trainable / total if total > 0 else 0.0

    msg = f"*_Total parameters:     {total:,}"
    print(msg); lines.append(msg)
    msg = f"*__Trainable parameters: {trainable:,}"
    print(msg); lines.append(msg)
    msg = f"*__Trainable ratio:      {ratio:.2f}%"
    print(msg); lines.append(msg)

    grouped: dict[str, list[str]] = defaultdict(list)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if (
            "roberta.encoder.layer" in name
            or "bert.encoder.layer" in name
            or "deberta.encoder.layer" in name
        ):
            top_group = ".".join(name.split(".")[:4])  # up to the layer index
        else:
            top_group = ".".join(name.split(".")[:2])
        grouped[top_group].append(name)

    for group, names in sorted(grouped.items()):
        msg = f"✅ {group}: ({len(names)} params)"
        print(msg)
        lines.append(msg)

    return lines


def _save_training_progress(trainer: Trainer, final_model_dir: str, eval_metrics: Dict[str, float] | None = None) -> None:
    if not isinstance(trainer, WeightedTrainer):
        return
    header = trainer._table_header
    lines = trainer._table_lines
    if not lines:
        return

    #If no eval_metrics were passed, try to pull the final eval record
    #from the trainer's log history.
    if eval_metrics is None:
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
        #epoch-by-epoch table
        if header is not None:
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
        for line in lines:
            f.write(line + "\n")

        #final eval metrics block
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

            epoch_val = getattr(trainer.state, "epoch", None)
            step_val = getattr(trainer.state, "global_step", None)
            if epoch_val is not None:
                metrics_block["epoch"] = epoch_val
            if step_val is not None:
                metrics_block["step"] = step_val

            if metrics_block:
                f.write("\n[Final evaluation metrics]\n")
                for key, value in metrics_block.items():
                    if isinstance(value, float):
                        f.write(f"  {key}: {value:.4f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")

        #layer-freezing / trainable-parameter summary
        freeze_summary = getattr(trainer, "freeze_summary_lines", None)
        if freeze_summary:
            f.write("\n[Layer freezing / trainable parameter summary]\n")
            for line in freeze_summary:
                f.write(line + "\n")


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
    plt.tight_layout()
    loss_path = PLOTS_DIR / f"{run_name_safe}_loss.png"
    plt.savefig(loss_path)
    plt.close()
    print(f"[Info] Saved loss curves to {loss_path}")


#Core training function
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
):

    #Encode labels
    unique_labels = sorted(train_df[label_col].dropna().unique())
    label2id: Dict[str, int] = {lab: i for i, lab in enumerate(unique_labels)}
    id2label: Dict[int, str] = {i: lab for lab, i in label2id.items()}

    #Cleaning
    train_df = train_df.dropna(subset=[arg1_key, arg2_key, label_col]).copy()
    dev_df = dev_df.dropna(subset=[arg1_key, arg2_key, label_col]).copy()
    train_df["label"] = train_df[label_col].map(label2id)
    dev_df["label"] = dev_df[label_col].map(label2id)

    #Class weights from TRAIN set only
    label_counts = train_df["label"].value_counts().sort_index()
    class_weights = (len(train_df) / (len(label_counts) * label_counts)).sort_index()
    class_weights_tensor = torch.tensor(class_weights.to_numpy(), dtype=torch.float)

    #Tokenizer & tokenization
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

    #Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label2id,
    )

    #Apply mentor-style partial unfreezing and capture the summary lines
    freeze_summary_lines = apply_partial_unfreezing(model, unfreeze_ratio)

    #TrainingArguments
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
        save_total_limit=1,
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
    #Map evaluation_strategy → eval_strategy if using a newer transformers API
    if "evaluation_strategy" not in ta_params and "eval_strategy" in ta_params:
        training_kwargs["eval_strategy"] = training_kwargs.pop("evaluation_strategy")
    #Drop any kwargs not supported by this install
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
        eval_dataset=dev_ds,          #test acts as eval set
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    #attach freezing summary to trainer so we can dump it into the txt later
    trainer.freeze_summary_lines = freeze_summary_lines

    print("[Info] Starting training with partial unfreezing…")
    trainer.train()

    #Save training progress table as .txt (includes layer freezing summary)
    _save_training_progress(trainer, final_model_dir)

    #Use the slug from output_dir as a run name for plots
    run_name = os.path.basename(os.path.normpath(output_dir)) or model_name

    #Plot training vs eval loss curves
    _plot_loss_curves(trainer, run_name=run_name)

    #Evaluate on dev set (because that's what we gave as eval_dataset)
    print("[Info] Evaluating (using dev/test set as eval_dataset)…")
    eval_metrics = trainer.evaluate(eval_dataset=dev_ds)
    for k, v in eval_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    #Resolve the best checkpoint path chosen during training (if any).
    best_ckpt = getattr(trainer.state, "best_model_checkpoint", None)

    #Save best model + tokenizer (falls back to last state if no best checkpoint tracked)
    os.makedirs(final_model_dir, exist_ok=True)
    if best_ckpt and os.path.isdir(best_ckpt):
        print(f"[Info] Loading best model from checkpoint: {best_ckpt}")
        best_model = AutoModelForSequenceClassification.from_pretrained(
            best_ckpt,
            num_labels=len(unique_labels),
            id2label=id2label,
            label2id=label2id,
        )
        best_model.save_pretrained(final_model_dir)
    else:
        print("[Warn] No best checkpoint found; saving current model state instead.")
        trainer.save_model(final_model_dir)

    tokenizer.save_pretrained(final_model_dir)
    print(f"[Info] Saved fine-tuned model & tokenizer to {final_model_dir}")
    _remove_checkpoint_dirs(output_dir)

    return trainer, id2label, label2id, tokenizer


#4) Pipeline: load CSVs, train with partial unfreezing
def main():
    parser = argparse.ArgumentParser(description="Encoder fine-tuning with partial unfreezing.")
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
    args = parser.parse_args()

    model_names = _expand_models(args.model_name)
    config = FIXED_CONFIG

    #Fixed set of encoder unfreezing ratios to sweep over
    unfreeze_ratios = (0.25, 0.50, 0.75)

    for resolved_model in model_names:
        for unfreeze_ratio in unfreeze_ratios:
            print(
                f"\n[Run] Training {resolved_model} with partial unfreezing "
                f"(unfreeze_ratio={unfreeze_ratio:.2f}), mode={args.param_mode}"
            )

            #Load the ENTIRE training CSV
            train_full_df = pd.read_csv(TRAIN_CSV_PATH)
            train_full_df = train_full_df.dropna(subset=[ARG1_KEY, ARG2_KEY, TARGET_COLUMN]).copy()

            #Load the TEST CSV (this becomes our eval/validation set)
            if not os.path.exists(TEST_CSV_PATH):
                raise FileNotFoundError(
                    f"[Fatal] Test CSV not found at {TEST_CSV_PATH}. "
                    "We now require it for eval_loss / early stopping."
                )
            test_full_df = pd.read_csv(TEST_CSV_PATH)
            test_full_df = test_full_df.dropna(subset=[ARG1_KEY, ARG2_KEY, TARGET_COLUMN]).copy()

            slug = _build_output_slug(resolved_model, args.param_mode, unfreeze_ratio)
            output_dir = os.path.join("results", "models", slug)
            final_model_dir = os.path.join("models", "encoders", "trained_models", slug)

            trainer, _, _, _ = train_model(
                model_name=resolved_model,
                train_df=train_full_df,
                dev_df=test_full_df,    # test acts as 'validation'
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
            )

            eval_metrics = trainer.evaluate()
            acc = float(eval_metrics.get("eval_accuracy", 0.0))
            f1_macro = float(eval_metrics.get("eval_f1_macro", 0.0))
            f1_micro = float(eval_metrics.get("eval_f1_micro", 0.0))
            f1_weighted = float(eval_metrics.get("eval_f1_weighted", 0.0))
            print(f"[Info] Eval accuracy (partial): {acc}")

            # Final best-model summary for this run
            print(f"\n{resolved_model} on DEV RESULTS (unfreeze_ratio={unfreeze_ratio:.2f}):")
            print(f"Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
            print(f"F1-Macro: {f1_macro:.4f}")
            print(f"F1-Micro: {f1_micro:.4f}")
            print(f"F1-Weighted: {f1_weighted:.4f}")
            print("[Info] Training run complete for this unfreeze_ratio. Skipping post-training prediction/report generation.")


if __name__ == "__main__":
    main()
