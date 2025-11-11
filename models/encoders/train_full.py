import inspect
import math
import os
import random
import re
import shutil
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from datasets import Dataset
try:
    from datasets import set_seed as set_datasets_seed
except ImportError:
    set_datasets_seed = None
from sklearn.metrics import accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from model_metadata import MODEL_METADATA

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
MODEL_NAME = "FacebookAI/xlm-roberta-base"

USE_EARLY_STOPPING = False   # set True to enable again
EARLY_STOPPING_PATIENCE = 20

TRAIN_CSV_PATH = os.path.join("datasets", "train_dataset.csv")
TEST_CSV_PATH = os.path.join("datasets", "test_dataset.csv")

TASK_ID = "t1"          # TODO: expose via CLI/config when multiple tasks are trained.
TUNE_STRATEGY = "full"  # TODO: make dynamic (full, freeze25, etc.).
PARAM_MODE = "fixed"    # TODO: switch to hps when hyperparameter sweeps are used.

def _build_output_slug(model_name: str) -> str:
    if model_name not in MODEL_METADATA:
        raise KeyError(
            f"Model '{model_name}' missing from MODEL_METADATA. "
            "Add it to models/encoders/model_metadata.py."
        )
    meta = MODEL_METADATA[model_name]
    slug = f"{TASK_ID}_{meta['arch']}_{meta['lang']}_{meta['size']}_{TUNE_STRATEGY}_{PARAM_MODE}"
    return _fs_safe_model_name(slug)


MODEL_OUTPUT_SLUG = _build_output_slug(MODEL_NAME)

OUTPUT_DIR = os.path.join("results", "models", MODEL_OUTPUT_SLUG)
FINAL_MODEL_DIR = os.path.join("models", "encoders", "trained_models", MODEL_OUTPUT_SLUG)
TARGET_COLUMN = "clarity_label"
ARG1_KEY = "question"
ARG2_KEY = "interview_answer"

# NOTE: we keep these hyperparams the same
NUM_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
MAX_LENGTH = 512

# NOTE: We now train on the *entire* train_dataset.csv and evaluate on test_dataset.csv.

# NEW: downsample config — drop some rows of the "Ambivalent" class from TRAIN before training
AMBIVALENT_LABEL_NAME = "Ambivalent"
AMBIVALENT_DROP_N = 1000  # drop up to this many from the training set


def set_global_seed(seed: int = SEED) -> None:
    """Match the PDF’s seed setup style for reproducibility."""
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

set_global_seed(SEED)  # :contentReference[oaicite:2]{index=2}


# -----------------------------------------------------------------------------
# 2) Helper: metrics (PDF-style, accuracy only)
# -----------------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}  # :contentReference[oaicite:3]{index=3}


class WeightedTrainer(Trainer):
    """Trainer that applies class-weighted cross-entropy to counter class imbalance."""

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.float()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def _epoch_to_int(epoch_value) -> int:
    """Convert a possibly fractional epoch count into a 1-based integer epoch index."""
    epoch_float = float(epoch_value)
    epoch_idx = int(math.floor(epoch_float + 0.5))
    return max(1, epoch_idx)


def _best_eval_epoch(log_history, target_metric) -> int | None:
    """Find the epoch associated with the target eval metric in Trainer.log_history."""
    if target_metric is None:
        return None
    for record in log_history or []:
        if "eval_loss" not in record:
            continue
        value = record.get("eval_loss")
        if value is None:
            continue
        if abs(float(value) - float(target_metric)) <= 1e-9:
            epoch_val = record.get("epoch")
            if epoch_val is not None:
                return _epoch_to_int(epoch_val)
    return None


def _remove_checkpoint_dirs(path: str) -> None:
    """Delete Hugging Face checkpoint folders to keep only the final exported model."""
    if not os.path.isdir(path):
        return
    removed_any = False
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path) and entry.startswith("checkpoint-"):
            shutil.rmtree(full_path, ignore_errors=True)
            removed_any = True
    if removed_any:
        print(f"[Info] Removed checkpoint directories under {path}")


# -----------------------------------------------------------------------------
# 3) Core training function, styled like the PDF’s `train(...)` (no freezing)
# -----------------------------------------------------------------------------
def train_model(
    model_name: str,
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    arg1_key: str,
    arg2_key: str,
    label_col: str,
):
    """
    Finetunes a transformer for sequence classification using (Arg1, Arg2) pair tokenization.

    IMPORTANT CHANGE:
    - train_df is now the *entire* training dataset (train_dataset.csv), after optional downsampling.
    - dev_df is now the held-out test set (test_dataset.csv).
      We pass dev_df as the eval_dataset to the Trainer, so "eval_loss"
      and early stopping are based on the test set.

    Returns (trainer, id2label, label2id, tokenizer).
    """

    # Encode labels (aligns train/dev/test to same mapping)
    unique_labels = sorted(train_df[label_col].dropna().unique())
    label2id: Dict[str, int] = {lab: i for i, lab in enumerate(unique_labels)}
    id2label: Dict[int, str] = {i: lab for lab, i in label2id.items()}

    # Basic cleaning, map labels -> ids
    train_df = train_df.dropna(subset=[arg1_key, arg2_key, label_col]).copy()
    dev_df = dev_df.dropna(subset=[arg1_key, arg2_key, label_col]).copy()
    train_df["label"] = train_df[label_col].map(label2id)
    dev_df["label"] = dev_df[label_col].map(label2id)

    # Class weights from TRAIN set only
    label_counts = train_df["label"].value_counts().sort_index()
    class_weights = (len(train_df) / (len(label_counts) * label_counts)).sort_index()
    class_weights_tensor = torch.tensor(class_weights.to_numpy(), dtype=torch.float)

    # Tokenizer & tokenization (pair inputs, like PDF)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

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

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label2id,
    )

    # TrainingArguments
    # NOTE: eval_dataset=dev_ds (the test set) so eval_loss is test loss.
    training_kwargs = dict(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=0.01,
        seed=SEED,
        data_seed=SEED,
        fp16=torch.cuda.is_available(),
        use_safetensors=True,
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
        eval_dataset=dev_ds,          # <-- test set drives eval_loss
        processing_class=tokenizer,   # mirrors PDF usage
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    print("[Info] Starting training on FULL TRAIN SET…")
    trainer.train()

    # Evaluate on test set (because that's what we gave as eval_dataset)
    print("[Info] Evaluating (using test set as eval_dataset)…")
    eval_metrics = trainer.evaluate(eval_dataset=dev_ds)
    for k, v in eval_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save final model + tokenizer
    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    print(f"[Info] Saved fine-tuned model & tokenizer to {FINAL_MODEL_DIR}")
    _remove_checkpoint_dirs(OUTPUT_DIR)

    return trainer, id2label, label2id, tokenizer


# -----------------------------------------------------------------------------
# 4) Pipeline: load CSVs, optional downsample, train on ALL train, validate on TEST
# -----------------------------------------------------------------------------
def main():
    # Load the ENTIRE training CSV
    train_full_df = pd.read_csv(TRAIN_CSV_PATH)
    train_full_df = train_full_df.dropna(subset=[ARG1_KEY, ARG2_KEY, TARGET_COLUMN]).copy()

    # === NEW: Downsample "Ambivalent" rows from TRAIN before training ===
    mask_amb = train_full_df[TARGET_COLUMN].astype(str).str.casefold() == AMBIVALENT_LABEL_NAME.casefold()
    amb_count = int(mask_amb.sum())
    drop_n = min(AMBIVALENT_DROP_N, amb_count)
    if drop_n > 0:
        to_remove = train_full_df[mask_amb].sample(n=drop_n, random_state=SEED)
        train_full_df = train_full_df.drop(index=to_remove.index).reset_index(drop=True)
        print(f"[Info] Dropped {drop_n} '{AMBIVALENT_LABEL_NAME}' rows from TRAIN "
              f"({amb_count} available). New train size: {len(train_full_df)}")
    else:
        print(f"[Info] No '{AMBIVALENT_LABEL_NAME}' rows to drop (found {amb_count}).")

    # Load the TEST CSV (this becomes our eval/validation set)
    if not os.path.exists(TEST_CSV_PATH):
        raise FileNotFoundError(
            f"[Fatal] Test CSV not found at {TEST_CSV_PATH}. "
            "We now require it for eval_loss / early stopping."
        )
    test_full_df = pd.read_csv(TEST_CSV_PATH)
    test_full_df = test_full_df.dropna(subset=[ARG1_KEY, ARG2_KEY, TARGET_COLUMN]).copy()

    train_model(
        model_name=MODEL_NAME,
        train_df=train_full_df,
        dev_df=test_full_df,    # test acts as 'validation'
        arg1_key=ARG1_KEY,
        arg2_key=ARG2_KEY,
        label_col=TARGET_COLUMN,
    )
    print("[Info] Training run complete. Skipping post-training prediction/report generation.")

if __name__ == "__main__":
    main()
