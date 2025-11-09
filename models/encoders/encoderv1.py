import inspect
import math
import os
import random
import re
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
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

# NEW: for plotting at the end
import matplotlib.pyplot as plt


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
MODEL_NAME_SAFE = _fs_safe_model_name(MODEL_NAME)

USE_EARLY_STOPPING = False   # set True to enable again
EARLY_STOPPING_PATIENCE = 20

TRAIN_CSV_PATH = os.path.join("datasets", "train_dataset.csv")
TEST_CSV_PATH = os.path.join("datasets", "test_dataset.csv")

OUTPUT_DIR = os.path.join("results", "models", MODEL_NAME_SAFE)
FINAL_MODEL_DIR = os.path.join("finetuned-models", MODEL_NAME_SAFE)
PREDICTIONS_DIR = os.path.join("results", "predictions")
METRICS_DIR = os.path.join("results", "metrics")
F1_METRICS_CSV = os.path.join(METRICS_DIR, "f1_scores.csv")

TARGET_COLUMN = "clarity_label"
ARG1_KEY = "question"
ARG2_KEY = "interview_answer"

# NOTE: we keep these hyperparams the same
NUM_EPOCHS = 20
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
# 2) Helper: metrics (PDF-style)
# -----------------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_micro = f1_score(labels, preds, average="micro")
    return {"accuracy": acc, "f1_macro": f1_macro, "f1_micro": f1_micro}  # :contentReference[oaicite:3]{index=3}


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


# NEW: helper to plot train/eval loss history at the end
def _plot_train_eval_loss(trainer) -> None:
    """Plot per-epoch training loss vs eval loss from Trainer.log_history."""
    logs = trainer.state.log_history if trainer is not None else []

    # Collect (epoch, loss) pairs
    train_points = [(e.get("epoch"), e.get("loss")) for e in logs if "epoch" in e and "loss" in e]
    eval_points  = [(e.get("epoch"), e.get("eval_loss")) for e in logs if "epoch" in e and "eval_loss" in e]

    # Deduplicate by epoch (keep last value if multiple logs per epoch)
    def _by_epoch(points):
        acc = {}
        for ep, val in points:
            if ep is None or val is None:
                continue
            acc[_epoch_to_int(ep)] = float(val)
        xs = sorted(acc.keys())
        ys = [acc[x] for x in xs]
        return xs, ys

    train_epochs, train_losses = _by_epoch(train_points)
    eval_epochs,  eval_losses  = _by_epoch(eval_points)

    if not train_epochs and not eval_epochs:
        print("[Warn] No loss logs found to plot.")
        return

    plt.figure()
    if train_epochs:
        plt.plot(train_epochs, train_losses, marker="o", label="Train loss")
    if eval_epochs:
        plt.plot(eval_epochs, eval_losses, marker="o", label="Eval loss (on test set)")

    combined_epochs = sorted(set(train_epochs + eval_epochs))
    if combined_epochs:
        plt.xticks(range(combined_epochs[0], combined_epochs[-1] + 1))

    best_epoch = best_loss = None
    if eval_epochs:
        best_idx = int(np.argmin(eval_losses))
        best_epoch = eval_epochs[best_idx]
        best_loss = eval_losses[best_idx]
        plt.scatter([best_epoch], [best_loss], marker="*", s=120, color="red", label="Best eval loss")
        loss_scale = max(best_loss, max(eval_losses) if eval_losses else 1.0)
        y_offset = max(0.02, 0.08 * loss_scale)
        plt.annotate(
            f"Best eval (epoch {best_epoch})",
            xy=(best_epoch, best_loss),
            xytext=(best_epoch, best_loss + y_offset),
            ha="center",
            va="bottom",
            arrowprops={"arrowstyle": "->", "color": "red", "lw": 0.8},
            fontsize=9,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Evaluation Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    if best_epoch is not None and best_loss is not None:
        print(f"[Info] Best eval loss observed at epoch {best_epoch}: {best_loss:.4f}")


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
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=0.01,
        seed=SEED,
        data_seed=SEED,
        fp16=torch.cuda.is_available(),
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

    return trainer, id2label, label2id, tokenizer


# -----------------------------------------------------------------------------
# 4) Pipeline: load CSVs, optional downsample, train on ALL train, validate on TEST, then reports
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

    trainer, id2label, label2id, tokenizer = train_model(
        model_name=MODEL_NAME,
        train_df=train_full_df,
        dev_df=test_full_df,    # test acts as 'validation'
        arg1_key=ARG1_KEY,
        arg2_key=ARG2_KEY,
        label_col=TARGET_COLUMN,
    )

    # =========================================================================
    # REBUILD TOKENIZED DATASETS so we can:
    #  - evaluate explicitly on TRAIN set (print train values),
    #  - evaluate explicitly on TEST set,
    #  - build confusion matrices and reports
    # =========================================================================

    # TRAIN SET
    train_proc = train_full_df[[ARG1_KEY, ARG2_KEY]].copy()
    train_proc["label"] = train_full_df[TARGET_COLUMN].map(label2id)

    train_ds = Dataset.from_pandas(train_proc, preserve_index=False).map(
        lambda batch: tokenizer(
            batch[ARG1_KEY],
            batch[ARG2_KEY],
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        ),
        batched=True,
    )

    # TEST SET
    test_proc = test_full_df[[ARG1_KEY, ARG2_KEY]].copy()
    test_proc["label"] = test_full_df[TARGET_COLUMN].map(label2id)

    test_ds = Dataset.from_pandas(test_proc, preserve_index=False).map(
        lambda batch: tokenizer(
            batch[ARG1_KEY],
            batch[ARG2_KEY],
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        ),
        batched=True,
    )

    # ================================
    # METRICS ON TRAIN SET
    # ================================
    print("[Info] Evaluating explicitly on TRAIN set…")
    train_metrics = trainer.evaluate(train_ds, metric_key_prefix="train")
    for k, v in train_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    train_pred = trainer.predict(train_ds)
    train_probs = torch.softmax(torch.tensor(train_pred.predictions), dim=-1).numpy()
    train_pred_ids = train_probs.argmax(axis=-1)
    train_true_ids = train_proc["label"].to_numpy()

    target_names = [id2label[i] for i in sorted(id2label)]
    train_report = classification_report(
        train_true_ids,
        train_pred_ids,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "train_report.txt"), "w", encoding="utf-8") as f:
        f.write(train_report)

    print("\n[Info] TRAIN classification report:")
    print(train_report)

    train_cm = confusion_matrix(
        train_true_ids,
        train_pred_ids,
        labels=list(range(len(id2label))),
    )
    train_cm_df = pd.DataFrame(train_cm, index=target_names, columns=target_names)
    train_cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_train.csv")
    train_cm_df.to_csv(train_cm_path)
    print(f"[Info] Saved TRAIN confusion matrix to {train_cm_path}")

    # ================================
    # METRICS ON TEST SET
    # ================================
    print("[Info] Evaluating explicitly on TEST set…")
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    test_pred = trainer.predict(test_ds)
    test_probs = torch.softmax(torch.tensor(test_pred.predictions), dim=-1).numpy()
    test_pred_ids = test_probs.argmax(axis=-1)
    test_true_ids = test_proc["label"].to_numpy()

    test_report = classification_report(
        test_true_ids,
        test_pred_ids,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )

    with open(os.path.join(OUTPUT_DIR, "test_report.txt"), "w", encoding="utf-8") as f:
        f.write(test_report)

    print("\n[Info] TEST classification report:")
    print(test_report)

    # CONFUSION MATRIX (TEST)
    test_cm = confusion_matrix(
        test_true_ids,
        test_pred_ids,
        labels=list(range(len(id2label))),
    )
    test_cm_df = pd.DataFrame(test_cm, index=target_names, columns=target_names)
    test_cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_test.csv")
    test_cm_df.to_csv(test_cm_path)
    print(f"[Info] Saved TEST confusion matrix to {test_cm_path}")

    # ================================
    # SAVE TEST PREDS CSV (required format)
    # ================================
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    pred_labels = [id2label[int(i)] for i in test_pred_ids]
    out_df = pd.DataFrame(
        {
            "question": test_full_df[ARG1_KEY].values,
            "answer": test_full_df[ARG2_KEY].values,
            "model_prediction": pred_labels,  # required column name
            "true_label": test_full_df[TARGET_COLUMN].astype(str).values,
        }
    )
    for i, lab in sorted(id2label.items()):
        out_df[f"prob_{lab}"] = test_probs[:, i]

    test_csv_path = os.path.join(PREDICTIONS_DIR, f"{MODEL_NAME_SAFE}_test_predictions.csv")
    out_df.to_csv(test_csv_path, index=False)
    print(f"[Info] Saved test predictions to {test_csv_path}")

    # ================================
    # LOG F1 MACRO/WEIGHTED (TEST SET)
    # ================================
    f1_macro_value = f1_score(test_true_ids, test_pred_ids, average="macro")
    f1_weighted_value = f1_score(test_true_ids, test_pred_ids, average="weighted")

    os.makedirs(METRICS_DIR, exist_ok=True)
    if os.path.exists(F1_METRICS_CSV):
        metrics_table = pd.read_csv(F1_METRICS_CSV, index_col="metric")
    else:
        metrics_table = pd.DataFrame()

    metrics_table.loc["f1_macro", MODEL_NAME_SAFE] = f1_macro_value
    metrics_table.loc["f1_weighted", MODEL_NAME_SAFE] = f1_weighted_value
    metrics_table.to_csv(F1_METRICS_CSV, index_label="metric")
    print(f"[Info] Logged F1 scores to {F1_METRICS_CSV}")

    # ================================
    # PLOT TRAIN vs TEST LOSS OVER EPOCHS
    # (train = whole train_dataset.csv after downsampling, eval = test_dataset.csv)
    # ================================
    _plot_train_eval_loss(trainer)


if __name__ == "__main__":
    main()
