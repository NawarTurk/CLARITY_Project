import inspect

import os
import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
try:
    from datasets import set_seed as set_datasets_seed
except ImportError:
    set_datasets_seed = None
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

# -----------------------------------------------------------------------------
# 1) Setup
# -----------------------------------------------------------------------------
SEED = 42
MODEL_NAME = "distilbert-base-uncased"

TRAIN_CSV_PATH = os.path.join("datasets", "train_dataset.csv")
TEST_CSV_PATH = os.path.join("datasets", "test_dataset.csv")

OUTPUT_DIR = os.path.join("results", "models", "distilbert-base-uncased")
FINAL_MODEL_DIR = os.path.join("finetuned-models", "distilbert-base-uncased")
PREDICTIONS_DIR = os.path.join("results", "predictions")

TARGET_COLUMN = "clarity_label"
ARG1_KEY = "question"
ARG2_KEY = "interview_answer"

NUM_EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
MAX_LENGTH = 256
VAL_SIZE = 0.10


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
    Finetunes a transformer for sequence classification using (Arg1, Arg2) pair tokenization,
    evaluates each epoch with early stopping, and returns (trainer, id2label, label2id, tokenizer).

    This mirrors the PDF’s structure (pair tokenization, Trainer, metrics) while skipping any
    layer freezing logic. :contentReference[oaicite:4]{index=4}
    """
    # Encode labels (aligns train/dev to same mapping)
    unique_labels = sorted(train_df[label_col].dropna().unique())
    label2id: Dict[str, int] = {lab: i for i, lab in enumerate(unique_labels)}
    id2label: Dict[int, str] = {i: lab for lab, i in label2id.items()}

    train_df = train_df.dropna(subset=[arg1_key, arg2_key, label_col]).copy()
    dev_df = dev_df.dropna(subset=[arg1_key, arg2_key, label_col]).copy()
    train_df["label"] = train_df[label_col].map(label2id)
    dev_df["label"] = dev_df[label_col].map(label2id)

    # Tokenizer & tokenization (pair inputs, like PDF) :contentReference[oaicite:5]{index=5}
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

    # TrainingArguments (epoch strategies, early stopping compatible)
    training_kwargs = dict(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=0.01,
        seed=SEED,
        data_seed=SEED,
        fp16=torch.cuda.is_available(),
        save_total_limit=3,
    )

    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    # Map evaluation_strategy → eval_strategy if the new name is required
    if "evaluation_strategy" not in ta_params and "eval_strategy" in ta_params:
        training_kwargs["eval_strategy"] = training_kwargs.pop("evaluation_strategy")
    # Drop any kwargs that are unsupported by this transformers build
    filtered_kwargs = {k: v for k, v in training_kwargs.items() if k in ta_params}

    training_args = TrainingArguments(**filtered_kwargs)

    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,  # mirrors PDF usage
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    print("[Info] Starting training…")
    trainer.train()

    print("[Info] Evaluating on validation split…")
    eval_metrics = trainer.evaluate(eval_dataset=dev_ds)
    for k, v in eval_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Detailed val report
    val_pred = trainer.predict(dev_ds)
    val_y_true = dev_df["label"].to_numpy()
    val_y_pred = np.argmax(val_pred.predictions, axis=-1)
    target_names = [id2label[i] for i in sorted(id2label)]
    val_report = classification_report(val_y_true, val_y_pred, target_names=target_names, digits=4, zero_division=0)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "validation_report.txt"), "w", encoding="utf-8") as f:
        f.write(val_report)
    print("\n[Info] Validation classification report:")
    print(val_report)

    # Save final model
    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    print(f"[Info] Saved fine-tuned model & tokenizer to {FINAL_MODEL_DIR}")

    return trainer, id2label, label2id, tokenizer


# -----------------------------------------------------------------------------
# 4) Pipeline: load CSVs, split, train, then run TEST predictions + reports
# -----------------------------------------------------------------------------
def main():
    # Load full training CSV and split into train/dev (stratified)
    full_df = pd.read_csv(TRAIN_CSV_PATH)
    full_df = full_df.dropna(subset=[ARG1_KEY, ARG2_KEY, TARGET_COLUMN]).copy()

    train_df, dev_df = train_test_split(
        full_df,
        test_size=VAL_SIZE,
        stratify=full_df[TARGET_COLUMN],
        random_state=SEED,
    )
    train_df = train_df.reset_index(drop=True)
    dev_df = dev_df.reset_index(drop=True)

    trainer, id2label, label2id, tokenizer = train_model(
        model_name=MODEL_NAME,
        train_df=train_df,
        dev_df=dev_df,
        arg1_key=ARG1_KEY,
        arg2_key=ARG2_KEY,
        label_col=TARGET_COLUMN,
    )

    # ================================
    # DEV/VALIDATION CONFUSION MATRIX
    # ================================
    # Rebuild a dev dataset (pair-tokenized) and compute confusion matrix
    dev_proc = dev_df[[ARG1_KEY, ARG2_KEY]].copy()
    dev_proc["label"] = dev_df[TARGET_COLUMN].map(label2id)

    dev_ds = Dataset.from_pandas(dev_proc, preserve_index=False).map(
        lambda batch: tokenizer(
            batch[ARG1_KEY],
            batch[ARG2_KEY],
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        ),
        batched=True,
    )

    dev_pred = trainer.predict(dev_ds)
    dev_probs = torch.softmax(torch.tensor(dev_pred.predictions), dim=-1).numpy()
    dev_pred_ids = dev_probs.argmax(axis=-1)

    target_names = [id2label[i] for i in sorted(id2label)]
    dev_true_ids = dev_proc["label"].to_numpy()

    dev_cm = confusion_matrix(dev_true_ids, dev_pred_ids, labels=list(range(len(id2label))))
    dev_cm_df = pd.DataFrame(dev_cm, index=target_names, columns=target_names)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dev_cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_val.csv")
    dev_cm_df.to_csv(dev_cm_path)
    print(f"[Info] Saved validation confusion matrix to {dev_cm_path}")

    # --- Test predictions (probabilities + CSV) ---
    if not os.path.exists(TEST_CSV_PATH):
        print(f"[Warn] Test CSV not found at {TEST_CSV_PATH}. Skipping test predictions.")
        return

    test_df = pd.read_csv(TEST_CSV_PATH).dropna(subset=[ARG1_KEY, ARG2_KEY, TARGET_COLUMN]).copy()
    test_proc = test_df[[ARG1_KEY, ARG2_KEY]].copy()
    test_proc["label"] = test_df[TARGET_COLUMN].map(label2id)

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

    # Evaluate (for metrics) and Predict (for raw logits → probs)
    print("[Info] Evaluating on test set…")
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("[Info] Generating test predictions…")
    test_pred = trainer.predict(test_ds)
    probs = torch.softmax(torch.tensor(test_pred.predictions), dim=-1).numpy()
    pred_ids = probs.argmax(axis=-1)
    pred_labels = [id2label[int(i)] for i in pred_ids]

    # Build predictions DataFrame
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    out_df = pd.DataFrame(
        {
            "question": test_df[ARG1_KEY].values,
            "answer": test_df[ARG2_KEY].values,
            "model_prediction": pred_labels,  # required column name
            "true_label": test_df[TARGET_COLUMN].astype(str).values,
        }
    )
    for i, lab in sorted(id2label.items()):
        out_df[f"prob_{lab}"] = probs[:, i]

    # Detailed test report (sklearn)
    y_true_ids = test_proc["label"].to_numpy()
    y_pred_ids = pred_ids
    target_names = [id2label[i] for i in sorted(id2label)]
    test_report = classification_report(y_true_ids, y_pred_ids, target_names=target_names, digits=4, zero_division=0)
    with open(os.path.join(OUTPUT_DIR, "test_report.txt"), "w", encoding="utf-8") as f:
        f.write(test_report)
    print("\n[Info] Test classification report:")
    print(test_report)

    # ========================
    # TEST CONFUSION MATRIX
    # ========================
    test_cm = confusion_matrix(y_true_ids, y_pred_ids, labels=list(range(len(id2label))))
    test_cm_df = pd.DataFrame(test_cm, index=target_names, columns=target_names)
    test_cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_test.csv")
    test_cm_df.to_csv(test_cm_path)
    print(f"[Info] Saved test confusion matrix to {test_cm_path}")

    # Save predictions CSV
    test_csv_path = os.path.join(PREDICTIONS_DIR, f"{MODEL_NAME}_test_predictions.csv")
    out_df.to_csv(test_csv_path, index=False)
    print(f"[Info] Saved test predictions to {test_csv_path}")
    


if __name__ == "__main__":
    main()
