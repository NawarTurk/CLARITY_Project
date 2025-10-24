import inspect
import os
import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
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
# Configuration
# -----------------------------------------------------------------------------
SEED = 42
MODEL_NAME = "distilbert-base-uncased"
TRAIN_CSV_PATH = os.path.join("datasets", "train_dataset.csv")
TEST_CSV_PATH = os.path.join("datasets", "test_dataset.csv")
OUTPUT_DIR = os.path.join("results", "models", "distilbert-base-uncased")
FINAL_MODEL_DIR = os.path.join("finetuned-models", "distilbert-base-uncased")
PREDICTIONS_DIR = os.path.join("results", "predictions")
TARGET_COLUMN = "clarity_label"
QUESTION_COLUMN = "question"
ANSWER_COLUMN = "interview_answer"

NUM_EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
MAX_LENGTH = 256
VAL_SIZE = 0.1


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def set_global_seed(seed: int) -> None:
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    print(f"[Info] Global seed set to {seed}")


set_global_seed(SEED)


def load_and_prepare_dataframe() -> Tuple[pd.DataFrame, Dict[str, int], Dict[int, str]]:
    """Load the training CSV, keep relevant columns, and encode labels."""
    df = pd.read_csv(TRAIN_CSV_PATH)

    df = df.dropna(subset=[QUESTION_COLUMN, ANSWER_COLUMN, TARGET_COLUMN]).copy()
    df["text"] = df.apply(
        lambda row: f"Interview_question : {row[QUESTION_COLUMN]}\nAnswer: {row[ANSWER_COLUMN]}",
        axis=1,
    )

    unique_labels = sorted(df[TARGET_COLUMN].unique())
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    df["label"] = df[TARGET_COLUMN].map(label2id)

    return df[["text", "label"]], label2id, id2label


def prepare_datasets(df: pd.DataFrame) -> Tuple[Dataset, Dataset]:
    """Split the dataframe into train/validation subsets and convert to Hugging Face Datasets."""
    train_df, eval_df = train_test_split(
        df,
        test_size=VAL_SIZE,
        stratify=df["label"],
        random_state=SEED,
    )

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True), preserve_index=False)
    eval_ds = Dataset.from_pandas(eval_df.reset_index(drop=True), preserve_index=False)
    return train_ds, eval_ds


def tokenize_datasets(train_ds: Dataset, eval_ds: Dataset, tokenizer: AutoTokenizer) -> Tuple[Dataset, Dataset]:
    """Tokenize both datasets with the provided tokenizer."""

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    train_ds = train_ds.map(tokenize, batched=True)
    eval_ds = eval_ds.map(tokenize, batched=True)

    columns_to_return = ["input_ids", "attention_mask", "label"]
    train_ds.set_format(type="torch", columns=columns_to_return)
    eval_ds.set_format(type="torch", columns=columns_to_return)
    return train_ds, eval_ds


def compute_metrics(eval_pred):
    """Compute accuracy and F1 scores for Trainer."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average="macro")
    f1_micro = f1_score(labels, predictions, average="micro")
    return {"accuracy": accuracy, "f1_macro": f1_macro, "f1_micro": f1_micro}


def build_training_arguments(common_args: Dict) -> Tuple[TrainingArguments, bool]:
    """Create TrainingArguments compatible with the installed transformers version."""
    signature_params = inspect.signature(TrainingArguments.__init__).parameters
    args = dict(common_args)

    epoch_params = {
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_accuracy",
        "greater_is_better": True,
        "logging_strategy": "epoch",
        "report_to": "none",
    }

    supports_epoch_strategies = all(key in signature_params for key in ("evaluation_strategy", "save_strategy"))
    if supports_epoch_strategies:
        for key, value in epoch_params.items():
            if key in signature_params:
                args[key] = value
        return TrainingArguments(**args), True

    step_params = {
        "eval_steps": 100,
        "save_steps": 100,
        "logging_steps": 100,
    }
    for key, value in step_params.items():
        if key in signature_params:
            args[key] = value

    return TrainingArguments(**args), False

def load_test_dataframe(label2id: Dict[str, int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the test CSV, build the text column, and map labels with the training label2id."""
    if not os.path.exists(TEST_CSV_PATH):
        raise FileNotFoundError(f"Test CSV not found at {TEST_CSV_PATH}")

    df = pd.read_csv(TEST_CSV_PATH)
    df = df.dropna(subset=[QUESTION_COLUMN, ANSWER_COLUMN, TARGET_COLUMN]).copy()

    df["text"] = df.apply(
        lambda row: f"Interview_question : {row[QUESTION_COLUMN]}\nAnswer: {row[ANSWER_COLUMN]}",
        axis=1,
    )

    df_proc = df[["text"]].copy()
    df_proc["label"] = df[TARGET_COLUMN].map(label2id)

    return df_proc, df



# -----------------------------------------------------------------------------
# Training pipeline
# -----------------------------------------------------------------------------
def run_training() -> None:
    df, label2id, id2label = load_and_prepare_dataframe()
    train_ds, eval_ds = prepare_datasets(df)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds, eval_ds = tokenize_datasets(train_ds, eval_ds, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    common_args = dict(  # shared args across transformer versions
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        seed=SEED,
        data_seed=SEED,
        fp16=torch.cuda.is_available(),
    )

    training_args, epoch_strategies_supported = build_training_arguments(common_args)

    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)] if epoch_strategies_supported else None

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    print("[Info] Starting training...")
    trainer.train()

    print("[Info] Evaluating on validation split...")
    metrics = trainer.evaluate(eval_dataset=eval_ds)
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}" if isinstance(value, float) else f"  {name}: {value}")

    predictions = trainer.predict(eval_ds)
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    true_labels = predictions.label_ids

    target_names = [id2label[i] for i in sorted(id2label)]
    report = classification_report(true_labels, pred_labels, target_names=target_names, digits=4, zero_division=0)
    print("\n[Info] Validation classification report:")
    print(report)

    report_path = os.path.join(OUTPUT_DIR, "validation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[Info] Saved classification report to {report_path}")

    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    print(f"[Info] Saved fine-tuned model and tokenizer to {FINAL_MODEL_DIR}")

    print(f"[Info] Best model checkpoints remain under {OUTPUT_DIR}")

    # ----- Test predictions -----
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    test_proc_df, test_raw_df = load_test_dataframe(label2id)

    test_ds = Dataset.from_pandas(test_proc_df.reset_index(drop=True), preserve_index=False)

    def tok_test(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    test_ds = test_ds.map(tok_test, batched=True)
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Predictions (no training!)
    test_predictions = trainer.predict(test_ds)
    probs = torch.softmax(torch.tensor(test_predictions.predictions), dim=-1).numpy()
    pred_ids = probs.argmax(axis=-1)
    pred_labels = [id2label[int(i)] for i in pred_ids]

    # Build a per-row predictions CSV
    pred_df = pd.DataFrame({
        "question": test_raw_df[QUESTION_COLUMN].values,
        "answer":   test_raw_df[ANSWER_COLUMN].values,
        "pred_label": pred_labels,
    })
    for i, lab in sorted(id2label.items()):
        pred_df[f"prob_{lab}"] = probs[:, i]

    true_labels = test_raw_df[TARGET_COLUMN].astype(str).values
    pred_df["true_label"] = true_labels

    # Compute metrics on the test set
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    print("\n[Info] Test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    y_true_ids = test_proc_df["label"].to_numpy()
    y_pred_ids = pred_ids
    target_names = [id2label[i] for i in sorted(id2label)]
    test_report = classification_report(
        y_true_ids, y_pred_ids, target_names=target_names, digits=4, zero_division=0
    )
    print("\n[Info] Test classification report:")
    print(test_report)
    with open(os.path.join(OUTPUT_DIR, "test_report.txt"), "w", encoding="utf-8") as f:
        f.write(test_report)

    # Save predictions CSV
    test_pred_path = os.path.join(PREDICTIONS_DIR, f"{MODEL_NAME}_test_predictions.csv")
    pred_df.to_csv(test_pred_path, index=False)
    print(f"[Info] Saved test predictions to {test_pred_path}")



if __name__ == "__main__":
    run_training()
