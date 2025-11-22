"""
Train encoder models using LoRA (low-rank adaptation).

Usage:
    python models/encoders/train_lora.py --model_name xlmr --param_mode fixed

Saves models to:
    models/encoders/trained_models/<t{task_id}_{arch}_{lang}_{size}_lora_{param_mode}>
"""

import argparse
import inspect
import os
import random
import re
from typing import Dict, Tuple
import shutil

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from model_metadata import MODEL_METADATA

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError as exc:
    raise ImportError(
        "peft is required for LoRA training. Install with `pip install peft`."
    ) from exc

# -----------------------------------------------------------------------------#
# Constants / configuration
# -----------------------------------------------------------------------------#
SEED = 42
TASK_ID = "t1"
TRAIN_CSV_PATH = os.path.join("datasets", "train_dataset.csv")
DEV_CSV_PATH = os.path.join("datasets", "test_dataset.csv")  # Using test as held-out
ARG1_KEY = "question"
ARG2_KEY = "interview_answer"
TARGET_COLUMN = "clarity_label"

FIXED_CONFIG = {
    "num_train_epochs": 3,
    "learning_rate": 5e-5,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "weight_decay": 0.01,
    "max_length": 512,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
}

HPS_CONFIG = {
    # Representative alternate settings (no sweep infra here).
    "num_train_epochs": 4,
    "learning_rate": 3e-5,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "weight_decay": 0.05,
    "max_length": 512,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.15,
}

HPS_SEARCH = [
    {
        "num_train_epochs": 4,
        "learning_rate": 3e-5,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "weight_decay": 0.05,
        "max_length": 512,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.15,
    },
    {
        "num_train_epochs": 5,
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "weight_decay": 0.1,
        "max_length": 512,
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
    },
]


# -----------------------------------------------------------------------------#
# Utilities
# -----------------------------------------------------------------------------#
def _fs_safe_model_name(name: str) -> str:
    normalized = (
        name.strip()
        .replace(os.sep, "-")
        .replace("/", "-")
        .replace("\\", "-")
    )
    safe = re.sub(r"[^0-9A-Za-z._-]+", "-", normalized).strip("-._")
    return safe or "model"


def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"[Info] Global seed set: {seed}")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


def _build_output_slug(model_name: str, param_mode: str) -> Tuple[str, Dict]:
    if model_name not in MODEL_METADATA:
        raise KeyError(
            f"Model '{model_name}' missing from MODEL_METADATA. "
            "Add it to models/encoders/model_metadata.py."
        )
    meta = MODEL_METADATA[model_name]
    slug = f"{TASK_ID}_{meta['arch']}_{meta['lang']}_{meta['size']}_lora_{param_mode}"
    return _fs_safe_model_name(slug), meta


def _resolve_model_name(user_name: str) -> str:
    """Map shorthand model keys to full HF IDs and infer size."""
    name = user_name.strip()
    lower = name.lower()
    is_large = "large" in lower

    if name in MODEL_METADATA:
        return name

    if "xlmr" in lower:
        return "FacebookAI/xlm-roberta-large" if is_large else "FacebookAI/xlm-roberta-base"
    if "roberta" in lower:
        return "roberta-large" if is_large else "roberta-base"
    if "deberta" in lower:
        return "microsoft/deberta-v3-large" if is_large else "microsoft/deberta-v3-base"
    if "mbert" in lower or "bert-base-multilingual" in lower:
        return "bert-base-multilingual-cased"
    if "bert" in lower:
        return "bert-base-uncased"

    return name


def _expand_models(user_name: str):
    """Return a list of model identifiers to train."""
    if user_name.strip().lower() == "all":
        return list(MODEL_METADATA.keys())
    return [_resolve_model_name(user_name)]


def _remove_checkpoint_dirs(path: str) -> None:
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


def _prepare_datasets(tokenizer, max_length: int) -> Tuple[Dataset, Dataset, Dict[str, int], Dict[int, str]]:
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    dev_df = pd.read_csv(DEV_CSV_PATH)

    train_df = train_df.dropna(subset=[ARG1_KEY, ARG2_KEY, TARGET_COLUMN]).copy()
    dev_df = dev_df.dropna(subset=[ARG1_KEY, ARG2_KEY, TARGET_COLUMN]).copy()

    for df in (train_df, dev_df):
        df[ARG1_KEY] = df[ARG1_KEY].astype(str)
        df[ARG2_KEY] = df[ARG2_KEY].astype(str)

    unique_labels = sorted(train_df[TARGET_COLUMN].dropna().unique())
    label2id: Dict[str, int] = {lab: i for i, lab in enumerate(unique_labels)}
    id2label: Dict[int, str] = {i: lab for lab, i in label2id.items()}

    train_df["labels"] = train_df[TARGET_COLUMN].map(label2id)
    dev_df["labels"] = dev_df[TARGET_COLUMN].map(label2id)
    if train_df["labels"].isnull().any() or dev_df["labels"].isnull().any():
        raise ValueError("Found label values not present in training label set; cannot encode labels.")

    def tokenize(batch):
        return tokenizer(
            batch[ARG1_KEY],
            batch[ARG2_KEY],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    train_enc = Dataset.from_pandas(train_df, preserve_index=False)
    dev_enc = Dataset.from_pandas(dev_df, preserve_index=False)

    train_enc = train_enc.map(tokenize, batched=True)
    dev_enc = dev_enc.map(tokenize, batched=True)

    def _format(ds: Dataset) -> Dataset:
        cols = ["input_ids", "attention_mask", "labels"]
        if "token_type_ids" in ds.column_names:
            cols.append("token_type_ids")
        ds.set_format("torch", columns=cols)
        return ds

    train_enc = _format(train_enc)
    dev_enc = _format(dev_enc)

    return train_enc, dev_enc, label2id, id2label


def train_once(model_name: str, param_mode: str, config: Dict, run_suffix: str | None = None) -> tuple[float, str]:
    slug, meta = _build_output_slug(model_name, param_mode)
    if run_suffix:
        slug = f"{slug}_{run_suffix}"
    output_dir = os.path.join("results", "models", slug)
    final_model_dir = os.path.join("models", "encoders", "trained_models", slug)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(final_model_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds, dev_ds, label2id, id2label = _prepare_datasets(tokenizer, max_length=config["max_length"])

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias="none",
    )
    model = get_peft_model(base_model, lora_cfg)

    training_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir=os.path.join(output_dir, "logs"),
        logging_strategy="epoch",
        report_to="none",
    )
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" not in ta_params and "eval_strategy" in ta_params:
        training_kwargs["eval_strategy"] = training_kwargs.pop("evaluation_strategy")
    filtered_kwargs = {k: v for k, v in training_kwargs.items() if k in ta_params}
    args = TrainingArguments(**filtered_kwargs)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate(eval_dataset=dev_ds)
    eval_acc = float(eval_metrics.get("eval_accuracy", 0.0))
    _remove_checkpoint_dirs(output_dir)
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print(f"[Done] Saved LoRA model to {final_model_dir}")
    return eval_acc, final_model_dir


def main():
    parser = argparse.ArgumentParser(description="Train encoders with LoRA.")
    parser.add_argument(
        "--model_name",
        required=True,
        help="Base HF model key (e.g., xlmr, roberta, mbert, deberta) or 'all' to train every listed model.",
    )
    parser.add_argument("--param_mode", required=True, choices=("fixed", "hps"), help="Parameter mode to use.")
    args = parser.parse_args()

    model_names = _expand_models(args.model_name)
    param_config = FIXED_CONFIG if args.param_mode == "fixed" else HPS_CONFIG

    set_global_seed(SEED)
    for resolved_model in model_names:
        print(f"\n[Run] Training {resolved_model} with LoRA, mode={args.param_mode}")
        if args.param_mode == "fixed":
            train_once(resolved_model, args.param_mode, param_config)
        else:
            best_acc = -1.0
            best_dir = None
            canonical_slug, _ = _build_output_slug(resolved_model, args.param_mode)
            canonical_final_dir = os.path.join("models", "encoders", "trained_models", canonical_slug)
            for idx, cfg in enumerate(HPS_SEARCH, start=1):
                run_suffix = f"hps{idx}"
                acc, final_dir = train_once(resolved_model, args.param_mode, cfg, run_suffix=run_suffix)
                if acc > best_acc:
                    best_acc = acc
                    best_dir = final_dir
            if best_dir:
                if os.path.exists(canonical_final_dir):
                    shutil.rmtree(canonical_final_dir, ignore_errors=True)
                shutil.copytree(best_dir, canonical_final_dir)
                print(f"[Info] Best HPS model saved to {canonical_final_dir} (acc={best_acc:.4f})")


if __name__ == "__main__":
    main()
