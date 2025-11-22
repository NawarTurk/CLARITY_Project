"""
Train encoder models using adapters (adapter-transformers style).

Usage:
    python models/encoders/train_adapter.py --model_name xlmr --param_mode fixed

Saves models to:
    models/encoders/trained_models/<t{task_id}_{arch}_{lang}_{size}_adapter_{param_mode}>
"""

import argparse
import os
import random
import re
import shutil
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, set_seed

from model_metadata import MODEL_METADATA

try:
    import adapters
    from adapters import AdapterConfig, AutoAdapterModel
except ImportError as exc:
    raise ImportError(
        "adapter-transformers support is required. Install with `pip install adapter-transformers` "
        "and ensure your transformers version is compatible."
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
ADAPTER_NAME = "clarity"

FIXED_CONFIG = {
    "num_train_epochs": 10,
    "learning_rate": 1e-4,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "weight_decay": 0.01,
    "max_length": 512,
    "adapter_reduction_factor": 16,
    "adapter_non_linearity": "relu",
}

HPS_CONFIG = {
    # Baseline for HPS sweep
    "num_train_epochs": 4,
    "learning_rate": 3e-5,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "weight_decay": 0.05,
    "max_length": 512,
    "adapter_reduction_factor": 8,
    "adapter_non_linearity": "swish",
}

HPS_SEARCH = [
    {
        "num_train_epochs": 4,
        "learning_rate": 3e-5,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "weight_decay": 0.05,
        "max_length": 512,
        "adapter_reduction_factor": 8,
        "adapter_non_linearity": "swish",
    },
    {
        "num_train_epochs": 5,
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "weight_decay": 0.1,
        "max_length": 512,
        "adapter_reduction_factor": 16,
        "adapter_non_linearity": "relu",
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
    slug = f"{TASK_ID}_{meta['arch']}_{meta['lang']}_{meta['size']}_adapter_{param_mode}"
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

    model = AutoAdapterModel.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    adapter_cfg = AdapterConfig.load(
        "pfeiffer",
        reduction_factor=config["adapter_reduction_factor"],
        non_linearity=config["adapter_non_linearity"],
    )
    model.add_adapter(ADAPTER_NAME, adapter_cfg)
    model.add_classification_head(ADAPTER_NAME, num_labels=len(label2id))
    model.train_adapter(ADAPTER_NAME)
    model.set_active_adapters(ADAPTER_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(
        train_ds,
        batch_size=config["per_device_train_batch_size"],
        shuffle=True,
        collate_fn=collator,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=config["per_device_eval_batch_size"],
        shuffle=False,
        collate_fn=collator,
    )

    # Only adapter/head params are trainable
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    best_acc = -1.0
    for epoch in range(config["num_train_epochs"]):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(1, len(train_loader))

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in dev_loader:
                labels = batch["labels"].to(device)
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        acc = accuracy_score(all_labels, all_preds)
        best_acc = max(best_acc, acc)
        print(f"[Epoch {epoch+1}] train_loss={avg_loss:.4f} eval_acc={acc:.4f}")

    _remove_checkpoint_dirs(output_dir)
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    model.save_adapter(final_model_dir, ADAPTER_NAME)
    model.save_head(final_model_dir, ADAPTER_NAME)
    print(f"[Done] Saved adapter model to {final_model_dir}")
    return best_acc, final_model_dir


def main():
    parser = argparse.ArgumentParser(description="Train encoders with adapters (adapter-transformers).")
    parser.add_argument(
        "--model_name",
        required=True,
        help="Base HF model key (e.g., xlmr, roberta, mbert, deberta) or 'all' to train every listed model.",
    )
    parser.add_argument("--param_mode", required=True, choices=("fixed", "hps"), help="Parameter mode to use.")
    args = parser.parse_args()

    model_names = _expand_models(args.model_name)

    set_global_seed(SEED)
    for resolved_model in model_names:
        print(f"\n[Run] Training {resolved_model} with adapters, mode={args.param_mode}")
        if args.param_mode == "fixed":
            train_once(resolved_model, args.param_mode, FIXED_CONFIG)
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
                print(f"[Info] Best HPS adapter model saved to {canonical_final_dir} (acc={best_acc:.4f})")


if __name__ == "__main__":
    main()
