import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from model_metadata import MODEL_METADATA

try:
    from peft import PeftModel  # type: ignore
except ImportError:
    PeftModel = None  # type: ignore

ROOT = Path(__file__).resolve().parents[2]
TEST_DATASET = ROOT / "datasets" / "test_dataset.csv"
TRAIN_DATASET = ROOT / "datasets" / "train_dataset.csv"
MODELS_DIR = ROOT / "models" / "encoders" / "trained_models"
OUT_DIR = ROOT / "results" / "predictions" / "encoder" / "stage1"

ARG1_KEY = "question"
ARG2_KEY = "interview_answer"
BATCH_SIZE = 16
MAX_LENGTH = 512
TARGET_COLUMN = "clarity_label"


def list_model_dirs(model_name: str):
    """Return model directories to run predictions against, validating existence."""
    available = sorted(p for p in MODELS_DIR.iterdir() if p.is_dir())
    if model_name.lower() == "all":
        if not available:
            raise FileNotFoundError(f"No trained models found in {MODELS_DIR}")
        return available

    candidate = MODELS_DIR / model_name
    if not candidate.exists() or not candidate.is_dir():
        available_names = ", ".join(p.name for p in available) if available else "None"
        raise FileNotFoundError(
            f"Trained model '{model_name}' not found in {MODELS_DIR}. "
            f"Available: {available_names}"
        )
    return [candidate]


def _is_lora_dir(model_dir: Path) -> bool:
    return any(
        (model_dir / name).exists()
        for name in ("adapter_model.safetensors", "adapter_config.json")
    )


def _validate_model_artifacts(model_dir: Path) -> None:
    """Ensure the trained model folder has the minimum files required to load.

    Supports two layouts:
      1) Full fine-tuned model: config.json + model weights (+ tokenizer)
      2) LoRA adapter-only: adapter_model.safetensors (+ adapter_config) + tokenizer
    """
    if not any(model_dir.iterdir()):
        raise FileNotFoundError(
            f"Trained model folder is empty: {model_dir}. "
            "Train the model or copy the saved artifacts into this directory."
        )

    def has_base_weights() -> bool:
        return any(
            (model_dir / name).exists()
            for name in ("pytorch_model.bin", "model.safetensors")
        )

    def has_tokenizer_files() -> bool:
        return any(
            (model_dir / name).exists()
            for name in (
                "tokenizer.json",
                "vocab.txt",
                "tokenizer.model",
                "spiece.model",
                "sentencepiece.bpe.model",
            )
        )

    has_config = (model_dir / "config.json").exists()
    has_model_bin = has_base_weights()
    has_tokenizer = has_tokenizer_files()
    lora_only = _is_lora_dir(model_dir) and not (has_config and has_model_bin)

    # Case 1: full model folder
    if has_config and has_model_bin and has_tokenizer:
        return

    # Case 2: LoRA-only folder (adapter + tokenizer; base will be loaded separately)
    if lora_only and has_tokenizer:
        if PeftModel is None:
            raise ImportError(
                "Detected a LoRA adapter-only checkpoint but the 'peft' package "
                "is not installed. Install it with `pip install -U peft`."
            )
        return

    missing = []
    if not has_tokenizer:
        missing.append("tokenizer files (tokenizer.json/vocab.txt/... )")
    if not (has_config and has_model_bin) and not lora_only:
        missing.append("config.json and model weights (pytorch_model.bin or model.safetensors)")
    if lora_only and not has_tokenizer:
        missing.append("tokenizer files for LoRA adapter")

    raise FileNotFoundError(
        f"Trained model folder is missing required files: {', '.join(missing)} in {model_dir}. "
        "For full models, include config.json + model weights + tokenizer. "
        "For LoRA adapters, include adapter_model.safetensors/adapter_config.json + tokenizer, "
        "and ensure 'peft' is installed."
    )


def _infer_base_model_from_readme(model_dir: Path) -> str | None:
    """Try to read a base_model field from README front matter."""
    readme = model_dir / "README.md"
    if not readme.exists():
        return None
    try:
        text = readme.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("base_model:"):
            parts = stripped.split(":", 1)
            if len(parts) == 2:
                candidate = parts[1].strip()
                if candidate:
                    return candidate
    return None


def _infer_base_model_from_slug(slug: str) -> str | None:
    """
    Reverse the slug logic used in training to recover the base HF model id.

    Slug format (from training scripts):
      TASK_ARCH_LANG_SIZE_STRATEGY_PARAMMODE_HEAD
      e.g. t1_bert_en_base_lora_fixed_defaultHead
    """
    parts = slug.split("_")
    if len(parts) < 4:
        return None
    _, arch, lang, size = parts[:4]
    for hf_name, meta in MODEL_METADATA.items():
        if (
            meta.get("arch") == arch
            and meta.get("lang") == lang
            and meta.get("size") == size
        ):
            return hf_name
    return None


def _get_base_model_name_for_lora(model_dir: Path) -> str:
    """Resolve the base model name for a LoRA adapter directory."""
    name = _infer_base_model_from_readme(model_dir)
    if name:
        return name

    slug = model_dir.name
    inferred = _infer_base_model_from_slug(slug)
    if inferred:
        return inferred

    raise ValueError(
        f"Could not infer base model name for LoRA directory {model_dir}. "
        "Add a 'base_model: <hf_id>' line to README.md or ensure the folder "
        "name follows the slug convention used in training."
    )


_CACHED_ID2LABEL = None


def _get_label_mapping():
    """Load label mapping used during training (sorted unique clarity_label)."""
    global _CACHED_ID2LABEL
    if _CACHED_ID2LABEL is not None:
        return _CACHED_ID2LABEL

    if not TRAIN_DATASET.exists():
        raise FileNotFoundError(
            f"Training dataset not found at {TRAIN_DATASET} "
            "needed to reconstruct label mapping for LoRA models."
        )
    df = pd.read_csv(TRAIN_DATASET)
    if TARGET_COLUMN not in df.columns:
        raise KeyError(
            f"Column '{TARGET_COLUMN}' not found in {TRAIN_DATASET}. "
            "Cannot reconstruct label mapping for LoRA models."
        )
    unique_labels = sorted(df[TARGET_COLUMN].dropna().unique())
    id2label = {i: lab for i, lab in enumerate(unique_labels)}
    _CACHED_ID2LABEL = id2label
    return id2label


def _load_tokenizer(model_dir: Path):
    return AutoTokenizer.from_pretrained(model_dir)


def _load_model(model_dir: Path, id2label=None):
    """Load either a full fine-tuned model or a base+LoRA PEFT model."""
    has_config = (model_dir / "config.json").exists()
    has_model_bin = any(
        (model_dir / name).exists()
        for name in ("pytorch_model.bin", "model.safetensors")
    )

    # Full model checkpoint stored in this directory
    if has_config and has_model_bin:
        return AutoModelForSequenceClassification.from_pretrained(model_dir)

    # LoRA adapter-only directory
    adapter_present = _is_lora_dir(model_dir)
    if adapter_present:
        if PeftModel is None:
            raise ImportError(
                "LoRA model directory detected but 'peft' is not installed. "
                "Install it with `pip install -U peft`."
            )
        base_model_name = _get_base_model_name_for_lora(model_dir)
        init_kwargs = {}
        if id2label is not None:
            num_labels = len(id2label)
            label2id = {v: k for k, v in id2label.items()}
            init_kwargs.update(
                {
                    "num_labels": num_labels,
                    "id2label": id2label,
                    "label2id": label2id,
                }
            )
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name, **init_kwargs
        )
        return PeftModel.from_pretrained(base_model, model_dir)

    raise ValueError(
        f"Could not determine how to load model from directory {model_dir}. "
        "Expected either a full HF model (config + weights) or a LoRA adapter."
    )


def predict(model_dir: Path, df: pd.DataFrame):
    _validate_model_artifacts(model_dir)
    tokenizer = _load_tokenizer(model_dir)
    # For LoRA-only directories, reconstruct the label mapping so that
    # the base model is initialized with the correct classifier size.
    id2label = None
    if _is_lora_dir(model_dir) and not (model_dir / "config.json").exists():
        id2label = _get_label_mapping()
    model = _load_model(model_dir, id2label=id2label)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    arg1 = df[ARG1_KEY].astype(str).fillna("").tolist()
    arg2 = df[ARG2_KEY].astype(str).fillna("").tolist()

    preds = []
    probs = []
    for start in range(0, len(df), BATCH_SIZE):
        batch_arg1 = arg1[start : start + BATCH_SIZE]
        batch_arg2 = arg2[start : start + BATCH_SIZE]
        inputs = tokenizer(
            batch_arg1,
            batch_arg2,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        batch_probs = torch.softmax(logits, dim=-1)
        max_probs, max_ids = batch_probs.max(dim=-1)
        preds.extend(max_ids.cpu().tolist())
        probs.extend(max_probs.cpu().tolist())

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    df_out = df.copy()
    df_out["predicted_label"] = [id2label.get(i, str(i)) for i in preds]
    df_out["predicted_confidence"] = probs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{model_dir.name}_predictions.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, help="Model folder name or 'all'.")
    args = parser.parse_args(argv)

    df = pd.read_csv(TEST_DATASET)
    for model_dir in list_model_dirs(args.model_name):
        try:
            predict(model_dir, df)
        except (FileNotFoundError, ImportError, ValueError) as e:
            # Gracefully skip models with incomplete/invalid artifacts
            print(f"Skipping {model_dir.name} because: {e}")


if __name__ == "__main__":
    main(sys.argv[1:])
