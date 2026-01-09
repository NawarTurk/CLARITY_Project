import sys
import os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput

# PEFT (LoRA)
from peft import PeftModel

from models.encoders.s4_data_augmentation.stage4_train_freeze import (
    CLSHeadSequenceClassifier,
)
from models.encoders.model_metadata import MODEL_METADATA


ARG1_KEY = "question"
ARG2_KEY = "interview_answer"
MAX_LENGTH = 512
BATCH_SIZE = 16
TARGET_COLUMN = "clarity_label"

# Project root (CLARITY_Project), used for dataset/model paths
ROOT = Path(__file__).resolve().parents[3]
TRAIN_DATASET = ROOT / "datasets" / "train_dataset.csv"


def _ensure_enriched_column(df: pd.DataFrame) -> None:
    if "enriched_input" in df.columns:
        return
    if "q" in df.columns:
        df["enriched_input"] = df["q"]
        return
    if "question" not in df.columns or "interview_question" not in df.columns:
        raise KeyError("Missing columns needed for enriched input: question/interview_question.")
    df["enriched_input"] = df.apply(
        lambda row: (
            f"Target question: {row['question']}\n\n"
            f"Full interviewer turn (context): {row['interview_question']}"
        ),
        axis=1,
    )


def _infer_input_mode_from_slug(slug: str) -> str:
    parts = slug.split("_")
    for raw in reversed(parts):
        if raw in ("atomic", "enriched"):
            return raw
    return "atomic"


# =============================================================================
# Head patching (copied from your LoRA training script so inference matches)
# =============================================================================

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
    # all token embeddings → masked average pooling → Linear → num_labels logits
    def __init__(self, hidden_size: int, num_labels: int = 3):
        super().__init__()
        self.fc = nn.Linear(hidden_size, num_labels)

    def forward(self, pooled_embedding: torch.Tensor) -> torch.Tensor:
        return self.fc(pooled_embedding)


def _filter_forward_args(module: nn.Module, kwargs: Dict[str, object]) -> Dict[str, object]:
    """Pass only kwargs that exist in the target module.forward signature."""
    try:
        import inspect
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


def _encode_batch(
    tokenizer: AutoTokenizer,
    texts1: list[str],
    texts2: list[str],
    truncation_mode: str,
):
    """
    Encode a batch using either standard head truncation or head-tail logic,
    mirroring the Stage 4 training scripts.
    """
    truncation_mode = (truncation_mode or "head").strip()
    if truncation_mode != "head_tail":
        return tokenizer(
            texts1,
            texts2,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

    # head_tail: keep first 256 and last 256 tokens for long sequences
    raw = tokenizer(
        texts1,
        texts2,
        truncation=False,
        padding=False,
        return_attention_mask=True,
    )

    input_ids_list = raw["input_ids"]
    attn_list = raw["attention_mask"]
    type_ids_list = raw.get("token_type_ids")

    head_len = MAX_LENGTH // 2
    tail_len = MAX_LENGTH - head_len

    new_input_ids: list[list[int]] = []
    new_attn: list[list[int]] = []
    new_type_ids: Optional[list[list[int]]] = [] if type_ids_list is not None else None

    for i in range(len(input_ids_list)):
        ids = input_ids_list[i]
        mask = attn_list[i]

        # Short sequences: keep as-is
        if len(ids) <= MAX_LENGTH:
            new_input_ids.append(ids)
            new_attn.append(mask)
            if new_type_ids is not None and type_ids_list is not None:
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

        if new_type_ids is not None and type_ids_list is not None:
            t_ids = type_ids_list[i]
            new_type_ids.append(t_ids[:head_len] + t_ids[-tail_len:])

    encoded: dict[str, list[list[int]]] = {
        "input_ids": new_input_ids,
        "attention_mask": new_attn,
    }
    if new_type_ids is not None:
        encoded["token_type_ids"] = new_type_ids

    padded = tokenizer.pad(
        encoded,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    return padded


# =============================================================================
# Shared helpers (mirroring Stage 1 prediction logic)
# =============================================================================

def _infer_base_model_from_slug(slug: str) -> str | None:
    """
    Reverse the slug logic used in training to recover the base HF model id.

    Slug format (from training scripts):
      TASK_ARCH_LANG_SIZE_STRATEGY_PARAMMODE_HEAD_TRUNC
      e.g. t1_bert_en_base_unfreezing25_fixed_defaultHead_head_tail
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


_CACHED_ID2LABEL: Dict[int, str] | None = None


def _get_label_mapping_fallback() -> Dict[int, str]:
    """Fallback label mapping (sorted unique clarity_label) from train_dataset.csv."""
    global _CACHED_ID2LABEL
    if _CACHED_ID2LABEL is not None:
        return _CACHED_ID2LABEL

    if not TRAIN_DATASET.exists():
        raise FileNotFoundError(
            f"Training dataset not found at {TRAIN_DATASET} "
            "needed to reconstruct label mapping."
        )
    df = pd.read_csv(TRAIN_DATASET)
    if TARGET_COLUMN not in df.columns:
        raise KeyError(
            f"Column '{TARGET_COLUMN}' not found in {TRAIN_DATASET}. "
            "Cannot reconstruct label mapping."
        )
    unique_labels = sorted(df[TARGET_COLUMN].dropna().unique())
    id2label = {i: lab for i, lab in enumerate(unique_labels)}
    _CACHED_ID2LABEL = id2label
    return id2label


def _load_label_mapping(model_dir: Path) -> Dict[int, str]:
    """
    Prefer the exact mapping saved at training time:
      1) head_meta.json (your custom saver)
      2) config.json id2label (HF standard)
      3) fallback to train_dataset.csv reconstruction
    """
    head_meta_path = model_dir / "head_meta.json"
    if head_meta_path.exists():
        try:
            with open(head_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            id2 = meta.get("id2label", None)
            if isinstance(id2, dict) and id2:
                # keys might be strings in JSON
                return {int(k): v for k, v in id2.items()}
        except Exception:
            pass

    cfg_path = model_dir / "config.json"
    if cfg_path.exists():
        try:
            cfg = AutoConfig.from_pretrained(model_dir)
            id2 = getattr(cfg, "id2label", None)
            if isinstance(id2, dict) and id2:
                return {int(k): v for k, v in id2.items()}
        except Exception:
            pass

    return _get_label_mapping_fallback()


def _infer_truncation_from_slug(slug: str) -> str:
    """
    Infer truncation mode by scanning slug tokens.

    Training uses:
      - 'head-tail' (or legacy 'head_tail') for head-tail truncation
      - 'head' for standard head truncation
    """
    parts = slug.split("_")
    if not parts:
        return "head"
    for raw in reversed(parts):
        if raw in ("head-tail", "head_tail"):
            return "head_tail"
        if raw == "head":
            return "head"
    return "head"


def _infer_head_type_from_slug_or_meta(slug: str, model_dir: Path) -> str:
    """
    Prefer head_meta.json; else infer from slug.
    """
    head_type = "mlp"
    head_meta_path = model_dir / "head_meta.json"
    if head_meta_path.exists():
        try:
            with open(head_meta_path, "r", encoding="utf-8") as f:
                head_meta = json.load(f)
            head_type = (head_meta.get("head_type", "mlp") or "mlp").strip().lower()
            print(f"[Info] Loaded head_type='{head_type}' from {head_meta_path}")
            return head_type
        except Exception:
            pass

    # Infer from slug when no explicit metadata is present
    if "defaultHead" in slug:
        return "default"
    if "avgPoolHead" in slug:
        return "avgpool"
    return "mlp"


def _patch_seqcls_head(model: nn.Module, head_type: str) -> None:
    """
    Replace the classifier head AND patch forward to use:
      - mlp: CLS token embedding (last_hidden[:,0,:]) → MLPHead
      - avgpool: masked mean pool(last_hidden) → AvgPoolHead

    This keeps the model as a HF PreTrainedModel (important for PEFT),
    while ensuring behavior matches your desired CLS strategy.
    """
    import types

    head_type = (head_type or "mlp").strip().lower()
    if head_type == "default":
        # Keep the HF default classification head; just record type for logging.
        setattr(model, "head_type", "default")
        return
    if head_type not in ("mlp", "avgpool"):
        raise ValueError(f"Unsupported head_type '{head_type}'. Use 'default', 'mlp', or 'avgpool'.")

    hidden_size = int(getattr(getattr(model, "config", None), "hidden_size", 0) or 0)
    if hidden_size <= 0:
        hidden_size = int(getattr(getattr(model, "config", None), "d_model", 0) or 0)
    if hidden_size <= 0:
        raise ValueError("Could not infer hidden size from model.config (hidden_size/d_model).")

    num_labels = int(getattr(getattr(model, "config", None), "num_labels", 0) or 0)
    if num_labels <= 0:
        raise ValueError("Could not infer num_labels from model.config.")

    # Replace classifier module
    if head_type == "mlp":
        new_head = MLPHead(hidden_size=hidden_size, num_labels=num_labels)
    else:
        new_head = AvgPoolHead(hidden_size=hidden_size, num_labels=num_labels)

    # Most seqcls models use .classifier; some use .score
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
        labels=None,  # ignored for inference
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
            pooled = last_hidden[:, 0, :]  # CLS/<s>
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


def _load_state_dict_any(model_dir: Path) -> dict:
    """
    Load weights from:
      - pytorch_model.bin
      - model.safetensors (optional)
    """
    bin_path = model_dir / "pytorch_model.bin"
    if bin_path.exists():
        return torch.load(bin_path, map_location="cpu")

    st_path = model_dir / "model.safetensors"
    if st_path.exists():
        try:
            from safetensors.torch import load_file as safe_load_file  # type: ignore
            return safe_load_file(str(st_path))
        except Exception as e:
            raise RuntimeError(f"Found {st_path} but could not load safetensors: {e}")

    raise FileNotFoundError(f"No weights found in {model_dir} (expected pytorch_model.bin or model.safetensors).")


# =============================================================================
# Loading logic (FIXED)
# =============================================================================

def _load_model_and_tokenizer(model_dir: Path, slug: str, *, merge_lora: bool = False):
    """Load a classifier for a given trained-model directory.

    Supports:
      1) Full HF checkpoints (default head): config.json + (pytorch_model.bin | model.safetensors)
      2) LoRA adapter-only dirs: adapter_model.* (+ adapter_config.json)
      3) Full custom CLSHeadSequenceClassifier checkpoints (mlp/avgpool): pytorch_model.bin (+ config.json optional)
      4) Legacy CLSHeadSequenceClassifier state dict: pytorch_model.bin only
    """
    # Label mapping
    id2label = _load_label_mapping(model_dir)
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)

    base_model_name = _infer_base_model_from_slug(slug)
    if base_model_name is None:
        raise ValueError(
            f"Could not infer base model name from slug '{slug}'. "
            "Ensure it follows the training naming convention."
        )

    head_type = _infer_head_type_from_slug_or_meta(slug, model_dir)

    has_config = (model_dir / "config.json").exists()
    has_full_weights = any((model_dir / name).exists() for name in ("pytorch_model.bin", "model.safetensors"))
    has_adapter = any((model_dir / name).exists() for name in ("adapter_model.bin", "adapter_model.safetensors"))

    # Tokenizer: prefer model_dir (saved tokenizer), fallback to base model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # -----------------------------
    # ✅ FIX #1: Custom-head FULL checkpoints must load as CLSHeadSequenceClassifier,
    # not AutoModelForSequenceClassification. Otherwise weights can mismatch / partially load.
    # -----------------------------
    is_custom_head = (head_type in ("mlp", "avgpool")) and (not has_adapter)

    if is_custom_head and (model_dir / "pytorch_model.bin").exists():
        model = CLSHeadSequenceClassifier(
            model_name=str(base_model_name),
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            head_type=head_type,
        )
        sd = _load_state_dict_any(model_dir)

        # ✅ FIX #2: strict=True so you cannot silently load the wrong architecture
        # (prevents "close but different" results from partially initialized weights).
        model.load_state_dict(sd, strict=True)

        print(f"[Info] Loaded FULL custom CLSHeadSequenceClassifier from {model_dir} (head_type={head_type})")
        return model, tokenizer, id2label

    # Case 2: LoRA adapter-only directory
    if has_adapter:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )

        # Recreate head+forward BEFORE loading adapter, matching training
        _patch_seqcls_head(base_model, head_type=head_type)

        model = PeftModel.from_pretrained(base_model, model_dir)

        if merge_lora:
            model = model.merge_and_unload()

        print(f"[Info] Loaded base model '{base_model_name}' + LoRA adapter from {model_dir} (head_type={head_type})")
        return model, tokenizer, id2label

    # Case 1: Default-head full HF checkpoint
    if has_config and has_full_weights and head_type == "default":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )
        print(f"[Info] Loaded full HF model (default head) from {model_dir}")
        return model, tokenizer, id2label

    # Legacy fallback: state_dict only
    if (model_dir / "pytorch_model.bin").exists():
        model = CLSHeadSequenceClassifier(
            model_name=str(base_model_name),
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            head_type=head_type,
        )
        sd = torch.load(model_dir / "pytorch_model.bin", map_location="cpu")
        model.load_state_dict(sd, strict=True)
        print(f"[Info] Loaded legacy CLSHeadSequenceClassifier state_dict from {model_dir} (head_type={head_type})")
        return model, tokenizer, id2label

    raise FileNotFoundError(
        f"No model weights found in {model_dir}. Expected one of:\n"
        "  - FULL default-head HF: config.json + (pytorch_model.bin | model.safetensors)\n"
        "  - FULL custom-head CLSHeadSequenceClassifier: pytorch_model.bin (+ head_meta.json)\n"
        "  - LoRA adapter-only: adapter_model.* with adapter_config.json\n"
        "  - legacy: pytorch_model.bin only"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict using Stage 4 encoder models")
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Model slug under Stage 4 trained models, or 'all' to run for every model",
    )
    parser.add_argument(
        "--input_mode",
        choices=("atomic", "enriched"),
        default=None,
        help="Override input mode (atomic/enriched). Defaults to slug inference.",
    )
    parser.add_argument(
        "--merge_lora",
        action="store_true",
        help="If set, merges LoRA weights into the base model for inference (LoRA models only).",
    )
    args = parser.parse_args()

    root = ROOT
    dataset_path = root / "datasets" / "test_dataset.csv"
    models_root = root / "models" / "encoders" / "s4_data_augmentation" / "stage4_trained_models"
    # Save Stage 4 predictions under the shared encoder results folder
    out_dir = root / "results" / "predictions" / "encoder" / "stage4"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Determine which model directories to run: one slug or all
    if args.model_dir.lower() == "all":
        model_dirs = sorted(p for p in models_root.iterdir() if p.is_dir())
        if not model_dirs:
            raise FileNotFoundError(f"No Stage 4 trained models found in {models_root}")
    else:
        candidate = models_root / args.model_dir
        if not candidate.exists() or not candidate.is_dir():
            available = [p.name for p in models_root.iterdir() if p.is_dir()] if models_root.exists() else []
            raise FileNotFoundError(
                f"Stage 4 model '{args.model_dir}' not found in {models_root}. "
                f"Available: {', '.join(available) if available else 'None'}"
            )
        model_dirs = [candidate]

    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=[ARG1_KEY, ARG2_KEY]).copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_dir in model_dirs:
        slug = model_dir.name
        print(f"[Stage 4] Predicting with model '{slug}'...")

        model, tokenizer, id2label = _load_model_and_tokenizer(model_dir, slug, merge_lora=args.merge_lora)
        model.to(device).eval()

        truncation_mode = _infer_truncation_from_slug(slug)
        input_mode = args.input_mode or _infer_input_mode_from_slug(slug)
        if input_mode == "enriched":
            _ensure_enriched_column(df)
            arg1_key = "enriched_input"
        else:
            arg1_key = ARG1_KEY

        arg1 = df[arg1_key].astype(str).tolist()
        arg2 = df[ARG2_KEY].astype(str).tolist()

        preds: list[int] = []
        probs: list[float] = []

        for start in range(0, len(df), BATCH_SIZE):
            batch_arg1 = arg1[start: start + BATCH_SIZE]
            batch_arg2 = arg2[start: start + BATCH_SIZE]

            inputs = _encode_batch(
                tokenizer=tokenizer,
                texts1=batch_arg1,
                texts2=batch_arg2,
                truncation_mode=truncation_mode,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

            batch_probs = torch.softmax(logits, dim=-1)
            max_probs, max_ids = batch_probs.max(dim=-1)

            preds.extend(max_ids.detach().cpu().tolist())
            probs.extend(max_probs.detach().cpu().tolist())

        df_out = df.copy()
        df_out["predicted_label"] = [id2label[int(i)] for i in preds]
        df_out["predicted_confidence"] = probs

        out_path = out_dir / f"{slug}_predictions.csv"
        df_out.to_csv(out_path, index=False)
        print(f"[DONE] Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
