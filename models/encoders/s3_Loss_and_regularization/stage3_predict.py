import sys
import os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

# PEFT (LoRA)
from peft import PeftModel

from models.encoders.model_metadata import MODEL_METADATA


ARG1_KEY = "question"
ARG2_KEY = "interview_answer"
MAX_LENGTH = 512
BATCH_SIZE = 16
TARGET_COLUMN = "clarity_label"

# Project root (CLARITY_Project), used for dataset/model paths
ROOT = Path(__file__).resolve().parents[3]
TRAIN_DATASET = ROOT / "datasets" / "train_dataset.csv"


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


class CLSHeadSequenceClassifier(nn.Module):
    """
    Lightweight classifier wrapper for legacy Stage 3 checkpoints that only
    store `pytorch_model.bin` (no config).
    """

    def __init__(
        self,
        model_name: str,
        *,
        num_labels: int,
        id2label: Dict[int, str],
        label2id: Dict[str, int],
        head_type: str = "mlp",
    ):
        super().__init__()

        head_type = (head_type or "mlp").strip().lower()
        if head_type not in ("mlp", "avgpool"):
            raise ValueError(f"Unsupported head_type='{head_type}'. Use 'mlp' or 'avgpool'.")

        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = int(num_labels)
        config.id2label = dict(id2label)
        config.label2id = dict(label2id)

        self.config = config
        self.model_name = model_name
        self.head_type = head_type

        self.base_model = AutoModel.from_pretrained(model_name, config=config)

        hidden_size = int(getattr(config, "hidden_size", None) or getattr(config, "d_model", 0) or 0)
        if hidden_size <= 0:
            raise ValueError(f"Could not determine hidden size from config for model: {model_name}")

        if head_type == "mlp":
            self.classifier = MLPHead(hidden_size=hidden_size, num_labels=num_labels)
        else:
            self.classifier = AvgPoolHead(hidden_size=hidden_size, num_labels=num_labels)

    @staticmethod
    def _masked_mean_pool(last_hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if attention_mask is None:
            return last_hidden.mean(dim=1)
        mask = attention_mask.to(dtype=last_hidden.dtype).unsqueeze(-1)  # [B, S, 1]
        summed = (last_hidden * mask).sum(dim=1)                          # [B, H]
        denom = mask.sum(dim=1).clamp(min=1e-6)                           # [B, 1]
        return summed / denom

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,  # ignored for inference
        **kwargs,
    ):
        payload = dict(kwargs)
        payload["input_ids"] = input_ids
        payload["attention_mask"] = attention_mask
        payload["token_type_ids"] = token_type_ids
        payload.setdefault("return_dict", True)

        filtered = _filter_forward_args(self.base_model, payload)
        outputs = self.base_model(**filtered)

        last_hidden = getattr(outputs, "last_hidden_state", None)
        if last_hidden is None:
            raise RuntimeError("Backbone did not return last_hidden_state; cannot pool.")

        if self.head_type == "mlp":
            embedding = last_hidden[:, 0, :]  # CLS / <s>
        else:
            embedding = self._masked_mean_pool(last_hidden, attention_mask)

        logits = self.classifier(embedding)
        return SequenceClassifierOutput(logits=logits)


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
    mirroring the Stage 2 training scripts.
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


def _get_label_mapping() -> Dict[int, str]:
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


def _infer_truncation_from_slug(slug: str) -> str:
    """
    Infer truncation mode from the slug tokens.

    Training uses a token for truncation ('head-tail'/'head_tail' or 'head'),
    and Stage 3 appends an extra loss/dropout token at the end, e.g.
      ..._head-tail_WCE-Dropout30

    For prediction, we scan from the end and pick the first token that matches
    a known truncation tag.
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


# =============================================================================
# Loading logic
# =============================================================================

def _load_model_and_tokenizer(model_dir: Path, slug: str, *, merge_lora: bool = False):
    """Load a classifier for a given trained-model directory.

    Supports:
      1) Full HF checkpoints: config.json + (pytorch_model.bin | model.safetensors)
      2) LoRA adapter-only dirs: adapter_model.* (+ adapter_config.json)
      3) Older CLSHeadSequenceClassifier checkpoints: pytorch_model.bin only
    """
    # Label mapping (same helper as models/encoders/predict.py)
    id2label = _get_label_mapping()
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)

    base_model_name = _infer_base_model_from_slug(slug)
    if base_model_name is None:
        raise ValueError(
            f"Could not infer base model name from slug '{slug}'. "
            "Ensure it follows the training naming convention."
        )

    # Head type – used for LoRA + legacy CLSHeadSequenceClassifier
    head_type = "mlp"
    head_meta_path = model_dir / "head_meta.json"
    if head_meta_path.exists():
        with open(head_meta_path, "r", encoding="utf-8") as f:
            head_meta = json.load(f)
        head_type = (head_meta.get("head_type", "mlp") or "mlp").strip().lower()
        print(f"[Info] Loaded head_type='{head_type}' from {head_meta_path}")
    else:
        # Infer head_type from slug when no explicit metadata is present
        if "defaultHead" in slug:
            head_type = "default"
        elif "avgPoolHead" in slug:
            head_type = "avgpool"

    has_config = (model_dir / "config.json").exists()
    has_full_weights = any((model_dir / name).exists() for name in ("pytorch_model.bin", "model.safetensors"))
    has_adapter = any((model_dir / name).exists() for name in ("adapter_model.bin", "adapter_model.safetensors"))

    # Tokenizer: prefer model_dir (saved tokenizer), fallback to base model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    if has_config and has_full_weights:
        # Case 1: standard HF checkpoint (includes custom head weights)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )
        print(f"[Info] Loaded full HF model from {model_dir}")

    elif has_adapter:
        # Case 2: LoRA adapter-only directory
        #
        # CRITICAL FIX:
        # - Training patched the head + forward (mlp/avgpool) BEFORE LoRA injection,
        #   and saved the head via modules_to_save.
        # - For correct inference, we must recreate that same head+forward BEFORE loading adapter,
        #   otherwise adapter head weights won't load / will mismatch silently.
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )

        _patch_seqcls_head(base_model, head_type=head_type)

        # Load adapter (handles .bin or .safetensors, adapter_config.json, modules_to_save, etc.)
        model = PeftModel.from_pretrained(base_model, model_dir)

        if merge_lora:
            # Optional: merge LoRA into base weights for faster inference
            model = model.merge_and_unload()

        print(f"[Info] Loaded base model '{base_model_name}' + LoRA adapter from {model_dir} (head_type={head_type})")

    elif (model_dir / "pytorch_model.bin").exists():
        # Case 3: legacy CLSHeadSequenceClassifier state dict
        state_dict_path = model_dir / "pytorch_model.bin"
        model = CLSHeadSequenceClassifier(
            model_name=str(base_model_name),
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            head_type=head_type,
        )
        state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        print(f"[Info] Loaded CLSHeadSequenceClassifier state_dict from {state_dict_path}")

    else:
        raise FileNotFoundError(
            f"No model weights found in {model_dir}. Expected one of:\n"
            "  - config.json + (pytorch_model.bin | model.safetensors)\n"
            "  - adapter_model.* with adapter_config.json\n"
            "  - pytorch_model.bin (CLSHeadSequenceClassifier state_dict)"
        )

    return model, tokenizer, id2label


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict using Stage 3 encoder models")
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Model slug under Stage 3 trained models, or 'all' to run for every model",
    )
    parser.add_argument(
        "--merge_lora",
        action="store_true",
        help="If set, merges LoRA weights into the base model for inference (LoRA models only).",
    )
    args = parser.parse_args()

    root = ROOT
    dataset_path = root / "datasets" / "test_dataset.csv"
    models_root = root / "models" / "encoders" / "s3_Loss_and_regularization" / "stage3_trained_models"
    # Save Stage 3 predictions under encoder/stage3
    out_dir = root / "results" / "predictions" / "encoder" / "stage3"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Determine which model directories to run: one slug or all
    if args.model_dir.lower() == "all":
        model_dirs = sorted(p for p in models_root.iterdir() if p.is_dir())
        if not model_dirs:
            raise FileNotFoundError(f"No Stage 3 trained models found in {models_root}")
    else:
        candidate = models_root / args.model_dir
        if not candidate.exists() or not candidate.is_dir():
            available = [p.name for p in models_root.iterdir() if p.is_dir()] if models_root.exists() else []
            raise FileNotFoundError(
                f"Stage 3 model '{args.model_dir}' not found in {models_root}. "
                f"Available: {', '.join(available) if available else 'None'}"
            )
        model_dirs = [candidate]

    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=[ARG1_KEY, ARG2_KEY]).copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_dir in model_dirs:
        slug = model_dir.name
        print(f"[Stage 3] Predicting with model '{slug}'...")

        model, tokenizer, id2label = _load_model_and_tokenizer(model_dir, slug, merge_lora=args.merge_lora)
        model.to(device).eval()

        truncation_mode = _infer_truncation_from_slug(slug)

        arg1 = df[ARG1_KEY].astype(str).tolist()
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
