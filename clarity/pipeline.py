import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .registry import resolve_model_id


def build_text_generation_pipeline(model_name: str):
    """Instantiate a text-generation pipeline for the requested model."""
    model_id = resolve_model_id(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model_kwargs = {}
    pipeline_kwargs = {"device": 0 if torch.cuda.is_available() else -1}
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    return pipeline("text-generation", model=model, tokenizer=tokenizer, **pipeline_kwargs)
