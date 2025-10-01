from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .registry import resolve_model_id


def build_text_generation_pipeline(model_name: str):
    """Instantiate a text-generation pipeline for the requested model."""
    model_id = resolve_model_id(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)
