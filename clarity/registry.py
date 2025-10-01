from typing import Dict

MODEL_REGISTRY: Dict[str, str] = {
    # LLaMA family
    "llama-2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    "llama-2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
    "llama-2-70b-chat": "meta-llama/Llama-2-70b-chat-hf",

    # Qwen family
    "qwen-0.5b": "Qwen/Qwen1.5-0.5B-Chat",
    "qwen-1.8b": "Qwen/Qwen1.5-1.8B-Chat",
    "qwen-7b": "Qwen/Qwen1.5-7B-Chat",

    # Other models
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "phi-2.7b": "microsoft/phi-2",
    "falcon-7b": "tiiuae/falcon-7b-instruct",
}


def resolve_model_id(name: str) -> str:
    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown model '{name}'. Known keys: {', '.join(sorted(MODEL_REGISTRY))}"
        ) from exc
