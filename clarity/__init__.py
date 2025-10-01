"""Convenience exports for the clarity package."""

from .cli import build_config_from_cli
from .config import Config, project_root
from .pipeline import build_text_generation_pipeline
from .predict import run_predictions
from .prompts import build_prompt, load_template
from .registry import MODEL_REGISTRY, resolve_model_id
