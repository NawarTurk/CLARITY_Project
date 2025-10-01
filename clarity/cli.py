from __future__ import annotations

import argparse
from typing import Sequence

from .config import Config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CLARITY predictions")
    parser.add_argument("--model", dest="model_name", help="Model key from the registry")
    parser.add_argument("--template", dest="template_name", help="Prompt template filename relative to prompts/")
    parser.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, help="Maximum new tokens to generate")
    parser.add_argument("--do-sample", dest="do_sample", action="store_true", help="Enable sampling (default off)")
    parser.add_argument("--checkpoint-interval", dest="checkpoint_interval", type=int, help="Rows between interim saves (0 disables)")
    parser.add_argument("--batch-size", dest="batch_size", type=int, help="Batch size placeholder (unused for now)")
    parser.add_argument("--sleep", dest="sleep_sec", type=float, help="Seconds to sleep between rows")
    parser.add_argument("--seed", dest="seed", type=int, help="Seed value for reproducibility")
    return parser


def build_config_from_cli(argv: Sequence[str] | None = None) -> Config:
    parser = _build_parser()
    args = parser.parse_args(argv)
    overrides = {key: value for key, value in vars(args).items() if value is not None}

    config = Config()
    for key, value in overrides.items():
        setattr(config, key, value)

    return config
