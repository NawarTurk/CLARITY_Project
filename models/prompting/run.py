from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from clarity import Config, run_predictions

# Update these constants directly when you want to try a different setup.
MODEL_NAME = "qwen-0.5b"
PROMPT_TEMPLATE = "zero-shot01.txt"
MAX_NEW_TOKENS = 100
DO_SAMPLE = False
CHECKPOINT_INTERVAL = 10
SLEEP_SECONDS = 0.0


def build_config() -> Config:
    return Config(
        model_name=MODEL_NAME,
        template_name=PROMPT_TEMPLATE,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        sleep_sec=SLEEP_SECONDS,
    )


def main(config: Config | None = None) -> None:
    config = config or build_config()
    run_predictions(config)


if __name__ == "__main__":
    main()
