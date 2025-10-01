from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .config import Config
from .pipeline import build_text_generation_pipeline
from .prompts import build_prompt, load_template


def _ensure_prediction_column(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        df[column] = None


def _generate_prediction(prompt: str, generator, *, max_new_tokens: int, do_sample: bool) -> str:
    raw_output = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )[0]["generated_text"]
    return raw_output[len(prompt) :].strip()


def run_predictions(config: Config) -> Path:
    test_df = pd.read_csv(config.test_csv)
    print(f"Loaded {len(test_df)} rows of the test set")

    prediction_column = f"{config.model_name}_{Path(config.template_name).stem}"
    _ensure_prediction_column(test_df, prediction_column)

    template = load_template(config.prompts_dir, config.template_name)
    generator = build_text_generation_pipeline(config.model_name)

    output_path = config.output_csv()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for idx, row in tqdm(
        test_df.iterrows(), total=len(test_df), desc="Processing", unit="row"
    ):
        if pd.notna(row[prediction_column]):
            continue

        prompt = build_prompt(
            template,
            question=row["question"],
            answer=row["interview_answer"],
        )
        prediction = _generate_prediction(
            prompt,
            generator,
            max_new_tokens=config.max_new_tokens,
            do_sample=config.do_sample,
        )

        test_df.at[idx, prediction_column] = prediction

        print(f"Question: {row['question']}")
        print(f"Prediction: {prediction}\n")

        if config.checkpoint_interval and (idx + 1) % config.checkpoint_interval == 0:
            test_df.to_csv(output_path, index=False)

        if config.sleep_sec:
            time.sleep(config.sleep_sec)

    test_df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")
    return output_path
