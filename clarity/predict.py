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
    """Run the generator and return only the newly generated text."""
    generator_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }

    formatted_prompt = prompt
    uses_chat_template = False

    tokenizer = getattr(generator, "tokenizer", None)
    chat_template = getattr(tokenizer, "chat_template", None) if tokenizer else None
    if chat_template:  # Leverage chat formatting for chat-optimized models (e.g. Qwen).
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a classifier that must respond with exactly one label: "
                    "Clear Reply, Ambivalent Reply, or Clear Non-Reply."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        generator_kwargs["return_full_text"] = False
        uses_chat_template = True

    raw_output = generator(formatted_prompt, **generator_kwargs)[0]['generated_text']

    if not uses_chat_template and raw_output.startswith(formatted_prompt):
        raw_output = raw_output[len(formatted_prompt) :]

    cleaned_output = raw_output.strip()

    # Some chat-aligned models (e.g., Qwen 3B series) prepend optional <think>
    # reasoning blocks before the final answer. Strip those so we only return the
    # final label text.
    lowered = cleaned_output.lower()
    think_start = lowered.find('<think>')
    if think_start != -1:
        think_end = lowered.find('</think>', think_start)
        if think_end != -1:
            cleaned_output = (
                cleaned_output[:think_start]
                + cleaned_output[think_end + len('</think>') :]
            )
        else:
            cleaned_output = cleaned_output[:think_start]
        cleaned_output = cleaned_output.strip()

    if cleaned_output:
        return cleaned_output

    # As a final fallback, if generation was cut short we still try to recover
    # the label by scanning the raw output for any of the expected options.
    LABELS = [
        'Clear Non-Reply',
        'Ambivalent Reply',
        'Clear Reply',
    ]
    raw_lower = raw_output.lower()
    for label in LABELS:
        idx = raw_lower.find(label.lower())
        if idx != -1:
            return label

    return cleaned_output


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
