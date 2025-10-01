from __future__ import annotations
import sys
import time
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from clarity import Config, build_prompt, load_template, resolve_model_id


def build_text_generation_pipeline(model_name: str):
    """Instantiate a text-generation pipeline for the requested model."""
    model_id = resolve_model_id(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)


def prepare_dataframe(df: pd.DataFrame, prediction_column: str) -> None:
    if prediction_column not in df.columns:
        df[prediction_column] = None


def main(config: Config | None = None) -> None:
    config = config or Config()

    test_df = pd.read_csv(config.test_csv)
    print(f"Loaded {len(test_df)} rows of the test set")

    prediction_column = f"{config.model_name}_{Path(config.template_name).stem}"
    prepare_dataframe(test_df, prediction_column)

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

        raw_output = generator(
            prompt,
            max_new_tokens=config.max_new_tokens,
            do_sample=config.do_sample,
        )[0]["generated_text"]

        prediction = raw_output[len(prompt) :].strip()
        test_df.at[idx, prediction_column] = prediction

        print(f"Question: {row['question']}")
        print(f"Prediction: {prediction}\n")

        if config.checkpoint_interval and (idx + 1) % config.checkpoint_interval == 0:
            test_df.to_csv(output_path, index=False)

        if config.sleep_sec:
            time.sleep(config.sleep_sec)

    test_df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
