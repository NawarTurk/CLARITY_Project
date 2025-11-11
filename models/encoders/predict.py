import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "datasets" / "test_dataset.csv"
MODELS_DIR = ROOT / "models" / "encoders" / "trained_models"
OUT_DIR = ROOT / "results" / "predictions" / "detailed" / "encoder"

ARG1_KEY = "question"
ARG2_KEY = "interview_answer"
BATCH_SIZE = 16
MAX_LENGTH = 512


def list_model_dirs(model_name: str):
    if model_name.lower() == "all":
        return sorted(p for p in MODELS_DIR.iterdir() if p.is_dir())
    return [MODELS_DIR / model_name]


def predict(model_dir: Path, df: pd.DataFrame):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    arg1 = df[ARG1_KEY].astype(str).fillna("").tolist()
    arg2 = df[ARG2_KEY].astype(str).fillna("").tolist()

    preds = []
    probs = []
    for start in range(0, len(df), BATCH_SIZE):
        batch_arg1 = arg1[start : start + BATCH_SIZE]
        batch_arg2 = arg2[start : start + BATCH_SIZE]
        inputs = tokenizer(
            batch_arg1,
            batch_arg2,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        batch_probs = torch.softmax(logits, dim=-1)
        max_probs, max_ids = batch_probs.max(dim=-1)
        preds.extend(max_ids.cpu().tolist())
        probs.extend(max_probs.cpu().tolist())

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    df_out = df.copy()
    df_out["predicted_label"] = [id2label.get(i, str(i)) for i in preds]
    df_out["predicted_confidence"] = probs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{model_dir.name}_predictions.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, help="Model folder name or 'all'.")
    args = parser.parse_args(argv)

    df = pd.read_csv(DATASET_PATH)
    for model_dir in list_model_dirs(args.model_name):
        predict(model_dir, df)


if __name__ == "__main__":
    main(sys.argv[1:])
