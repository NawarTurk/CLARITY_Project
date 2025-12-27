import random
from pathlib import Path
import pandas as pd
import json


RANDOM_SEED = 42
random.seed(RANDOM_SEED)

NEGATIVE_COUNT = 5  # per class, 5 + 5 = 10 in total

CLASSES = [
    'Clear Reply',
    'Ambivalent',
    'Clear Non-Reply'
]

PROJECT_ROOT= Path(__file__).resolve().parent.parent
TRAIN_DATASET_PATH = PROJECT_ROOT / "datasets" / "train_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "agent_examples"/"v2"
OUTPUT_DIR.mkdir(exist_ok=True)

def format_qa(row):
    return {
        "question": row["question"],
        "answer": row["interview_answer"],  
        "clarity_label": row["clarity_label"],
        "evasion_label": row["evasion_label"]
    }

def sample_positive(df, target_label):
    filtered_positive = df[df['clarity_label'] == target_label]
    grouped_positive = filtered_positive.groupby('evasion_label')
    
    if target_label == "Clear Reply":
        # per_group = 15
        per_group = 10
    elif target_label == "Ambivalent":
        # per_group = 3
        per_group = 2
    elif target_label == "Clear Non-Reply":
        # per_group = 5
        per_group = 4
    else:
        raise ValueError("Unknown clarity label")
    
    positive_samples = []

    for evasion_label, group in grouped_positive:
        n = min(per_group, len(group))
        sampled = group.sample(n=n, random_state=RANDOM_SEED, replace=False)
        for _, row in sampled.iterrows():
            positive_samples.append(format_qa(row))
    
    return positive_samples[:10] # ++

def sample_negative(df, target_label):
    filtered_negative = df[df['clarity_label'] != target_label]
    grouped_negative = filtered_negative.groupby('clarity_label')

    negative_samples = []

    for evasion_label, group in grouped_negative:
        n = min(NEGATIVE_COUNT, len(group))
        sampled = group.sample(n=n, random_state=RANDOM_SEED, replace=False)
        for _, row in sampled.iterrows():
            negative_samples.append(format_qa(row))
    
    return negative_samples

if __name__ == "__main__":
    df = pd.read_csv(TRAIN_DATASET_PATH)

    for target_label in CLASSES:
        positive_samples = sample_positive(df, target_label)
        negative_samples = sample_negative(df, target_label)

        output = {
            "target_label": target_label,
            "positives": positive_samples,
            "negatives": negative_samples
        }      

        output_path = OUTPUT_DIR / f"agent_examples_{target_label.replace(' ', '_').lower()}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)



