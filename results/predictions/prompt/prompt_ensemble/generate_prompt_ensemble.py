"""
Majority-vote ensemble for all VALIDATED LLM prompt prediction CSVs in the script directory.
- Task: t1
- Majority vote only
- Ties resolved as Ambivalent Reply
"""

from pathlib import Path
import pandas as pd
from collections import Counter

SCRIPT_DIR = Path(__file__).parent

ID_COL = "index"
TARGET_COL = "clarity_label"
PRED_COL = "model_prediction"
AMBIVALENT_LABEL = "Ambivalent"

OUTPUT_PATH = SCRIPT_DIR / "gpt-5-gemini-3-flash-preview-qwen3-235b-instruct_promptEnsembleMajorityVote_promptEnsembleMajorityVote_promptEnsembleMajorityVote_promptEnsembleMajorityVote_promptEnsembleMajorityVote_promptEnsembleMajorityVote_VALIDATED.csv"


def majority_vote(labels_by_model):
    labels = [v for v in labels_by_model.values() if pd.notna(v)]
    if not labels:
        return None
    counts = Counter(labels)
    max_count = max(counts.values())
    top_labels = [l for l, c in counts.items() if c == max_count]
    if len(top_labels) == 1:
        return top_labels[0]
    return AMBIVALENT_LABEL


def main():
    ensemble_files = [
        f for f in SCRIPT_DIR.glob("*VALIDATED.csv")
        if f.name != OUTPUT_PATH.name
    ]

    if not ensemble_files:
        raise FileNotFoundError(f"No VALIDATED CSVs found in {SCRIPT_DIR}")

    print(f"Found {len(ensemble_files)} VALIDATED files:")
    for f in ensemble_files:
        print(f"  - {f.name}")

    gold_df = pd.read_csv(ensemble_files[0])
    gold_labels = gold_df.set_index(ID_COL)[TARGET_COL]

    preds = {}
    for f in ensemble_files:
        df = pd.read_csv(f)
        preds[f.stem] = df.set_index(ID_COL)[PRED_COL]

    all_ids = sorted(set().union(*[s.index for s in preds.values()]))

    rows = []
    tie_count = 0

    for idx in all_ids:
        row = {ID_COL: idx, TARGET_COL: gold_labels.loc[idx]}
        labels_by_model = {model: preds[model].get(idx) for model in preds}

        for model, label in labels_by_model.items():
            row[f"{model}__predicted_label"] = label

        ensemble_label = majority_vote(labels_by_model)
        row[PRED_COL] = ensemble_label

        counts = Counter(v for v in labels_by_model.values() if pd.notna(v))
        if counts and len([c for c in counts.values() if c == max(counts.values())]) > 1:
            tie_count += 1

        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n✅ Ensemble written to: {OUTPUT_PATH}")
    print(f"Rows: {len(out_df)}")
    print(f"Ties mapped to Ambivalent: {tie_count}")


if __name__ == "__main__":
    main()