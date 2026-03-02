"""
Majority-vote ensemble for all encoder prediction CSVs in the script directory.
- Task: t1
- Majority vote; ties resolved by highest mean predicted_confidence
"""

from pathlib import Path
import pandas as pd
from collections import Counter

SCRIPT_DIR = Path(__file__).parent

ID_COL = "index"
TARGET_COL = "clarity_label"
PRED_COL = "predicted_label"
CONF_COL = "predicted_confidence"

OUTPUT_PATH = SCRIPT_DIR / "t1_encoder-ensemble_en+multi_base+large_encoder-ensemble_fixed_encoder-ensemble_encoder-ensemble_WCE_encoder-ensemble_encoder-ensemble_predictions.csv"


def majority_vote(labels_by_model, conf_by_model):
    labels = [v for v in labels_by_model.values() if pd.notna(v)]
    if not labels:
        return None
    counts = Counter(labels)
    max_count = max(counts.values())
    top_labels = [l for l, c in counts.items() if c == max_count]
    if len(top_labels) == 1:
        return top_labels[0]

    # Tie: pick label with highest mean confidence
    label_conf = {}
    for label in top_labels:
        confs = [
            conf_by_model[model]
            for model, pred in labels_by_model.items()
            if pred == label and pd.notna(conf_by_model.get(model))
        ]
        label_conf[label] = sum(confs) / len(confs) if confs else 0
    return max(label_conf, key=label_conf.get)


def main():
    ensemble_files = [
        f for f in SCRIPT_DIR.glob("*predictions.csv")
        if f.name != OUTPUT_PATH.name
    ]

    if not ensemble_files:
        raise FileNotFoundError(f"No prediction CSVs found in {SCRIPT_DIR}")

    print(f"Found {len(ensemble_files)} files:")
    for f in ensemble_files:
        print(f"  - {f.name}")

    gold_df = pd.read_csv(ensemble_files[0])
    gold_labels = gold_df.set_index(ID_COL)[TARGET_COL]

    preds = {}
    confs = {}
    for f in ensemble_files:
        df = pd.read_csv(f).set_index(ID_COL)
        key = f.stem
        preds[key] = df[PRED_COL]
        confs[key] = df[CONF_COL] if CONF_COL in df.columns else pd.Series(dtype=float)

    all_ids = sorted(set().union(*[s.index for s in preds.values()]))

    rows = []
    tie_count = 0

    for idx in all_ids:
        row = {ID_COL: idx, TARGET_COL: gold_labels.loc[idx]}
        labels_by_model = {model: preds[model].get(idx) for model in preds}
        conf_by_model = {model: confs[model].get(idx) for model in confs}

        for model, label in labels_by_model.items():
            row[f"{model}__predicted_label"] = label

        counts = Counter(v for v in labels_by_model.values() if pd.notna(v))
        if counts and len([c for c in counts.values() if c == max(counts.values())]) > 1:
            tie_count += 1

        ensemble_label = majority_vote(labels_by_model, conf_by_model)
        row[PRED_COL] = ensemble_label

        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n✅ Ensemble written to: {OUTPUT_PATH}")
    print(f"Rows: {len(out_df)}")
    print(f"Ties resolved by confidence: {tie_count}")


if __name__ == "__main__":
    main()