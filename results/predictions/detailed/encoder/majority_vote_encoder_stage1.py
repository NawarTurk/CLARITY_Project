"""
Majority-vote ensemble for encoder prediction CSVs.

- Task: t1
- Head: fixed_defaultHead
- Adds clarity_label so F1 evaluation works
- Majority vote with confidence-based tie breaking
"""

from pathlib import Path
import pandas as pd
from collections import Counter
import re


# ---------------- CONFIG ----------------

ENSEMBLE_FILES = [
    "t1_deberta_en_base_unfreezing25_fixed_defaultHead_originalData_truncHead_lossCE_predictions.csv",
    "t1_mdeberta_multi_base_unfreezing25_fixed_defaultHead_originalData_truncHead_lossCE_predictions.csv",
    "t1_bert_en_base_lora16_fixed_defaultHead_originalData_truncHead_lossCE_predictions.csv",
    "t1_mbert_multi_base_unfreezing25_fixed_defaultHead_originalData_truncHead_lossCE_predictions.csv",
    "t1_roberta_en_base_unfreezing75_fixed_defaultHead_originalData_truncHead_lossCE_predictions.csv",
    "t1_roberta_en_large_unfreezing25_fixed_defaultHead_originalData_truncHead_lossCE_predictions.csv",
    "t1_xlmr_multi_base_unfreezing25_fixed_defaultHead_originalData_truncHead_lossCE_predictions.csv",
    "t1_xlmr_multi_large_unfreezing25_fixed_defaultHead_originalData_truncHead_lossCE_predictions.csv",
]

ID_COL = "index"
TARGET_COL = "clarity_label"
PRED_COL = "predicted_label"
CONF_COL = "predicted_confidence"

DATA_DIR = Path(__file__).parent
OUTPUT_PATH = DATA_DIR / "t1_majorityStage1_majorityStage1_majorityStage1_majorityStage1_fixed_defaultHead_predictions.csv"


# ---------------- HELPERS ----------------

def sanitize_name(filename: str) -> str:
    s = filename.lower()
    s = s.replace("t1_", "").replace("_fixed_defaulthead_predictions.csv", "")
    s = re.sub(r"[^a-z0-9_\-]", "", s)
    return s


def majority_vote(labels_by_model, confs_by_model):
    labels = [v for v in labels_by_model.values() if pd.notna(v)]
    if not labels:
        return None

    counts = Counter(labels)
    max_count = max(counts.values())
    top_labels = [l for l, c in counts.items() if c == max_count]

    # clear winner
    if len(top_labels) == 1:
        return top_labels[0]

    # tie → highest confidence wins
    best_label = None
    best_conf = -1.0

    for label in top_labels:
        confs = [
            confs_by_model[m]
            for m, l in labels_by_model.items()
            if l == label and pd.notna(confs_by_model.get(m))
        ]
        if confs and max(confs) > best_conf:
            best_conf = max(confs)
            best_label = label

    # deterministic fallback
    return best_label if best_label is not None else sorted(top_labels)[0]


# ---------------- MAIN ----------------

def main():
    # sanity check
    for fname in ENSEMBLE_FILES:
        path = DATA_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    # load gold labels from first file
    gold_df = pd.read_csv(DATA_DIR / ENSEMBLE_FILES[0])
    gold_labels = gold_df.set_index(ID_COL)[TARGET_COL]

    preds = {}
    confs = {}

    for fname in ENSEMBLE_FILES:
        df = pd.read_csv(DATA_DIR / fname)
        key = sanitize_name(fname)

        preds[key] = df.set_index(ID_COL)[PRED_COL]

        if CONF_COL in df.columns:
            confs[key] = df.set_index(ID_COL)[CONF_COL]
        else:
            confs[key] = pd.Series(index=preds[key].index, dtype=float)

    all_ids = sorted(set().union(*[s.index for s in preds.values()]))

    rows = []
    tie_count = 0

    for idx in all_ids:
        row = {
            ID_COL: idx,
            TARGET_COL: gold_labels.loc[idx],
        }

        labels_by_model = {}
        confs_by_model = {}

        for model in preds:
            label = preds[model].get(idx)
            conf = confs[model].get(idx)

            labels_by_model[model] = label
            confs_by_model[model] = conf

            row[f"{model}__predicted_label"] = label
            row[f"{model}__predicted_confidence"] = conf

        ensemble_label = majority_vote(labels_by_model, confs_by_model)
        row[PRED_COL] = ensemble_label

        # count ties
        counts = Counter(v for v in labels_by_model.values() if pd.notna(v))
        if counts and len([c for c in counts.values() if c == max(counts.values())]) > 1:
            tie_count += 1

        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_PATH, index=False)

    print(f"✅ Ensemble written to: {OUTPUT_PATH}")
    print(f"Models used: {len(ENSEMBLE_FILES)}")
    print(f"Rows: {len(out_df)}")
    print(f"Ties resolved by confidence: {tie_count}")


if __name__ == "__main__":
    main()
