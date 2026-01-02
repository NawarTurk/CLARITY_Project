"""
Majority-vote ensemble for selected LLM prompt prediction CSVs.

- Task: t1
- Majority vote only
- No confidence
- Ties are resolved as Ambivalent Reply
- Explicit, canonical filenames
"""

from pathlib import Path
import pandas as pd
from collections import Counter


# ---------------- CONFIG ----------------

SCRIPT_DIR = Path(__file__).parent
# ENSEMBLE_FILES = ["qwen3-235b-instruct_04_t1_fs_base-27-shot_IQ_nebius_VALIDATED.csv",
#         "mixtral-8x22b-Instruct_01_t1_zs_re2_IQ_nscale_VALIDATED.csv",
#         "llama-3.1-405b-Instruct_02_t1_fs_base-3-shot_Q_nebius_VALIDATED.csv",
# "gpt-5_01_t1_zs_re2_Q_openai_VALIDATED.csv"]

ENSEMBLE_FILES = [
    # Qwen-235B | FS | base-27-shot | IQ
    "qwen3-235b-instruct_04_t1_fs_base-27-shot_IQ_nebius_VALIDATED.csv",

    # Qwen-Coder-480B | ZS | re2 | IQ
    "qwen3-coder-480b-instruct_01_t1_zs_re2_IQ_nebius_VALIDATED.csv",

    # GPT-5 | ZS | re2 | Q
    "gpt-5_01_t1_zs_re2_Q_openai_VALIDATED.csv",

    # LLaMA-3.3-70B | FS | base-27-shot | IQ
    "llama-3.3-70b-instruct_04_t1_fs_base-27-shot_IQ_nebius_VALIDATED.csv",
]

ID_COL = "index"
TARGET_COL = "clarity_label"
PRED_COL = "model_prediction"

AMBIVALENT_LABEL = "Ambivalent"

OUTPUT_PATH = (
    SCRIPT_DIR
    / "gpt-5_BOARD_majority_majority_majority_majority_majority_VALIDATED.csv"
)


# ---------------- HELPERS ----------------

def majority_vote(labels_by_model):
    labels = [v for v in labels_by_model.values() if pd.notna(v)]
    if not labels:
        return None

    counts = Counter(labels)
    max_count = max(counts.values())
    top_labels = [l for l, c in counts.items() if c == max_count]

    # clear winner
    if len(top_labels) == 1:
        return top_labels[0]

    # tie → Ambivalent
    return AMBIVALENT_LABEL


# ---------------- MAIN ----------------

def main():
    # sanity check
    for fname in ENSEMBLE_FILES:
        path = SCRIPT_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    print("Using the following models:")
    for f in ENSEMBLE_FILES:
        print(f"  - {f}")

    # load gold labels from first file
    gold_df = pd.read_csv(SCRIPT_DIR / ENSEMBLE_FILES[0])
    gold_labels = gold_df.set_index(ID_COL)[TARGET_COL]

    preds = {}

    for fname in ENSEMBLE_FILES:
        df = pd.read_csv(SCRIPT_DIR / fname)
        model_key = fname.replace(".csv", "")
        preds[model_key] = df.set_index(ID_COL)[PRED_COL]

    all_ids = sorted(set().union(*[s.index for s in preds.values()]))

    rows = []
    tie_count = 0

    for idx in all_ids:
        row = {
            ID_COL: idx,
            TARGET_COL: gold_labels.loc[idx],
        }

        labels_by_model = {}

        for model in preds:
            label = preds[model].get(idx)
            labels_by_model[model] = label
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
