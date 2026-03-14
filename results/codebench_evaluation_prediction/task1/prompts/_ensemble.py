import pandas as pd
from collections import Counter

# ===============================
# INPUT FILES (top 3 models)
# ===============================

FILES = [
    "gemini-3-flash-preview_04_t1_fs_base-27-shot_IQ-label-details-president_gemini_VALIDATED.csv",
    "gpt-5_04_t1_fs_base-27-shot_IQ-label-details_openai_VALIDATED.csv",
    "qwen3-235b-instruct_04_t1_fs_base-27-shot_IQ-label-details-president_together_VALIDATED.csv",
]

# ===============================
# OUTPUT FILE (YOUR NAME — NOT CHANGED)
# ===============================

OUTPUT_FILE = "gpt-5-gemini-3-flash-preview-qwen3-235b-instruct_04_t1_fs_base-27-shot_IQ-label-details-president-1_ensamble_VALIDATED.csv"

DEFAULT_LABEL = "Ambivalent"

# ===============================
# LOAD FILES (NO MERGE)
# ===============================

dfs = [pd.read_csv(f) for f in FILES]

num_rows = len(dfs[0])

for df in dfs:
    assert len(df) == num_rows
    assert "model_prediction" in df.columns

# ===============================
# MAJORITY VOTE (ROW-WISE)
# ===============================

final_preds = []

for i in range(num_rows):
    labels = [df.loc[i, "model_prediction"] for df in dfs]

    counts = Counter(labels)
    max_count = max(counts.values())
    winners = [l for l, c in counts.items() if c == max_count]

    final_label = winners[0] if len(winners) == 1 else DEFAULT_LABEL
    final_preds.append(final_label)

# ===============================
# OVERWRITE + SAVE
# ===============================

out = dfs[0].copy()
out["model_prediction"] = final_preds

out.to_csv(OUTPUT_FILE, index=False)

print(f"Saved → {OUTPUT_FILE}")
