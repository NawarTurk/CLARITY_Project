import os
import pandas as pd
from collections import Counter, defaultdict

STAGE_DIR = "."  # run from inside the stage folder
OUTPUT_FILE = "t1_majorityS4_en_majorityS4_majorityS4_majorityS4_majorityS4_majorityS4_majorityS4_majorityS4_predictions.csv"

# --------------------------------------------------
# Collect all CSV files recursively
# --------------------------------------------------
csv_files = []
for root, _, files in os.walk(STAGE_DIR):
    for f in files:
        if f.endswith(".csv") and f != OUTPUT_FILE:
            csv_files.append(os.path.join(root, f))

dfs = [pd.read_csv(f) for f in csv_files]

# --------------------------------------------------
# Majority voting
# --------------------------------------------------
num_rows = len(dfs[0])
final_labels = []

for i in range(num_rows):
    labels = []
    confs = defaultdict(list)

    for df in dfs:
        label = df.loc[i, "predicted_label"]
        conf  = df.loc[i, "predicted_confidence"]
        labels.append(label)
        confs[label].append(conf)

    counts = Counter(labels)
    max_count = max(counts.values())

    tied = [l for l, c in counts.items() if c == max_count]

    if len(tied) == 1:
        final_label = tied[0]
    else:
        final_label = max(
            tied,
            key=lambda l: sum(confs[l]) / len(confs[l])
        )

    final_labels.append(final_label)

# --------------------------------------------------
# Overwrite predictions and remove confidence
# --------------------------------------------------
out = dfs[0].copy()
out["predicted_label"] = final_labels

if "predicted_confidence" in out.columns:
    out = out.drop(columns=["predicted_confidence"])

out.to_csv(OUTPUT_FILE, index=False)

print(f"Saved → {OUTPUT_FILE}")
