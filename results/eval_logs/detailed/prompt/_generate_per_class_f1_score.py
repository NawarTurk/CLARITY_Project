import re
from pathlib import Path
import pandas as pd

REPORT_DIR = Path(".")
OUTPUT_CSV = "_per_class_f1_summary.csv"

LABELS = [
    "Ambivalent",
    "Clear Non-Reply",
    "Clear Reply",
]

# Match label rows in the classification report
ROW_RE = re.compile(
    r"^\s*(Ambivalent|Clear Non-Reply|Clear Reply)\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)",
    re.MULTILINE
)

rows = []

files = list(REPORT_DIR.glob("*_f1Report.txt"))
print(f"Found {len(files)} report files")

for report_path in files:
    text = report_path.read_text()

    parts = report_path.stem.split("_")

    # Expected minimal structure:
    # model, exp_id, task, prompt_type, prompt_variant, Q, provider, VALIDATED, f1Report
    if len(parts) < 9:
        print(f"⚠️ Skipping malformed filename: {report_path.name}")
        continue

    row = {
        "model": parts[0],
        "exp_id": parts[1],
        "task": parts[2],
        "prompt_type": parts[3],
        "prompt_variant": parts[4],
        "question_mode": parts[5],
        "provider": parts[6],
        "file": report_path.name,
    }

    matches = ROW_RE.findall(text)
    for label, _, _, f1, _ in matches:
        row[f"{label}_f1"] = float(f1)

    if len(matches) != 3:
        print(f"⚠️ Missing label rows in {report_path.name}")

    rows.append(row)

df = pd.DataFrame(rows)

if df.empty:
    raise RuntimeError("No reports were parsed. Check filenames or regex.")

# Order columns nicely
df = df[
    [
        "model",
        "provider",
        "prompt_type",
        "prompt_variant",
        "question_mode",
        "Ambivalent_f1",
        "Clear Non-Reply_f1",
        "Clear Reply_f1",
        "file",
    ]
]

df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved summary to {OUTPUT_CSV}")
