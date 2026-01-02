import re
from pathlib import Path
import pandas as pd

REPORT_DIR = Path(".")
OUTPUT_CSV = "_f1_per_label_summary.csv"

LABELS = [
    "Ambivalent",
    "Clear Non-Reply",
    "Clear Reply",
]

ROW_RE = re.compile(
    r"^\s*(Ambivalent|Clear Non-Reply|Clear Reply)\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)",
    re.MULTILINE
)

rows = []

files = list(REPORT_DIR.glob("*_f1Report.txt"))
print(f"Found {len(files)} encoder reports")

for report_path in files:
    text = report_path.read_text()
    parts = report_path.stem.split("_")

    # Expected minimum structure
    # t1, bert, en, base, lora16, fixed, head, data, trunc, loss, f1Report
    if len(parts) < 11:
        print(f"⚠️ Skipping malformed filename: {report_path.name}")
        continue

    row = {
        "task": parts[0],
        "arch": parts[1],
        "lang": parts[2],
        "size": parts[3],
        "tuning": parts[4],
        "param_mode": parts[5],
        "head": parts[6],
        "data": parts[7],
        "truncation": parts[8],
        "loss": parts[9],
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
    raise RuntimeError("No encoder reports parsed")

df = df[
    [
        "task",
        "arch",
        "lang",
        "size",
        "tuning",
        "param_mode",
        "head",
        "data",
        "truncation",
        "loss",
        "Ambivalent_f1",
        "Clear Non-Reply_f1",
        "Clear Reply_f1",
        "file",
    ]
]

df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved → {OUTPUT_CSV}")
