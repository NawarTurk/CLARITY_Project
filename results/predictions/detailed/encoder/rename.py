from pathlib import Path

PRED_DIR = Path(".")  # current directory

TAG = "_originalData_truncHead_lossCE"
PRED_SUFFIX = "_predictions.csv"

for f in PRED_DIR.glob(f"*{PRED_SUFFIX}"):
    name = f.name

    # skip if already tagged
    if "originalData" in name or "truncHead" in name or "lossCE" in name:
        print(f"[SKIP] {name}")
        continue

    new_name = name.replace(
        PRED_SUFFIX,
        f"{TAG}{PRED_SUFFIX}"
    )

    f.rename(f.with_name(new_name))
    print(f"[RENAMED] {name} -> {new_name}")
