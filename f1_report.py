from __future__ import annotations
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


DEFAULT_TARGET_COLUMN = "clarity_label"
DEFAULT_PREDICTIONS_DIR = Path("results/predictions")
DEFAULT_OUTPUT_DIR = Path("results/eval_logs/detailed")


def resolve_pred_column(csv_path: Path, columns: list[str]) -> str:
    stem = csv_path.stem
    candidate_stems: list[str] = [stem]

    for suffix in ("-question", "_question"):
        if stem.endswith(suffix):
            candidate_stems.append(stem[: -len(suffix)])

    for candidate in dict.fromkeys(candidate_stems):
        column_name = f"standard_{candidate}"
        if column_name in columns:
            return column_name

    standard_columns = [col for col in columns if col.startswith("standard_")]
    if len(standard_columns) == 1:
        return standard_columns[0]

    if not standard_columns:
        raise ValueError(
            f"Unable to locate a 'standard_' prediction column for {csv_path}."
        )

    raise ValueError(
        f"Multiple 'standard_' columns found in {csv_path}; unable to decide."
    )

def generate_report(
    csv_path: Path,
    target_column: str = DEFAULT_TARGET_COLUMN,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in {csv_path}.")

    pred_column = resolve_pred_column(csv_path, list(df.columns))

    if df[target_column].isna().any():
        raise ValueError(f"Missing labels detected in '{target_column}'.")
    if df[pred_column].isna().any():
        raise ValueError(f"Missing labels detected in '{pred_column}'.")

    y_true = df[target_column].astype(str).str.strip().tolist()
    y_pred = df[pred_column].astype(str).str.strip().tolist()

    label_order = sorted(set(y_true) | set(y_pred))

    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_micro = f1_score(y_true, y_pred, average="micro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    accuracy = accuracy_score(y_true, y_pred)

    clf_report = classification_report(
        y_true,
        y_pred,
        labels=label_order,
        digits=3,
        zero_division=0,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{csv_path.stem}_f1_report.txt"

    report_lines = [
        f"F1 Macro: {f1_macro:.3f}",
        f"F1 Micro: {f1_micro:.3f}",
        f"F1 Weighted: {f1_weighted:.3f}",
        f"Accuracy: {accuracy:.3f}",
        "Classification Report:",
        clf_report,
    ]

    output_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Saved report to {output_path}")
    return output_path


def main() -> None:
    csv_files = sorted(DEFAULT_PREDICTIONS_DIR.rglob("*.csv"))
    if not csv_files:
        print(f"No CSV files found under {DEFAULT_PREDICTIONS_DIR}")
        return

    for csv_path in csv_files:
        try:
            print(f"Processing {csv_path}")
            generate_report(csv_path)
        except Exception as exc:
            print(f"Failed to process {csv_path}: {exc}")


if __name__ == "__main__":
    main()
