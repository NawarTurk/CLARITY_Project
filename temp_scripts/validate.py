#!/usr/bin/env python3
"""Validate prediction CSV files and enforce label consistency."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path("results/predictions")
DEFAULT_COLUMN = "prediction"
CANONICAL_LABELS = {
    "clear reply": "Clear Reply",
    "ambivalent": "Ambivalent",
    "clear non-reply": "Clear Non-Reply",
    "invalid output": "Invalid Output",
}
EXPECTED_LABELS = [
    "Clear Reply",
    "Ambivalent",
    "Clear Non-Reply",
]
CHOICE_MAP = {
    "1": EXPECTED_LABELS[0],
    "2": EXPECTED_LABELS[1],
    "3": EXPECTED_LABELS[2],
    "4": "Invalid Output",
}
ALIAS_MAP = {
    "clear answer": EXPECTED_LABELS[0],
    "clear-answer": EXPECTED_LABELS[0],
    "ambivalent reply": EXPECTED_LABELS[1],
    "ambivalent answer": EXPECTED_LABELS[1],
    "ambivalent-answer": EXPECTED_LABELS[1],
    "clear non answer": EXPECTED_LABELS[2],
    "clear non-answer": EXPECTED_LABELS[2],
    "clear non reply": EXPECTED_LABELS[2],
}


def manual_select_label(raw_text: str, file_path: Path, row_identifier: str) -> str:
    """Prompt the user when a prediction value does not match the expected labels."""
    print("\n--- Manual validation required ---")
    print(f"File: {file_path}")
    print(f"Row: {row_identifier}")
    print("Raw prediction:")
    print(raw_text if raw_text else "<empty>")
    print("Options: 1) Clear Reply  2) Ambivalent  3) Clear Non-Reply  4) Invalid Output")

    while True:
        try:
            choice = input("Enter choice [1-4]: ").strip()
        except EOFError:
            print("\nInput interrupted. Aborting.")
            sys.exit(1)

        if choice in CHOICE_MAP:
            selected = CHOICE_MAP[choice]
            print(f"Selected: {selected}\n")
            return selected

        print("Invalid selection. Please enter a number between 1 and 4.")



def candidate_columns(csv_path: Path) -> list[str]:
    stem = csv_path.stem
    candidates = {DEFAULT_COLUMN, stem}
    for prefix in ("test_", "train_", "val_", "valid_", "validation_"):
        if stem.startswith(prefix):
            candidates.add(stem[len(prefix) :])
    return [col for col in candidates if col]



def resolve_prediction_column(df: pd.DataFrame, csv_path: Path) -> str | None:
    for column in candidate_columns(csv_path):
        if column in df.columns:
            return column

    normalized_labels = {label.lower() for label in EXPECTED_LABELS}
    normalized_labels.update(ALIAS_MAP.keys())
    normalized_labels.add("invalid output")
    for column in reversed(df.columns):
        series = df[column].dropna().astype(str).str.strip().str.lower()
        if series.empty:
            continue
        sample = series.sample(min(len(series), 20), random_state=42)
        if sample.isin(normalized_labels).any():
            return column

    return None



def extract_label(raw_text: str) -> str | None:
    text_lower = raw_text.lower()

    if "invalid output" in text_lower:
        return CANONICAL_LABELS["invalid output"]

    matched_labels: list[str] = []

    for label in EXPECTED_LABELS:
        if label.lower() in text_lower:
            matched_labels.append(label)

    for alias, canonical in ALIAS_MAP.items():
        if alias in text_lower and canonical not in matched_labels:
            matched_labels.append(canonical)

    if len(matched_labels) == 1:
        return matched_labels[0]

    return None



def validate_file(csv_path: Path) -> None:
    """Normalize predictions in a single CSV file."""
    print(f"Processing {csv_path}")
    df = pd.read_csv(csv_path)

    column_name = resolve_prediction_column(df, csv_path)
    if not column_name:
        print("  Skipped: unable to determine prediction column.")
        return

    standard_column = f"standard_{column_name}"
    normalized_values: list[str] = []

    for idx, value in df[column_name].items():
        raw_text = "" if pd.isna(value) else str(value)
        label = extract_label(raw_text)

        if label is None:
            label = manual_select_label(raw_text, csv_path, str(idx))

        normalized_values.append(label)

    df[standard_column] = normalized_values
    df.to_csv(csv_path, index=False)
    print(f"  Updated {csv_path}\n")



def main() -> None:
    if not BASE_DIR.exists():
        print(f"No prediction files found. Expected directory: {BASE_DIR}")
        return

    csv_files = sorted(BASE_DIR.rglob("*.csv"))
    if not csv_files:
        print(f"No CSV files located under {BASE_DIR}")
        return

    for csv_path in csv_files:
        try:
            validate_file(csv_path)
        except KeyboardInterrupt:
            print("\nValidation interrupted by user.")
            sys.exit(1)


if __name__ == "__main__":
    main()
