from pathlib import Path
import pandas as pd

PREDICTIONS_DIR = Path(__file__).resolve().parents[1] / "results" / "predictions"

LABEL_MAP = {
    "Clear Reply": ["clear reply"],
    "Ambivalent": ["ambivalent", "ambivalent reply", "label: ambivalent reply"],
    "Clear Non-Reply": ["clear non-reply", "label: clear non-reply"]
}
VALID_LABELS = {v for values in LABEL_MAP.values() for v in values}
REVERSED_LABEL_MAP = {v: k for k, vals in LABEL_MAP.items() for v in vals}
KEYWORDS = ["ambivalent", "clear non-reply", "clear reply"]

def manual_select_label(raw_text: str, file_name: str) -> str:
    """Ask the user to manually choose a label when the prediction can't be auto-mapped."""

    CHOICE_MAP = {
    "1": "Clear Reply",
    "2": "Ambivalent",
    "3": "Clear Non-Reply",
    "4": "Invalid Output"
    }

    print("\n--- Manual Check ---")
    print(f"File: {file_name}")
    print(f"\nPrediction: {raw_text}\n")
    print("1) Clear Reply")
    print("2) Ambivalent")
    print("3) Clear Non-Reply")
    print("4) Invalid Output")

    while True:
        choice = input("Choose label [1-4]: ").strip()
        if choice in CHOICE_MAP:
            label = CHOICE_MAP[choice]
            print(f"✅ Selected: {label}\n")
            return label
        print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")


def main():
    files = [f for f in PREDICTIONS_DIR.glob("*.csv") if not f.name.endswith("_VALIDATED.csv")]
    for f in files:
        df = pd.read_csv(f)
        validate_col = f.stem

        if validate_col not in df.columns:
            print(f"[WARN] {validate_col} not found in {f.name}")
            continue

        predictions = df[validate_col].astype(str).str.lower().str.strip()
        model_predictions = []
        
        for prediction in predictions:
            if prediction in VALID_LABELS:
                label = REVERSED_LABEL_MAP[prediction]
            else:
                matches = [k for k in KEYWORDS if k in prediction]
                if len(matches) == 1:  # none or more than one keyword found
                    if "ambivalent" in matches[0]:
                        label = "Ambivalent"
                    elif "clear non-reply" in matches[0]:
                        label = "Clear Non-Reply"
                    elif "clear reply" in matches[0]:
                        label = "Clear Reply"
                else:
                    prediction = manual_select_label(prediction, f)
            model_predictions.append(label)

        df["model_prediction"] = model_predictions
        validated_path = f.with_name(f.stem + "_VALIDATED.csv")
        df.to_csv(validated_path, index=False)
        f.unlink()
        print(f"[OK] {f.name} → {validated_path.name} (old file deleted)")

if __name__ == "__main__":
    main()