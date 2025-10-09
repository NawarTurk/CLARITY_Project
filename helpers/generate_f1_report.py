import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, f1_score

PREDICTION_DIR = Path(__file__).resolve().parents[1] / "results" / "predictions"
TARGET_COLUMN = "clarity_label"
MODEL_PREDICTION_COLUMN = "model_prediction"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "eval_logs" / "detailed"

def main():
    files = [f for f in PREDICTION_DIR.glob("*_VALIDATED.csv")]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    for f in files:
        df = pd.read_csv(f)
        y_true = df[TARGET_COLUMN].astype(str).str.strip().tolist()
        y_pred = df[MODEL_PREDICTION_COLUMN].astype(str).str.strip().tolist()
        label_order = sorted(set(y_true))

        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_micro = f1_score(y_true, y_pred, average="micro")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")
        accuracy = accuracy_score(y_true, y_pred)

        report = classification_report(
            y_true,
            y_pred,
            labels=label_order,
            digits=3,
            zero_division=0,
        )

        report_lines = (
            f"File: {f.name}\n"
            f"F1 Macro: {f1_macro:.3f}\n"
            f"F1 Micro: {f1_micro:.3f}\n"
            f"F1 Weighted: {f1_weighted:.3f}\n"
            f"Accuracy: {accuracy:.3f}\n\n"
            f"Classification Report:\n{report}\n"
        )

        out_path = OUTPUT_DIR / f"{f.stem}_f1Report.txt"
        out_path.write_text(report_lines)
        print(f"Saved report to {out_path}")
        count += 1
    print(f'{count} reports were created.')

if __name__ == "__main__":
    main()
