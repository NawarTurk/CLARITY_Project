import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, f1_score

PREDICTION_DIR = Path(__file__).resolve().parents[1] / "results" / "predictions"
TARGET_COLUMN = "clarity_label"
MODEL_PREDICTION_COLUMN = "model_prediction"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "eval_logs" / "detailed"
GLOBAL_REPORT_DIR = Path(__file__).resolve().parents[1] / "results" / "eval_logs" / "global"

MODEL_FAMILY_INFO = {
    # ---- LLaMA family ----
    "llama-3.1-nemotron-253b": ("LLaMA", 253),
    "llama-3.1-405b-Instruct": ("LLaMA", 405),
    "llama-3.3-70b-Instruct": ("LLaMA", 70),

    # ---- Qwen family ----
    "qwen3-coder-480b-Instruct": ("Qwen", 480),
    "qwen3-235b-instruct": ("Qwen", 235),
    "qwen3-80b-instruct": ("Qwen", 80),
    "qwen3-32b-instruct": ("Qwen", 32),

    # ---- Mixtral family ----
    "mixtral-8x22b-Instruct": ("Mixtral", 176),
    "mixtral-8x7b-Instruct": ("Mixtral", 56),

    # ---- GPT family ----
    "gpt-5": ("GPT", None),  # parameter count undisclosed
}

def save_global_report(global_report):
    """Save the global summary CSV."""
    if not global_report:
        print("⚠️ No data to save in global report.")
        return

    df_global = pd.DataFrame(global_report)
    GLOBAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    global_path = GLOBAL_REPORT_DIR / "global_f1_summary.csv"
    df_global.to_csv(global_path, index=False)
    print(f"\n✅ Global summary saved to {global_path}")

def main():
    files = [f for f in PREDICTION_DIR.glob("*_VALIDATED.csv")]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GLOBAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    global_report = []
    for f in files:
        df = pd.read_csv(f)
        y_true = df[TARGET_COLUMN].astype(str).str.strip().tolist()
        y_pred = df[MODEL_PREDICTION_COLUMN].astype(str).str.strip().tolist()
        label_order = sorted(set(y_true) | set(y_pred)) 

        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_micro = f1_score(y_true, y_pred, average="micro")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")
        accuracy = accuracy_score(y_true, y_pred)

        parts = f.stem.split('_')
        model, prompt_id, task_id, prompt_technique, prompt_sub_technique, question_columns, provider, validated = parts
        model_family, param_count = MODEL_FAMILY_INFO[model]
        global_report.append({
            'llm_model': model,
            'model_family': model_family,
            'param_count': param_count,
            'prompt_id': prompt_id,
            'prompt_technique': prompt_technique,
            'prompt_sub_technique': prompt_sub_technique,
            'question_columns': question_columns,
            'provider': provider,
            'validated': validated,
            'task_id': task_id,
            "file_name": f.stem,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "f1_weighted": f1_weighted,
            "accuracy": accuracy,
        })

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
        print(f"Saved {f.stem} report")
        count += 1

    print(f'✅ {count} reports were created.')
    save_global_report(global_report)

if __name__ == "__main__":
    main()
