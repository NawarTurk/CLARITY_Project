import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, f1_score

PREDICTION_DIR = Path(__file__).resolve().parents[1] / "results" / "predictions" / "prompt"
TARGET_COLUMN = "clarity_label"
MODEL_PREDICTION_COLUMN = "model_prediction"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "eval_logs" / "detailed" / "prompt"
GLOBAL_REPORT_DIR = Path(__file__).resolve().parents[1] / "results" / "eval_logs" / "global"

ENCODER_PREDICTION_DIR = Path(__file__).resolve().parents[1] / "results" / "predictions" / "detailed" / "encoder"
ENCODER_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "eval_logs" / "detailed" / "encoder"
ENCODER_TARGET_COLUMN = "clarity_label"
ENCODER_PRED_COLUMN = "predicted_label"

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

def save_global_report(global_report, filename="prompt_global_f1_summary.csv"):
    """Save the global summary CSV."""
    if not global_report:
        print("⚠️ No data to save in global report.")
        return

    df_global = pd.DataFrame(global_report)
    GLOBAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    global_path = GLOBAL_REPORT_DIR / filename
    df_global.to_csv(global_path, index=False)
    print(f"\n✅ Global summary saved to {global_path}")


def process_encoder_predictions():
    files = sorted(ENCODER_PREDICTION_DIR.glob("*_predictions.csv"))
    ENCODER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GLOBAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    encoder_global_report = []
    count = 0
    for f in files:
        df = pd.read_csv(f)

        y_true = df[ENCODER_TARGET_COLUMN].astype(str).str.strip().tolist()
        y_pred = df[ENCODER_PRED_COLUMN].astype(str).str.strip().tolist()
        label_order = sorted(set(y_true) | set(y_pred))

        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_micro = f1_score(y_true, y_pred, average="micro")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")
        accuracy = accuracy_score(y_true, y_pred)

        model_name = f.stem
        suffix = "_predictions"
        if model_name.endswith(suffix):
            model_name = model_name[: -len(suffix)]
        parts = model_name.split("_")
        if len(parts) < 6:
            print(f"[Warn] Skipping {f.name}: unable to parse model metadata.")
            continue
        task_id, arch, lang, size, tune, param_mode = parts[:6]
        head_type = parts[6] if len(parts) >= 7 else "defaultHead"
        report_basename = model_name if model_name.endswith(head_type) else f"{model_name}_{head_type}"

        report = classification_report(
            y_true,
            y_pred,
            labels=label_order,
            digits=3,
            zero_division=0,
        )

        report_lines = (
            f"Model: {model_name}\n"
            f"Head Type: {head_type}\n"
            f"F1 Macro: {f1_macro:.3f}\n"
            f"F1 Micro: {f1_micro:.3f}\n"
            f"F1 Weighted: {f1_weighted:.3f}\n"
            f"Accuracy: {accuracy:.3f}\n\n"
            f"Classification Report:\n{report}\n"
        )

        out_path = ENCODER_OUTPUT_DIR / f"{report_basename}_f1Report.txt"
        out_path.write_text(report_lines)
        print(f"Saved encoder report for {model_name}")
        count += 1

        encoder_global_report.append({
            "model_name": model_name,
            "task_id": task_id,
            "arch": arch,
            "lang": lang,
            "size": size,
            "tune": tune,
            "param_mode": param_mode,
            "head_type": head_type,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "f1_weighted": f1_weighted,
            "accuracy": accuracy,
        })

    print(f"✅ {count} encoder reports were created.")
    if encoder_global_report:
        encoder_global_report = sorted(
            encoder_global_report,
            key=lambda r: (r["f1_macro"], r["accuracy"]),
            reverse=True,
        )
        save_global_report(encoder_global_report, filename="encoder_f1_global_summary.csv")

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
    process_encoder_predictions()

if __name__ == "__main__":
    main()
