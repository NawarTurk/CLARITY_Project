import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, f1_score

PREDICTION_DIR = Path(__file__).resolve().parents[1] / "results" / "predictions" / "prompt"
TARGET_COLUMN = "clarity_label"
MODEL_PREDICTION_COLUMN = "model_prediction"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "eval_logs" / "detailed" / "prompt"
GLOBAL_REPORT_DIR = Path(__file__).resolve().parents[1] / "results" 

ENCODER_PREDICTION_DIR = Path(__file__).resolve().parents[1] / "results" / "predictions" / "detailed" / "encoder"
ENCODER_STAGE1_PREDICTION_DIR = Path(__file__).resolve().parents[1] / "results" / "predictions" / "encoder" / "stage1"
ENCODER_STAGE2_PREDICTION_DIR = Path(__file__).resolve().parents[1] / "results" / "predictions" / "encoder" / "stage2"
ENCODER_STAGE3_PREDICTION_DIR = Path(__file__).resolve().parents[1] / "results" / "predictions" / "encoder" / "stage3"
STAGE2_ENCODER_PREDICTION_DIR = Path(__file__).resolve().parents[1] / "results" / "predictions" / "stage2_encoder"
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


def _parse_encoder_experiment(experiment_name: str):
    parts = experiment_name.split("_")

    # Unified 11-token format:
    # task_id_arch_lang_size_tune_param_mode_head_type_truncation_loss_type_data_variant_input_mode
    if len(parts) >= 11:
        (
            task_id,
            arch,
            lang,
            size,
            tune,
            param_mode,
            head_type,
            truncation,     # head / head-tail
            loss_type,      # WCE / FOCAL-Dropout10 / ...
            data_variant,   # originalData / augmentedData
            input_mode,     # atomic / enriched
        ) = parts[:11]
        return task_id, arch, lang, size, tune, param_mode, head_type, truncation, data_variant, loss_type, input_mode

    return None



def process_encoder_predictions():
    # Collect encoder prediction files from the encoder stage folders (stage1, stage2, ...)
    # and keep support for legacy, non-staged encoder prediction folders.
    files = []

    encoder_root = Path(__file__).resolve().parents[1] / "results" / "predictions" / "encoder"
    if encoder_root.exists():
        for f in sorted(encoder_root.rglob("*_predictions.csv")):
            stage_name = f.parent.name
            files.append((f, stage_name))


    ENCODER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GLOBAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    encoder_global_report = []
    count = 0
    for f, stage_name in files:
        df = pd.read_csv(f)

        y_true = df[ENCODER_TARGET_COLUMN].astype(str).str.strip().tolist()
        y_pred = df[ENCODER_PRED_COLUMN].astype(str).str.strip().tolist()
        label_order = sorted(set(y_true) | set(y_pred))

        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_micro = f1_score(y_true, y_pred, average="micro")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")
        accuracy = accuracy_score(y_true, y_pred)
        experiment_name = f.stem
        suffix = "_predictions"
        if experiment_name.endswith(suffix):
            experiment_name = experiment_name[: -len(suffix)]

        # parsed = _parse_encoder_experiment(experiment_name, stage_name)
        parsed = _parse_encoder_experiment(experiment_name)
        if parsed is None:
            print(f"[Warn] Skipping {f.name} | stage={stage_name}: unexpected filename format.")
            continue
        (
            task_id,
            arch,
            lang,
            size,
            tune,
            param_mode,
            head_type,
            truncation,
            data_variant,
            loss_type,
            input_mode
        ) = parsed

        model_name = f"{arch}-{size}"

        report_basename = experiment_name 

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

   
        if stage_name:
            out_dir = ENCODER_OUTPUT_DIR / stage_name
        else:
            out_dir = ENCODER_OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{report_basename}_f1Report.txt"
        out_path.write_text(report_lines)
        print(f"Saved encoder report for {model_name}")
        count += 1

        encoder_global_report.append({
           "stage": stage_name,
            "experiment": experiment_name,
            "task_id": task_id,
            "model_name": model_name,
            "lang": lang,
            "size": size,
            "tune": tune,
            "param_mode": param_mode,
            "head_type": head_type,
            "truncation": truncation,
            "data_variant": data_variant,
            "loss_type": loss_type,
            "input_mode": input_mode,
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
        # label_order = sorted(set(y_true) | set(y_pred))
        INVALID_LABEL = "Invalid Output"
        invalid_rate = sum(p == INVALID_LABEL for p in y_pred) / len(y_pred)

        
        OFFICIAL_LABELS = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]
        # f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_macro = f1_score(
            y_true,
            y_pred,
            labels=OFFICIAL_LABELS,
            average="macro",
            zero_division=0,
        )
         
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
            "invalid_rate": invalid_rate,
            "file_name_complete":f.name
        })

        report = classification_report(
            y_true,
            y_pred,
            # labels=label_order,
            labels=OFFICIAL_LABELS,
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
