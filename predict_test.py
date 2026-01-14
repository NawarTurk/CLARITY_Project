import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent
BEST_MODELS_PATH = ROOT / "config" / "best_models.yaml"

STAGE_ORDER = ["stage1", "stage2", "stage3", "stage4", "stage5"]

STAGE_SCRIPTS = {
    "stage1": ROOT / "models" / "encoders" / "s1_encoder_adaptation" / "predict.py",
    "stage2": ROOT / "models" / "encoders" / "s2_representation_classification" / "stage2_predict.py",
    "stage3": ROOT / "models" / "encoders" / "s3_Loss_and_regularization" / "stage3_predict.py",
    "stage4": ROOT / "models" / "encoders" / "s4_data_augmentation" / "stage4_predict.py",
}

STAGE_ARG = {
    "stage1": "--model_name",
    "stage2": "--model_dir",
    "stage3": "--model_dir",
    "stage4": "--model_dir",
}

ENCODER_PRED_DIR = {
    "stage1": ROOT / "results" / "predictions" / "encoder" / "stage1",
    "stage2": ROOT / "results" / "predictions" / "encoder" / "stage2",
    "stage3": ROOT / "results" / "predictions" / "encoder" / "stage3",
    "stage4": ROOT / "results" / "predictions" / "encoder" / "stage4",
}

OUT_ROOT = ROOT / "results" / "codebench_evaluation_prediction" / "encoders"
ALLOWED_LABELS = {"Ambivalent", "Clear Reply", "Clear Non-Reply"}
EXPECTED_TEST_COUNT = 308


def _load_best_models() -> dict:
    if not BEST_MODELS_PATH.exists():
        raise FileNotFoundError(f"Missing config file: {BEST_MODELS_PATH}")
    with BEST_MODELS_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML structure in {BEST_MODELS_PATH}; expected a mapping.")
    return data


def _resolve_stages(requested: Iterable[str], config: dict) -> list[str]:
    requested = [s.strip().lower() for s in requested if s]
    if "all" in requested:
        return [stage for stage in STAGE_ORDER if stage in config]
    ordered = []
    seen = set()
    for stage in STAGE_ORDER:
        if stage in requested and stage not in seen:
            ordered.append(stage)
            seen.add(stage)
    return ordered


def _infer_stage_from_checkpoint(checkpoint: Optional[Path]) -> Optional[str]:
    if not checkpoint:
        return None
    parts = set(checkpoint.parts)
    if "stage1_trained_models" in parts or "s1_encoder_adaptation" in parts:
        return "stage1"
    if "stage2_trained_models" in parts or "s2_representation_classification" in parts:
        return "stage2"
    if "stage3_trained_models" in parts or "s3_Loss_and_regularization" in parts:
        return "stage3"
    if "stage4_trained_models" in parts or "s4_data_augmentation" in parts:
        return "stage4"
    return None


def _run_predictor(stage: str, model_arg: str, dataset_path: Path) -> None:
    script = STAGE_SCRIPTS.get(stage)
    if script is None:
        raise ValueError(f"No predictor script registered for stage '{stage}'.")
    arg_name = STAGE_ARG[stage]
    cmd = [sys.executable, str(script), arg_name, model_arg, "--dataset", str(dataset_path)]
    subprocess.run(cmd, check=True)


def _write_codabench_file(out_dir: Path, df: pd.DataFrame, *, expected_len: Optional[int]) -> None:
    label_col = "predicted_label"
    if label_col not in df.columns:
        if "prediction" in df.columns:
            label_col = "prediction"
        else:
            raise KeyError("Prediction column not found (expected 'predicted_label' or 'prediction').")
    labels = df[label_col].tolist()
    if expected_len is not None and len(labels) != expected_len:
        raise ValueError(f"Expected {expected_len} predictions, got {len(labels)}.")
    bad = set(labels) - ALLOWED_LABELS
    if bad:
        raise ValueError(f"Invalid labels found: {sorted(bad)}.")
    codabench_file = out_dir / "prediction"
    with codabench_file.open("w", encoding="utf-8") as f:
        for label in labels:
            f.write(f"{label}\n")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate test predictions for best models.")
    parser.add_argument(
        "--stage",
        action="append",
        required=True,
        help="Stage to run: stage1, stage2, stage3, stage4, stage5, or all.",
    )
    parser.add_argument(
        "--split",
        choices=("dev", "test"),
        default="dev",
        help="Dataset split to use from config/best_models.yaml (default: dev).",
    )
    args = parser.parse_args(argv)

    config = _load_best_models()
    datasets = config.get("datasets", {}) if isinstance(config, dict) else {}
    if not isinstance(datasets, dict):
        raise ValueError(f"Invalid datasets block in {BEST_MODELS_PATH}; expected a mapping.")
    dataset_ref = datasets.get(args.split)
    if not dataset_ref:
        raise ValueError(
            f"Missing datasets.{args.split} in {BEST_MODELS_PATH}. "
            "Add the dataset path there to proceed."
        )
    dataset_path = Path(dataset_ref)
    if not dataset_path.is_absolute():
        dataset_path = (ROOT / dataset_path).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path} (from datasets.{args.split} in {BEST_MODELS_PATH})."
        )

    stages = _resolve_stages(args.stage, config)
    if not stages:
        print("No valid stages requested. Use --stage stage1|stage2|stage3|stage4|stage5|all.", file=sys.stderr)
        return 1

    expected_len = EXPECTED_TEST_COUNT if args.split == "test" else None

    for stage in stages:
        models = config.get(stage)
        if not models:
            print(f"[WARN] No models defined for {stage} in {BEST_MODELS_PATH}")
            continue
        if not isinstance(models, list):
            raise ValueError(f"Invalid list for {stage} in {BEST_MODELS_PATH}; expected a list.")

        for entry in models:
            if not isinstance(entry, dict):
                raise ValueError(f"Invalid model entry under {stage}; expected a mapping.")
            model_name = entry.get("name") or entry.get("model")
            if not model_name:
                raise ValueError(f"Missing model name under {stage}.")
            checkpoint = entry.get("checkpoint")

            checkpoint_path = Path(checkpoint) if checkpoint else None
            if checkpoint_path and not checkpoint_path.is_absolute():
                checkpoint_path = (ROOT / checkpoint_path).resolve()
            if checkpoint_path and not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found for {stage}:{model_name}: {checkpoint_path}")

            predictor_stage = _infer_stage_from_checkpoint(checkpoint_path) or (
                stage if stage in STAGE_SCRIPTS else None
            )
            if predictor_stage not in STAGE_SCRIPTS:
                print(
                    f"[WARN] No predictor script registered for stage '{predictor_stage or stage}'. "
                    f"Skipping {stage}:{model_name}."
                )
                continue

            predictor_slug = checkpoint_path.name if checkpoint_path else model_name
            if checkpoint_path and not checkpoint_path.is_dir():
                raise FileNotFoundError(
                    f"Checkpoint path for {stage}:{model_name} must be a directory: {checkpoint_path}"
                )

            print(f"[{stage}] Running {predictor_stage} predictor for {model_name}...")
            predictor_arg = str(checkpoint_path) if checkpoint_path else model_name
            _run_predictor(predictor_stage, predictor_arg, dataset_path)

            pred_dir = ENCODER_PRED_DIR[predictor_stage]
            pred_path = pred_dir / f"{predictor_slug}_predictions.csv"
            alt_pred_path = pred_dir / f"{model_name}_predictions.csv"
            if not pred_path.exists() and alt_pred_path.exists():
                pred_path = alt_pred_path
            if not pred_path.exists():
                available = []
                if pred_dir.exists():
                    available = sorted(p.name for p in pred_dir.glob("*_predictions.csv"))
                found = ", ".join(available) if available else "none"
                raise FileNotFoundError(
                    f"Expected predictions at {pred_path} (or {alt_pred_path}); found: {found}"
                )

            df = pd.read_csv(pred_path)
            out_dir = OUT_ROOT / stage / model_name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_csv = out_dir / f"{model_name}.csv"
            df.to_csv(out_csv, index=False)
            _write_codabench_file(out_dir, df, expected_len=expected_len)
            print(f"[DONE] Wrote {out_csv} and {out_dir / 'prediction'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
