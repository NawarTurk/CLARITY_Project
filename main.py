import argparse
import subprocess
import sys
from pathlib import Path
import yaml
from helpers import generate_f1_report
from helpers import validate

TRAINING_SCRIPTS = ("train_full", "train_freeze", "train_lora", "train_adapter")

def run_stage2_from_config():
    project_root = Path(__file__).resolve().parent
    config_path = project_root / "config" / "stage2.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    truncations = cfg.get("truncation_modes", [])
    heads = cfg.get("classification_heads", [])

    for run in cfg["runs"]:
        script = (
            project_root
            / "models"
            / "encoders"
            / "s2_representation_classification"
            / run["script"]
        )

        if not script.exists():
            raise FileNotFoundError(f"Script not found: {script}")

        for trunc in truncations:
            for head in heads:

                if trunc == "head" and head == "default":
                    continue

                cmd = [
                    sys.executable,
                    str(script),
                    "--model_name", run["model"],
                    "--param_mode", "fixed",
                    "--head_type", head,
                    "--truncation", trunc,
                ]

                # optional params
                if "unfreeze_ratio" in run:
                    cmd += ["--unfreeze_ratio", str(run["unfreeze_ratio"])]

                print("\n[Stage 2] Running:", " ".join(cmd))
                subprocess.run(cmd, check=True)

def run_stage3_from_config():
    project_root = Path(__file__).resolve().parent
    config_path = project_root / "config" / "stage3.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    loss_fns = cfg.get("loss_functions", [])
    dropouts = cfg.get("dropout", [])

    for run in cfg["runs"]:
        script = (
            project_root
            / "models"
            / "encoders"
            / "s3_Loss_and_regularization"
            / run["script"]
        )

        if not script.exists():
            raise FileNotFoundError(f"Script not found: {script}")

        trunc = run["truncation"]
        head = run["classification_head"]

        for loss_fn in loss_fns:
            for use_dropout in dropouts:

                 # Reason: WCE and dropout 0.1 was what was used in Stage 2
                if loss_fn == "WCE" and use_dropout == 0.1:
                    continue

                cmd = [
                    sys.executable,
                    str(script),
                    "--model_name", run["model"],
                    "--truncation", trunc,
                    "--head_type", head,
                    "--loss_fn", loss_fn,
                    "--dropout", str(use_dropout),
                ]

                # optional params
                if "unfreeze_ratio" in run:
                    cmd += ["--unfreeze_ratio", str(run["unfreeze_ratio"])]

                if "lora_rank" in run:
                    cmd += ["--lora_rank", str(run["lora_rank"])]

                if "lora_top_layers" in run:
                    cmd += ["--lora_top_layers", str(run["lora_top_layers"])]

                print("\n[Stage 3] Running:", " ".join(cmd))
                subprocess.run(cmd, check=True)


def run_stage4_from_config():
    project_root = Path(__file__).resolve().parent
    config_path = project_root / "config" / "stage4.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    truncations = cfg.get("truncation_modes")
    heads = cfg.get("classification_heads")
    datasets = cfg.get("dataset", ["original"])
    input_modes = cfg.get("input_mode", ["atomic"])

    for run in cfg["runs"]:
        script = (
            project_root
            / "models"
            / "encoders"
            / "s4_data_augmentation"
            / run["script"]
        )

        if not script.exists():
            raise FileNotFoundError(f"Script not found: {script}")

        if truncations and heads:
            trunc_iter = truncations
            head_iter = heads
        else:
            trunc_iter = [run.get("truncation")]
            head_iter = [run.get("classification_head") or run.get("head_type")]

        for trunc in trunc_iter:
            for head in head_iter:
                if trunc is None or head is None:
                    raise ValueError(
                        f"Missing truncation/head_type for stage4 run: {run}"
                    )

                for dataset in datasets:
                    for input_mode in input_modes:
                        if dataset == "original" and input_mode == "atomic":
                            continue

                        cmd = [
                            sys.executable,
                            str(script),
                            "--model_name", run["model"],
                            "--param_mode", "fixed",
                            "--head_type", head,
                            "--truncation", trunc,
                            "--dataset", dataset,
                            "--input_mode", input_mode,
                        ]

                        if "unfreeze_ratio" in run:
                            cmd += ["--unfreeze_ratio", str(run["unfreeze_ratio"])]

                        print("\n[Stage 4] Running:", " ".join(cmd))
                        subprocess.run(cmd, check=True)


def run_encoder_training(train_script: str, model_name: str, param_mode: str) -> None:
    """Invoke one of the encoder training pipelines."""
    project_root = Path(__file__).resolve().parent
    script_path = project_root / "models" / "encoders" / f"{train_script}.py"
    if not script_path.exists():
        raise FileNotFoundError(
            f"Could not locate training script for '{train_script}' at {script_path}"
        )

    cmd = [
        sys.executable,
        str(script_path),
        "--model_name",
        model_name,
        "--param_mode",
        param_mode,
    ]
    print(f"Launching encoder training via {script_path} ...")
    subprocess.run(cmd, check=True)


def run_encoder_prediction(model_folder: str) -> None:
    """Invoke encoder prediction for a trained model folder."""
    project_root = Path(__file__).resolve().parent
    script_path = project_root / "models" / "encoders" / "predict.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Could not locate prediction script at {script_path}")

    cmd = [
        sys.executable,
        str(script_path),
        "--model_name",
        model_folder,
    ]
    print(f"Running encoder prediction via {script_path} for model '{model_folder}' ...")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Clarity Project Controller")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation on model predictions",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Generate F1 score reports",
    )

    parser.add_argument(
        "--train_",
        choices=TRAINING_SCRIPTS,
        help="Encoder training script to run",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Base Hugging Face model identifier (e.g., mbert, roberta, xlmr)",
    )
    parser.add_argument(
        "--param_mode",
        choices=("fixed", "hps"),
        help="Parameter mode used during training",
    )
    parser.add_argument(
        "--predict",
        type=str,
        help="Run encoder prediction for a trained model folder name",
    )
    parser.add_argument(
        "--train-stage-2",
        dest="train_stage_2",
        action="store_true",
    )
    parser.add_argument(
        "--predict-stage-2",
        action="store_true",
        help="Run Stage 2 predictions"
    )

    parser.add_argument(
        "--train-stage-3",
        dest="train_stage_3",
        action="store_true",
    )

    parser.add_argument(
        "--predict-stage-3",
        action="store_true",
        help="Run Stage 3 predictions"
    )
    parser.add_argument(
        "--train-stage-4",
        dest="train_stage_4",
        action="store_true",
    )
    parser.add_argument(
        "--predict-stage-4",
        action="store_true",
        help="Run Stage 4 predictions"
    )

    args = parser.parse_args()

    if args.predict:
        if args.train_:
            parser.error("--predict cannot be combined with --train_.")
        run_encoder_prediction(args.predict)
        return

    if args.train_:
        if not args.model_name:
            parser.error("--model_name is required when --train_ is provided.")
        if not args.param_mode:
            parser.error("--param_mode is required when --train_ is provided.")
        run_encoder_training(args.train_, args.model_name, args.param_mode)
        return

    if args.model_name or args.param_mode:
        parser.error("--model_name and --param_mode can only be used with --train_.")

    if args.validate:
        print("Running Validation")
        validate.main()
    elif args.evaluate:
        print("Generating F1 Score Reports")
        generate_f1_report.main()


    if args.train_stage_2:
        run_stage2_from_config()
        return
    
    if args.train_stage_3:
        run_stage3_from_config()
        return

    if args.train_stage_4:
        run_stage4_from_config()
        return
    
    if args.predict_stage_2:
        project_root = Path(__file__).resolve().parent
        script = project_root / "models" / "encoders" / "s2_representation_classification" / "stage2_predict.py"
        print("[Stage 2] Running prediction script:", script)
        # By default, run predictions for all Stage 2 trained models.
        subprocess.run([sys.executable, str(script), "--model_dir", "all"], check=True)
        return
    
    if args.predict_stage_3:
        project_root = Path(__file__).resolve().parent
        script = project_root / "models" / "encoders" / "s3_Loss_and_regularization" / "stage3_predict.py"
        print("[Stage 3] Running prediction script:", script)
        # By default, run predictions for all Stage 3 trained models.
        subprocess.run([sys.executable, str(script), "--model_dir", "all"], check=True)
        return

    if args.predict_stage_4:
        project_root = Path(__file__).resolve().parent
        script = project_root / "models" / "encoders" / "s4_data_augmentation" / "stage4_predict.py"
        print("[Stage 4] Running prediction script:", script)
        # By default, run predictions for all Stage 4 trained models.
        subprocess.run([sys.executable, str(script), "--model_dir", "all"], check=True)
        return


if __name__ == "__main__":
    main()
