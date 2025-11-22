import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

GLOBAL_REPORT_DIR = Path(__file__).resolve().parents[1] / "results" / "eval_logs" / "global"
PLOTS_DIR = Path(__file__).resolve().parents[1] / "results" /  "plots" / "prompt"
ENCODER_PLOTS_DIR = Path(__file__).resolve().parents[1] / "results" / "plots" / "encoder"
ENCODER_GLOBAL_CSV = GLOBAL_REPORT_DIR / "encoder_f1_global_summary.csv"

def main():
    """Plot F1 macro/micro/weighted + accuracy together for each grouping variable, with values on bars."""
    csv_path = GLOBAL_REPORT_DIR / "prompt_global_f1_summary.csv"
    if not csv_path.exists():
        print(f"⚠️ Global summary not found at {csv_path}")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # metrics = ["f1_macro", "f1_micro", "f1_weighted", "accuracy"]
    metrics = ["f1_macro", "f1_weighted"]

    group_columns = ["question_columns", "prompt_technique", "prompt_sub_technique", 
                     "prompt_id", 'llm_model', 'model_family', 'param_count']

    for group_col in group_columns:
        # average over models/providers (optional, for readability)
        grouped = df.groupby(group_col)[metrics].mean().reset_index()

        # reshape for plotting
        long_df = grouped.melt(
            id_vars=[group_col], value_vars=metrics,
            var_name="Metric", value_name="Score"
        )

        plt.figure(figsize=(9, 5))
        ax = sns.barplot(data=long_df, x=group_col, y="Score", hue="Metric", palette="tab10")
        plt.title(f"Comparison of Metrics by {group_col.replace('_', ' ').title()}")
        plt.ylabel("Score")
        plt.xlabel(group_col.replace('_', ' ').title())
        plt.ylim(0, 1.0)
        plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.xticks(rotation=30, ha="right")

        # 🔢 Add values on top of bars
        for p in ax.patches:
            height = p.get_height()
            if not pd.isna(height):
                ax.text(
                    p.get_x() + p.get_width() / 2,
                    height + 0.01,          # position slightly above bar
                    f"{height:.2f}",        # show 2 decimal places
                    ha="center", va="bottom", fontsize=9, color="black"
                )

        plt.tight_layout()
        filename = f"metrics_comparison_by_{group_col}.png"
        plt.savefig(PLOTS_DIR / filename, dpi=300)
        plt.close()
        print(f"✅ Saved plot: {filename}")
    
    print(f"All combined-metric plots saved to: {PLOTS_DIR}\n")

    # --- Helper to generate one leaderboard plot ---
    def plot_top7(df, sort_metric, filename_suffix, title_suffix):
        top7 = df.sort_values(by=sort_metric, ascending=False).head(7)
        top7_long = top7.melt(
            id_vars=["file_name"],
            value_vars=["f1_macro", "f1_weighted"],
            var_name="Metric",
            value_name="Score"
        )

        plt.figure(figsize=(12, 6))
        ax = sns.barplot(
            data=top7_long,
            x="file_name",
            y="Score",
            hue="Metric",
            palette="tab10"
        )
        plt.title(f"Top 7 Experiments — {title_suffix}", fontsize=13)
        plt.ylabel("Score")
        plt.xlabel("Experiment File Name")
        plt.ylim(0, 1.0)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc="upper left")

        # 🔢 Add numeric values above bars
        for p in ax.patches:
            height = p.get_height()
            if not pd.isna(height):
                ax.text(
                    p.get_x() + p.get_width() / 2,
                    height + 0.01,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="black"
                )

        plt.tight_layout()
        filename = f"top7_experiments_sorted_by_{filename_suffix}.png"
        plt.savefig(PLOTS_DIR / filename, dpi=300)
        plt.close()
        print(f"✅ Saved plot: {filename}")

    # --- Generate both leaderboards ---
    plot_top7(df, sort_metric="f1_macro",
                filename_suffix="f1macro",
                title_suffix="Sorted by F1 Macro")

    plot_top7(df, sort_metric="f1_weighted",
                filename_suffix="f1weighted",
                title_suffix="Sorted by F1 Weighted")

    # Encoder plots
    if ENCODER_GLOBAL_CSV.exists():
        ENCODER_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        enc_df = pd.read_csv(ENCODER_GLOBAL_CSV)
        enc_metrics = ["f1_macro", "f1_weighted", "accuracy"]
        enc_groups = ["model_name", "arch", "lang", "size", "tune", "param_mode"]

        for group_col in enc_groups:
            grouped = enc_df.groupby(group_col)[enc_metrics].mean().reset_index()
            long_df = grouped.melt(
                id_vars=[group_col], value_vars=enc_metrics,
                var_name="Metric", value_name="Score"
            )

            plt.figure(figsize=(9, 5))
            ax = sns.barplot(data=long_df, x=group_col, y="Score", hue="Metric", palette="tab10")
            plt.title(f"Encoder Metrics by {group_col.replace('_', ' ').title()}")
            plt.ylabel("Score")
            plt.xlabel(group_col.replace('_', ' ').title())
            plt.ylim(0, 1.0)
            plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.xticks(rotation=30, ha="right")

            for p in ax.patches:
                height = p.get_height()
                if not pd.isna(height):
                    ax.text(
                        p.get_x() + p.get_width() / 2,
                        height + 0.01,
                        f"{height:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        color="black"
                    )

            plt.tight_layout()
            filename = f"encoder_metrics_comparison_by_{group_col}.png"
            plt.savefig(ENCODER_PLOTS_DIR / filename, dpi=300)
            plt.close()
            print(f"�o. Saved plot: {filename}")

        # Freezing ratios
        freeze_df = enc_df[enc_df["tune"].str.startswith("freeze")]
        if not freeze_df.empty:
            grouped = freeze_df.groupby("tune")[enc_metrics].mean().reset_index()
            long_df = grouped.melt(
                id_vars=["tune"], value_vars=enc_metrics,
                var_name="Metric", value_name="Score"
            )

            plt.figure(figsize=(7, 4))
            ax = sns.barplot(data=long_df, x="tune", y="Score", hue="Metric", palette="tab10")
            plt.title("Encoder Metrics by Freeze Ratio")
            plt.ylabel("Score")
            plt.xlabel("Freeze strategy")
            plt.ylim(0, 1.0)
            plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc="upper left")

            for p in ax.patches:
                height = p.get_height()
                if not pd.isna(height):
                    ax.text(
                        p.get_x() + p.get_width() / 2,
                        height + 0.01,
                        f"{height:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        color="black"
                    )

            plt.tight_layout()
            filename = "encoder_metrics_by_freeze.png"
            plt.savefig(ENCODER_PLOTS_DIR / filename, dpi=300)
            plt.close()
            print(f"�o. Saved plot: {filename}")
    else:
        print(f"�s��,? Encoder global summary not found at {ENCODER_GLOBAL_CSV}")

if __name__ == "__main__":
    main()
