import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
METRICS_CSV = BASE_DIR / "results" / "metrics" / "f1_scores.csv"
PLOTS_DIR = BASE_DIR / "results" / "plots"
TARGET_METRICS = ["f1_macro", "f1_weighted"]


def _prepare_long_df(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape the metrics table into a long format filtered to target metrics."""
    long_df = df.melt(id_vars="metric", var_name="model", value_name="score")
    long_df = long_df[long_df["metric"].isin(TARGET_METRICS)].copy()
    long_df["model"] = long_df["model"].astype(str)
    long_df["score"] = long_df["score"].astype(float)
    return long_df


def _add_value_labels(ax) -> None:
    """Annotate bars with their score."""
    for patch in ax.patches:
        height = patch.get_height()
        if pd.isna(height):
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + 0.01,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
        )


def _plot_metric(metric_df: pd.DataFrame, metric_name: str) -> None:
    order = metric_df.sort_values("score", ascending=False)["model"].tolist()
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=metric_df, x="model", y="score", order=order, palette="tab10")
    ax.set_title(f"{metric_name.replace('_', ' ').title()} by Encoder")
    ax.set_ylabel("Score")
    ax.set_xlabel("Encoder Model")
    ax.set_ylim(0, 1.0)
    plt.xticks(rotation=30, ha="right")
    _add_value_labels(ax)
    plt.tight_layout()
    filename = PLOTS_DIR / f"encoder_{metric_name}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"✅ Saved plot: {filename.name}")


def _plot_combined(long_df: pd.DataFrame) -> None:
    order = (
        long_df.groupby("model")["score"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=long_df,
        x="model",
        y="score",
        hue="metric",
        order=order,
        palette="tab10",
    )
    ax.set_title("Encoder F1 Comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("Encoder Model")
    ax.set_ylim(0, 1.0)
    plt.xticks(rotation=30, ha="right")
    _add_value_labels(ax)
    plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    filename = PLOTS_DIR / "encoder_f1_comparison.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"✅ Saved plot: {filename.name}")


def main() -> None:
    """Plot F1 macro and weighted scores for encoder runs."""
    if not METRICS_CSV.exists():
        print(f"⚠️ F1 metrics CSV not found at {METRICS_CSV}")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    df = pd.read_csv(METRICS_CSV)
    if "metric" not in df.columns:
        print(f"⚠️ 'metric' column missing in {METRICS_CSV}")
        return

    long_df = _prepare_long_df(df)
    if long_df.empty:
        print("⚠️ No F1 metrics available to plot.")
        return

    for metric_name in TARGET_METRICS:
        metric_df = long_df[long_df["metric"] == metric_name]
        if metric_df.empty:
            print(f"⚠️ Metric '{metric_name}' not found in CSV; skipping.")
            continue
        _plot_metric(metric_df, metric_name)

    _plot_combined(long_df)
    print(f"All encoder F1 plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
