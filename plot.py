import matplotlib.pyplot as plt

# =========================
# Data
# =========================
techniques = ["FS27", "FS3", "FS9", "ZS+Re2", "ZS", "CoT-style"]
tech_scores = [0.56, 0.53, 0.51, 0.51, 0.49, 0.49]

inputs = ["Enriched", "Atomic"]
input_scores = [0.54, 0.49]

models = [
    "GPT-5",
    "Qwen3-235B",
    "Qwen3-80B",
    "Qwen3-32B",
    "Mixtral-8x22B",
    "LLaMA-3.3-70B",
    "LLaMA-3.1-253B"
]
model_scores = [0.62, 0.59, 0.53, 0.51, 0.51, 0.42, 0.31]

COLOR_PROMPT = "#72B7B2"
COLOR_INPUT  = "#E7A977"
COLOR_MODEL  = "#4C78A8"

fig = plt.figure(figsize=(10, 7))
gs = fig.add_gridspec(2, 4)

ax_prompt = fig.add_subplot(gs[0, :3])
ax_input  = fig.add_subplot(gs[0, 3])
ax_models = fig.add_subplot(gs[1, :])

# ---- Prompt Technique
bars1 = ax_prompt.bar(techniques, tech_scores, color=COLOR_PROMPT)
ax_prompt.set_title("Prompting Technique", fontsize=16)
ax_prompt.set_ylabel("Macro-F1", fontsize=12)
ax_prompt.set_ylim(0, 1.0)
ax_prompt.tick_params(axis="x", labelsize=16, rotation=20)

# ---- Input Mode
bars2 = ax_input.bar(inputs, input_scores, color=COLOR_INPUT)
ax_input.set_title("Input Mode", fontsize=16)
ax_input.set_ylim(0, 1.0)
ax_input.tick_params(axis="x", labelsize=16)

# ---- Models
bars3 = ax_models.bar(models, model_scores, color=COLOR_MODEL)
ax_models.set_title("LLM Model Comparison", fontsize=16)
ax_models.set_ylabel("Macro-F1", fontsize=12)   # <-- added line
ax_models.set_ylim(0, 1.0)
ax_models.tick_params(axis="x", labelsize=14, rotation=25)

# ---- Y-axis tick size for all
for ax in [ax_prompt, ax_input, ax_models]:
    ax.tick_params(axis='y', labelsize=7)

# ---- Value labels
def add_labels(ax, bars, values):
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02,
            f"{val:.2f}",
            ha="center",
            fontsize=12
        )

add_labels(ax_prompt, bars1, tech_scores)
add_labels(ax_input, bars2, input_scores)
add_labels(ax_models, bars3, model_scores)

plt.tight_layout()
plt.savefig("design_vs_model_dev_macro_f1.png", dpi=300)
plt.show()