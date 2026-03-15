import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path

CSV_PATH = Path("gpt-5-gemini-3-flash-preview-qwen3-235b-instruct_promptEnsembleMajorityVote_promptEnsembleMajorityVote_promptEnsembleMajorityVote_promptEnsembleMajorityVote_promptEnsembleMajorityVote_promptEnsembleMajorityVote_VALIDATED.csv")

df = pd.read_csv(CSV_PATH)

labels = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]

cm = confusion_matrix(df["clarity_label"], df["model_prediction"], labels=labels)
print(cm)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    annot_kws={"size": 14}
)
plt.xlabel("Predicted")
plt.ylabel("Gold")
plt.title("LLM Ensemble - Confusion Matrix (Dev)")
plt.tight_layout()
plt.savefig("confusion_matrix_llm_ensemble.png", dpi=300)
plt.close()
print("✅ Saved confusion_matrix_llm_ensemble.png")