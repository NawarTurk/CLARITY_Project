import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path

CSV_PATH_ENC = Path("t1_encoder-ensemble_en+multi_base+large_encoder-ensemble_fixed_encoder-ensemble_encoder-ensemble_WCE_encoder-ensemble_encoder-ensemble_predictions.csv")

labels = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]

df = pd.read_csv(CSV_PATH_ENC)
cm = confusion_matrix(df["clarity_label"], df["predicted_label"], labels=labels)
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
plt.xlabel("Predicted", fontsize=13)
plt.ylabel("Gold", fontsize=13)
plt.title("Encoder Ensemble - Confusion Matrix (Dev)", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig("confusion_matrix_encoder_ensemble.png", dpi=300)
plt.close()
print("✅ Saved confusion_matrix_encoder_ensemble.png")