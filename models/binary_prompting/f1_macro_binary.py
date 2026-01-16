import pandas as pd
from sklearn.metrics import f1_score, classification_report

# Load predictions
results_path = "../../results/predictions/binary"
# pred_file = "qwen3-235b-instruct_binary-28-shot_nebius.csv"
pred_file = "gemini-3-flash-preview_binary-28-shot_gemini.csv"

df = pd.read_csv(f"{results_path}/{pred_file}")
print(df['clarity_label'].value_counts())


# Map binary classifiers to their ground truth in clarity_label
# CLASSIFIER_MAPPING = {
#     'qwen3-235b-instruct_binary-clear-reply': 'Clear Reply',  
#     'qwen3-235b-instruct_binary-ambivalent': 'Ambivalent',
#     'qwen3-235b-instruct_binary-clear-non-reply': 'Clear Non-Reply'  
# }
# CLASSIFIER_MAPPING = {
#     'claude-opus-4-5_binary-clear-reply': 'Clear Reply',  
#     'claude-opus-4-5_binary-ambivalent': 'Ambivalent',
#     'claude-opus-4-5_binary-clear-non-reply': 'Clear Non-Reply'  
# }
CLASSIFIER_MAPPING = {
    'gemini-3-flash-preview_binary-clear-reply': 'Clear Reply',  
    'gemini-3-flash-preview_binary-ambivalent': 'Ambivalent',
    'gemini-3-flash-preview_binary-clear-non-reply': 'Clear Non-Reply'  
}

# Calculate F1 for each binary classifier
for pred_col, target_class in CLASSIFIER_MAPPING.items():
    print(f"\n{'='*60}")
    print(f"Classifier: {pred_col}")
    print(f"Target class: {target_class}")
    print(f"{'='*60}")
    
    # Convert clarity_label to binary: Yes if matches target_class, No otherwise
    y_true = (df['clarity_label'] == target_class).map({True: 'Yes', False: 'No'})
    y_pred = df[pred_col]
    
    # F1 macro
    f1_macro = f1_score(y_true, y_pred, average='macro')
    print(f"F1 Macro: {f1_macro:.4f}")
    
    # F1 per class
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=['Yes', 'No'])
    print(f"Yes F1: {f1_per_class[0]:.4f}")
    print(f"No F1:  {f1_per_class[1]:.4f}")
    
    # Full report
    print(classification_report(y_true, y_pred))