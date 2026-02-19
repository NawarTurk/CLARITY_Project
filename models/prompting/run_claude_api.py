from anthropic import Anthropic
import pandas as pd
import os, time
from dotenv import load_dotenv

# -------------------------------------------------
# Load environment
# -------------------------------------------------
load_dotenv()

if "ANTHROPIC_API_KEY" not in os.environ:
    raise RuntimeError("Missing ANTHROPIC_API_KEY in environment")

anthropic_token = os.environ["ANTHROPIC_API_KEY"]
# -------------------------------------------------
# Client (Claude only)
# -------------------------------------------------
client = Anthropic(api_key=anthropic_token)

# -------------------------------------------------
# Paths
# -------------------------------------------------
test_data_path = "../../datasets/test_dataset.csv"
prediction_path = "../../results/predictions/prompt"
prompts_path = "../../prompts"
prompt_template = "04_t1_fs_base-27-shot_IQ.txt"

# Guards
if not os.path.exists(test_data_path):
    raise FileNotFoundError(f"Dataset not found: {test_data_path}")

prompt_path = os.path.join(prompts_path, prompt_template)
if not os.path.exists(prompt_path):
    raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

# -------------------------------------------------
# Model info (Claude)
# -------------------------------------------------
model_id = "claude-opus-4-5-20251101"
provider = "anthropic"

# -------------------------------------------------
# Output naming
# -------------------------------------------------
prompt_name = os.path.splitext(prompt_template)[0]
pred_col = f"{model_id}_{prompt_name}_{provider}"

out_path = os.path.join(prediction_path, pred_col, f"{pred_col}.csv")
os.makedirs(os.path.dirname(out_path), exist_ok=True)

# -------------------------------------------------
# Load system prompt
# -------------------------------------------------
with open(prompt_path, "r", encoding="utf-8") as f:
    system_msg = f.read().strip()

if not system_msg:
    raise ValueError("System prompt is empty")

# -------------------------------------------------
# Load dataset
# -------------------------------------------------
df = pd.read_csv(test_data_path)

required_cols = {"question", "interview_question", "interview_answer"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

if pred_col not in df.columns:
    df[pred_col] = None

# -------------------------------------------------
# Run predictions
# -------------------------------------------------
for i, row in df.iterrows():
    val = row[pred_col]
    if pd.notna(val) and val != "Error":
        continue

    target_q = row["question"]
    full_q = row["interview_question"]
    answer = row["interview_answer"]

    user_msg = (
        f"Target question (to evaluate): {target_q}\n"
        f"Full interviewer turn (may contain multiple questions): {full_q}\n"
        f"Answer: {answer}\n"
        f"Label:"
    )

    try:
        out = client.messages.create(
            model=model_id,
            max_tokens=100,
            system=system_msg,
            messages=[
                {"role": "user", "content": user_msg}
            ],
        )

        raw = (out.content[0].text or "").strip()
        if not raw:
            raise ValueError("Empty response from Claude")

        pred = raw.strip()
        df.at[i, pred_col] = pred

        print(
            f"Index: {row.get('index', i)}\n"
            f"Gold: {row.get('clarity_label', 'N/A')}\n"
            f"Pred: {pred}\n"
        )

    except Exception as e:
        print(f"⚠️ Claude error at index {i}: {e}")
        df.at[i, pred_col] = "Error"
        time.sleep(3)

    # save every row (crash-safe)
    df.to_csv(out_path, index=False)
    print(f"💾 Saved progress up to index {i}")

# -------------------------------------------------
# Final save
# -------------------------------------------------
df.to_csv(out_path, index=False)
print(f"\n✅ All predictions saved to {out_path}")
