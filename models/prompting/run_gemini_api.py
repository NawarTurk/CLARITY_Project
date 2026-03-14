from google import genai
import pandas as pd
import os, time

from dotenv import load_dotenv
load_dotenv()

# ---- configuration ----
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

test_data_path  = "../../datasets/test_dataset_with_president.csv"

prediction_path = "../../results/predictions/prompt"

prompts_path    = "../../prompts"
prompt_template = "04_t1_fs_base-27-shot_IQ-label-details-president.txt"

# ---- model info ----
model_id = "gemini-3-flash-preview"
provider = "gemini"

# ---- output column / file ----
prompt_name = os.path.splitext(prompt_template)[0]
pred_col = f"{model_id}_{prompt_name}_{provider}"

out_path = os.path.join(prediction_path, pred_col, f"{pred_col}.csv")
os.makedirs(os.path.dirname(out_path), exist_ok=True)

# ---- load system prompt ----
with open(os.path.join(prompts_path, prompt_template), "r", encoding="utf-8") as f:
    system_msg = f.read().strip()

# ---- load dataset ----
df = pd.read_csv(test_data_path)
if pred_col not in df.columns:
    df[pred_col] = None

# ---- run predictions ----
for i, row in df.iterrows():
    if pd.notna(row[pred_col]):
        continue

    target_q = row["question"]
    full_q   = row["interview_question"]
    answer   = row["interview_answer"]
    president = row.get("president")
    print(president)
    
    if pd.isna(president):
        president = None

    user_msg = (
        f"Target question (to evaluate): {target_q}\n"
    )
    # optional metadata
    if president and isinstance(president, str) and president.strip():
        user_msg += f"Speaker: {president}\n"

    user_msg += (
        f"Full interviewer turn (may contain multiple questions): {full_q}\n"
        f"Answer: {answer}\n"
        f"Label:"
    )

    contents = [
        {
            "role": "user",
            "parts": [{"text": user_msg}],
        }
    ]

    try:
        out = client.models.generate_content(
            model=model_id,
            contents=contents,
            config={
                "system_instruction": system_msg,
                "max_output_tokens": 1000,
            },
        )

        raw = (out.text or "").strip()
        if not raw:
            raise ValueError("Empty response from Gemini")

        pred = raw.strip()
        df.at[i, pred_col] = pred

        print(
            f"Index: {row['index']}\n"
            f"Gold: {row['clarity_label']}\n"
            f"Pred: {pred}\n"
        )

    except Exception as e:
        print(f"⚠️ Gemini error at index {i}: {e}")
        df.at[i, pred_col] = "Error"
        time.sleep(3)

    if (i + 1) % 1 == 0:
        df.to_csv(out_path, index=False)
        print(f"💾 Saved progress up to index {i}")

# ---- final save ----
df.to_csv(out_path, index=False)
print(f"\n✅ All predictions saved to {out_path}")
