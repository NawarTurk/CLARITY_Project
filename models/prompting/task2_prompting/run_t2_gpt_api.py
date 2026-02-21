from openai import OpenAI
import pandas as pd
import os, time
from dotenv import load_dotenv
load_dotenv()

# ---- configuration ----
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# test_data_path  = "../../datasets/test_dataset_with_president.csv"
# test_data_path  = os.path.join("..", "..", "..", "datasets", "codebench_evaluation_dataset", "clarity_task_evaluation_dataset.csv")  # for codebench evl data precdictions
test_data_path  = "aaa.csv"  # for codebench evl data precdictions

# prediction_path = "../../results/predictions/prompt"
prediction_path = os.path.join("..", "..", "..", "results", "codebench_evaluation_prediction", "prompts_task2") # for codebench evl data precdictions

prompts_path    = os.path.join("..", "..", "..", "prompts")
prompt_template = "00_t2_bfs_base-27-shot_IQ-label-details.txt"
question_col    = "question"

# ---- model info ----
llm_name  = "gpt-5"
provider  = "openai"

# ---- output file name (consistent with HF style) ----
prompt_name = os.path.splitext(prompt_template)[0]
pred_col = f"{llm_name}_{prompt_name}_{provider}"

# out_path = os.path.join(prediction_path, f"{pred_col}.csv")
out_path = os.path.join(prediction_path, f"{pred_col}", f"{pred_col}.csv") # # for codebench evl data precdictions
os.makedirs(os.path.dirname(out_path), exist_ok=True)

# ---- load prompt ----
with open(os.path.join(prompts_path, prompt_template), "r", encoding="utf-8") as f:
    system_prompt = f.read().strip()

# ---- load test data ----
df = pd.read_csv(test_data_path)
if pred_col not in df.columns:
    df[pred_col] = None

# ---- classify function ----

def classify(target_q: str, full_q: str, answer: str) -> str:
    user_msg = (
            f"Target question (to evaluate): {target_q}\n"
            f"Full interviewer turn (may contain multiple questions): {full_q}\n"
            f"Answer: {answer}\n"
            f"Label:"
        )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]

    try:
        completion = client.chat.completions.create(
            model=llm_name,
            messages=messages,

            # temperature=0,
            max_completion_tokens=1500,
        )
        # print("RESPONSE:", completion)
        # print("CHOICE:", completion.choices[0].message)
        # print("CONTENT:", completion.choices[0].message.content)
        return (completion.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"⚠️ Error: {e}")
        time.sleep(3)
        return "Error"
    
# ---- run predictions ----
for i, row in df.iterrows():
    # val = row[pred_col]

    # # ONLY predict if blank
    # if isinstance(val, str) and val.strip():
    #  continue
    # if pd.notna(row[pred_col]):
    #     continue
    val = row[pred_col]
    if isinstance(val, str) and val.strip() and val != "Error":
        continue


    target_q = row["question"]
    full_q = row["interview_question"]
    answer = row["interview_answer"]


    pred = classify(target_q, full_q, answer)
    df.at[i, pred_col] = pred
    print(f"Index:: {row['index']}\n\nPrediction: {pred}\n")

    if (i + 1) % 10 == 0:
        os.makedirs(prediction_path, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"💾 Saved progress up to index {i}")

# ---- final save ----
os.makedirs(prediction_path, exist_ok=True)
df.to_csv(out_path, index=False)
print(f"\n✅ All predictions saved to {out_path}")
