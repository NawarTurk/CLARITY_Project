from openai import OpenAI
import pandas as pd
import os, time

# ---- configuration ----
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

test_data_path  = "../../datasets/test_dataset.csv"
prediction_path = "../../results/predictions"
prompts_path    = "../../prompts"
prompt_template = "01_t1_zs_re2.txt"
question_col    = "interview_question"

# ---- model info ----
llm_name  = "gpt-5"
provider  = "openai"

# ---- output file name (consistent with HF style) ----
prompt_name = os.path.splitext(prompt_template)[0]
pred_col = f"{llm_name}_{prompt_name}_{provider}_{'question' if question_col == 'question' else 'interview-question'}"
out_path = os.path.join(prediction_path, f"{pred_col}.csv")

# ---- load prompt ----
with open(os.path.join(prompts_path, prompt_template), "r", encoding="utf-8") as f:
    system_prompt = f.read().strip()

# ---- load test data ----
df = pd.read_csv(test_data_path)
if pred_col not in df.columns:
    df[pred_col] = None

# ---- classify function ----
def classify(question: str, answer: str) -> str:
    try:
        completion = client.chat.completions.create(
            model=llm_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"Question: {question}\nAnswer: {answer}\nLabel:"},
            ],
            # temperature=0,
            max_completion_tokens=1500,
        )
        print("RESPONSE:", completion)
        print("CHOICE:", completion.choices[0].message)
        print("CONTENT:", completion.choices[0].message.content)
        return (completion.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"⚠️ Error: {e}")
        time.sleep(3)
        return "Error"

# ---- run predictions ----
for i, row in df.iterrows():
    if pd.notna(row[pred_col]):
        continue

    q, a = row[question_col], row["interview_answer"]
    p = classify(q, a)
    df.at[i, pred_col] = p
    print(f"Index:: {row['index']}\nCorrect Label: {row['clarity_label']}\nPrediction: {p}\n")

    if (i + 1) % 10 == 0:
        os.makedirs(prediction_path, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"💾 Saved progress up to index {i}")

# ---- final save ----
os.makedirs(prediction_path, exist_ok=True)
df.to_csv(out_path, index=False)
print(f"\n✅ All predictions saved to {out_path}")
