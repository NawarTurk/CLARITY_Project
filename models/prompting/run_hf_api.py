# models/prompting/run_hf_api.py
import os, time, pandas as pd
from tqdm import tqdm
from huggingface_hub import InferenceClient

test_data_path  = os.path.join("..", "..", "datasets", "test_dataset.csv")
prediction_path = os.path.join("..", "..", "results", "predictions")
prompts_path    = os.path.join("..", "..", "prompts")

# NOTE: you need to check if a certain model is deployed by any HF inference provider
MODEL_REGISTRY = {
    # LLaMA family
    "llama-2-7b-chat":  "meta-llama/Llama-2-7b-chat-hf",
    "llama-2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
    "llama-2-70b-chat": "meta-llama/Llama-2-70b-chat-hf",

    # Qwen family
    "qwen-0.5b": "Qwen/Qwen1.5-0.5B-Chat",
    "qwen-1.8b": "Qwen/Qwen1.5-1.8B-Chat",
    "qwen-7b":   "Qwen/Qwen1.5-7B-Chat",
    "qwen-2.5-7b":   "Qwen/Qwen2.5-14B-Instruct",

    # Other models
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "phi-2.7b":   "microsoft/phi-2",
    "falcon-7b":  "tiiuae/falcon-7b-instruct",
}

llm_name       = "qwen-2.5-7b"  
model_id       = MODEL_REGISTRY[llm_name]

prompt_template = "01_t1_zs_re2.txt"  
prompt_name     = os.path.splitext(prompt_template)[0] 
pred_col        = f"{llm_name}_{prompt_name}"          # e.g., "qwen-1.8b_01_t1_zs_re2"
out_path        = os.path.join(prediction_path, f"{pred_col}.csv")

# ---- auth & client ----
hf_token = os.environ["HF_TOKEN"]  # token with “Make calls to Inference Providers”
client   = InferenceClient(model=model_id, token=hf_token)

# load system message from prompts/
with open(os.path.join(prompts_path, prompt_template), "r", encoding="utf-8") as f:
    system_msg = f.read().strip()

def classify(question: str, answer: str) -> str:
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": f"Question: {question}\nAnswer: {answer}\nClarity:"},
    ]
    out = client.chat.completions.create(
        messages=messages, temperature=0, max_tokens=10
    )
    return (out.choices[0].message.content or "").strip()

# ---- run ----
test_df = pd.read_csv(test_data_path)
if pred_col not in test_df.columns:
    test_df[pred_col] = None

for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing"):
    if pd.notna(row[pred_col]):    
        continue  
     
    q = row["question"]
    a = row["interview_answer"]
    try:
        p = classify(q, a)
        test_df.at[i, pred_col] = p
    except Exception:
        time.sleep(2.0)
        p = classify(q, a)
        test_df.at[i, pred_col] = p

    tqdm.write(f"Question: {q}\nAnswer: {a}\nPrediction: {p}")

    if (i + 1) % 10 == 0:
        os.makedirs(prediction_path, exist_ok=True)
        test_df.to_csv(out_path, index=False)
        time.sleep(2.0)  

os.makedirs(prediction_path, exist_ok=True)
test_df.to_csv(out_path, index=False)
print(f"Saved predictions → {out_path}")
