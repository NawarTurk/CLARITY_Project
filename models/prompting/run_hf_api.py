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
    "llama-3.1-nemotron-253b": "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1", # nebius
    "llama-3.1-405b-Instruct": "meta-llama/Llama-3.1-405B-Instruct",  # nebius
    # "llama-3-405b-Intruct":    "meta-llama/Llama-3-405B-Instruct",  # ???
    "llama-3.3-70b-Instruct": "meta-llama/Llama-3.3-70B-Instruct",  # nebius 

    # Qwen family
    "qwen3-coder-480b-Instruct": "Qwen/Qwen3-Coder-480B-A35B-Instruct",  # nebius 

    "qwen3-235b-instruct": "Qwen/Qwen3-235B-A22B-Instruct-2507", # nebius 
    "qwen3-80b-instruct": "Qwen/Qwen3-Next-80B-A3B-Instruct", # novita try together 
    "qwen3-32b-instruct": "Qwen/Qwen3-30B-A3B-Instruct-2507", # nebius 

    # Other models
    "mixtral-8x22b-Instruct":   "mistralai/Mixtral-8x22B-Instruct-v0.1", # nscale  
    "mixtral-8x7b-Instruct": "mistralai/Mixtral-8x7B-Instruct-v0.1",  # together

    # "gpt-oss-120b": "openai/gpt-oss-120b",  # returning empty replies  

    # "phi-2.7b":   "microsoft/phi-2",
    # "falcon-7b":  "tiiuae/falcon-7b-instruct",
}

llm_name       = "qwen3-80b-instruct"  
provider       = "novita"  # nebius
model_id       = MODEL_REGISTRY[llm_name]
prompt_template = "01_t1_zs_re2.txt"  

prompt_name     = os.path.splitext(prompt_template)[0] 
pred_col        = f"{llm_name}_{prompt_name}_{provider}"          # e.g., "qwen-1.8b_01_t1_zs_re2_nebius"
out_path        = os.path.join(prediction_path, f"{pred_col}.csv")

# ---- auth & client ----
hf_token = os.environ["HF_TOKEN"]  # token with “Make calls to Inference Providers”
client   = InferenceClient(model=model_id, token=hf_token, provider=provider)

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
        time.sleep(0.5)
        p = classify(q, a)
        test_df.at[i, pred_col] = p

    # tqdm.write(f"Question: {a}\nCorrect Label:{row['clarity_label']}\n'Prediction: {p}")
    tqdm.write(f"Index:: {row['index']}\nCorrect Label:{row['clarity_label']}\nPrediction: {p}\n")

    if (i + 1) % 10 == 0:
        os.makedirs(prediction_path, exist_ok=True)
        test_df.to_csv(out_path, index=False)
        time.sleep(2.0)  

os.makedirs(prediction_path, exist_ok=True)
test_df.to_csv(out_path, index=False)
print(f"Saved predictions → {out_path}")
