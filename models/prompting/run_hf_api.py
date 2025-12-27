# models/prompting/run_hf_api.py
import os, time, pandas as pd
from tqdm import tqdm
from huggingface_hub import InferenceClient

test_data_path  = os.path.join("..", "..", "datasets", "test_dataset.csv")
prediction_path = os.path.join("..", "..", "results", "predictions", "prompt")
prompts_path    = os.path.join("..", "..", "prompts")

# NOTE: you need to check if a certain model is deployed by any HF inference provider
MODEL_REGISTRY = {
    # ---- LLaMA family ----
    "llama-3.1-nemotron-253b": ("nvidia/Llama-3_1-Nemotron-Ultra-253B-v1", 'nebius'),
    # "llama-3.1-405b-Instruct": ("meta-llama/Llama-3.1-405B-Instruct", 'nebius'), no longer working
    "llama-3.3-70b-Instruct": ("meta-llama/Llama-3.3-70B-Instruct", 'nebius'), 

    # # ---- Qwen family ---- 
    "qwen3-coder-480b-Instruct": ("Qwen/Qwen3-Coder-480B-A35B-Instruct",  'nebius'), 

    "qwen3-235b-instruct": ("Qwen/Qwen3-235B-A22B-Instruct-2507",'nebius'), 
    "qwen3-80b-instruct": ("Qwen/Qwen3-Next-80B-A3B-Instruct", 'novita'), 
    "qwen3-32b-instruct": ("Qwen/Qwen3-30B-A3B-Instruct-2507",'nebius'), 

    # ---- Mixtral ---- 
    "mixtral-8x22b-Instruct": ("mistralai/Mixtral-8x22B-Instruct-v0.1", 'nscale'),  
    # "mixtral-8x7b-Instruct": ("mistralai/Mixtral-8x7B-Instruct-v0.1", 'together'), no longer working
}

# ---- configuration ----
prompt_template = "07_t1_cot_thot_Q.txt" 
question_col =  "question"  
hf_token = os.environ["HF_TOKEN"]  # token with “Make calls to Inference Providers”

for llm_name, (model_id, provider) in MODEL_REGISTRY.items():
    prompt_name = os.path.splitext(prompt_template)[0] 
    pred_col = f"{llm_name}_{prompt_name}_{provider}"          # e.g., "qwen-1.8b_01_t1_zs_re2_nebius"
    out_path = os.path.join(prediction_path, f"{pred_col}.csv")

    # client
    client   = InferenceClient(model=model_id, token=hf_token, provider=provider)

    # load system message from prompts/
    with open(os.path.join(prompts_path, prompt_template), "r", encoding="utf-8") as f:
        system_msg = f.read().strip()

    def classify(question: str, answer: str) -> str:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": f"Question: {question}\nAnswer: {answer}\nLabel:"},
        ]
        out = client.chat.completions.create(
            messages=messages, temperature=0, max_tokens=1000
        )
        return (out.choices[0].message.content or "").strip()

    # ---- run ----
    test_df = pd.read_csv(test_data_path)
    if pred_col not in test_df.columns:
        test_df[pred_col] = None

    print(f'Running prediction using model: {llm_name}')
    for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing"):
        if pd.notna(row[pred_col]):    
            continue  
        
        q = row[question_col]
        a = row["interview_answer"]
        try:
            p = classify(q, a)
            test_df.at[i, pred_col] = p
        except Exception:
            time.sleep(2)
            p = classify(q, a)
            test_df.at[i, pred_col] = p

        # tqdm.write(f"Question: {a}\nCorrect Label:{row['clarity_label']}\n'Prediction: {p}")
        tqdm.write(f"Index:: {row['index']}\nCorrect Label:{row['clarity_label']}\nPrediction: {p}\n")

        if (i + 1) % 10 == 0:
            os.makedirs(prediction_path, exist_ok=True)
            test_df.to_csv(out_path, index=False)
            time.sleep(0.2)  

    os.makedirs(prediction_path, exist_ok=True)
    test_df.to_csv(out_path, index=False)
    print(f"Saved predictions → {out_path}")
