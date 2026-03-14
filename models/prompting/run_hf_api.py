# models/prompting/run_hf_api.py
import os, time, pandas as pd
from tqdm import tqdm
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
load_dotenv()

test_data_path  = os.path.join("..", "..", "datasets", "test_dataset.csv")

prediction_path = os.path.join("..", "..", "results", "predictions", "prompt")

prompts_path    = os.path.join("..", "..", "prompts")

MODEL_REGISTRY = {
    # ---- LLaMA family ----
    "llama-3.1-nemotron-253b": ("nvidia/Llama-3_1-Nemotron-Ultra-253B-v1", 'nebius'),
    "llama-3.3-70b-Instruct": ("meta-llama/Llama-3.3-70B-Instruct", 'nebius'), 

    # # # ---- Qwen family ---- 
    "qwen3-235b-instruct": ("Qwen/Qwen3-235B-A22B-Instruct-2507",'together'), # nebius till feb1
    "qwen3-80b-instruct": ("Qwen/Qwen3-Next-80B-A3B-Instruct", 'novita'), 
    "qwen3-32b-instruct": ("Qwen/Qwen3-30B-A3B-Instruct-2507",'novita'), 

    # # ---- Mixtral ---- 
    "mixtral-8x22b-Instruct": ("mistralai/Mixtral-8x22B-Instruct-v0.1", 'nscale'),  
    }

# ---- configuration ----
prompt_template = "00_t1_zs_base_IQ.txt" 
question_col =  "question"  
hf_token = os.environ["HF_TOKEN"]  

for llm_name, (model_id, provider) in MODEL_REGISTRY.items():
    prompt_name = os.path.splitext(prompt_template)[0] 
    pred_col = f"{llm_name}_{prompt_name}_{provider}"          # e.g., "qwen-1.8b_01_t1_zs_re2_nebius"
    
    out_path = os.path.join(prediction_path, f"{pred_col}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # client
    client   = InferenceClient(model=model_id, token=hf_token, provider=provider)

    # load system message from prompts/
    with open(os.path.join(prompts_path, prompt_template), "r", encoding="utf-8") as f:
        system_msg = f.read().strip()


    def classify(target_q: str, full_q: str, answer: str, president: str = None) -> str:
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
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
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

        target_q = row["question"]
        full_q = row["interview_question"]
        answer = row["interview_answer"]
        president = row.get("president")
        print(president)
        if pd.isna(president):
            president = None

        try:
            pred = classify(target_q, full_q, answer)
            test_df.at[i, pred_col] = pred
        except Exception:
            time.sleep(2)
            pred = classify(target_q, full_q, answer)
            test_df.at[i, pred_col] = pred

        tqdm.write(f"Index:: {row['index']}\nCorrect Label:{row['clarity_label']}\nPrediction: {pred}\n") #_________put it back for eval

        if (i + 1) % 10 == 0:
            os.makedirs(prediction_path, exist_ok=True)
            test_df.to_csv(out_path, index=False)
            time.sleep(0.2)  

    os.makedirs(prediction_path, exist_ok=True)
    test_df.to_csv(out_path, index=False)
    print(f"Saved predictions → {out_path}")

