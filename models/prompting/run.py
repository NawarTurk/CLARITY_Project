from pathlib import Path
import os
import time
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

ROOT = Path(__file__).resolve().parents[2]
TEST_DATA_PATH = ROOT / "datasets" / "test_dataset.csv"
PREDICTION_PATH = ROOT / "results" / "predictions"

prompt_template_name = "prompt_t1_01_base-template.txt"
llm_name = "qwen-0.5b"

out_path = os.path.join(
    PREDICTION_PATH, f"test_{llm_name}_{prompt_template_name}.csv"
)

MODEL_REGISTRY = {
    # LLaMA family
    "llama-2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    "llama-2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
    "llama-2-70b-chat": "meta-llama/Llama-2-70b-chat-hf",

    # Qwen family
    "qwen-0.5b": "Qwen/Qwen1.5-0.5B-Chat",
    "qwen-1.8b": "Qwen/Qwen1.5-1.8B-Chat",
    "qwen-7b": "Qwen/Qwen1.5-7B-Chat",

    # Other models
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "phi-2.7b": "microsoft/phi-2",
    "falcon-7b": "tiiuae/falcon-7b-instruct",
}

test_df = pd.read_csv(TEST_DATA_PATH)
print(f"Loaded {len(test_df)} rows of the test set")

def build_prompt(question, answer, prompt_template):
    template_path = os.path.join("prompts", prompt_template)
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()
    return template.format(question=question, answer=answer)

def get_pipeline(llm_name):
    model_id = MODEL_REGISTRY[llm_name]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

pred_col = f"{llm_name}_{prompt_template_name}"
if pred_col not in test_df.columns:
    test_df[pred_col] = None

pipe = get_pipeline(llm_name)

for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing"):
    if pd.notna(row[pred_col]):
        continue

    prompt = build_prompt(
        row["question"], row["interview_answer"], prompt_template_name
    )
    raw_output = pipe(
        prompt,
        max_new_tokens=100,
        do_sample=False
    )[0]["generated_text"]
    # Remove the echoed prompt
    prediction = raw_output[len(prompt):].strip()

    test_df.at[i, pred_col] = prediction

    print(f"Question: {row['question']}")
    # print(f"Answer: {row['interview_answer']}")
    print(f"Prediction: {prediction}\n\n")

    if (i+1) % 10 == 0:
        test_df.to_csv(out_path, index=False)
    
    time.sleep(1)

test_df.to_csv(out_path, index=False)
print(f"Saved predictions → {out_path}")