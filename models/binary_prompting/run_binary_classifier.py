import os
import time
import pandas as pd
from tqdm import tqdm
from huggingface_hub import InferenceClient
from openai import OpenAI
from dotenv import load_dotenv
from anthropic import Anthropic
from google import genai
from google.generativeai import types


load_dotenv()

# Paths
# dataset_path = os.path.join("..", "..", "datasets", "test_dataset.csv")
dataset_path = os.path.join("..", "..", "results", "predictions", "binary", "gemini-3-flash-preview_binary-28-shot_gemini.csv")



results_path = os.path.join("..", "..", "results", "predictions", "binary")
prompts_path = os.path.join(".", "prompts")

# Load test dataset once
test_df = pd.read_csv(dataset_path)
print(test_df['clarity_label'].value_counts())
# Model registry
MODELS = {
    # "qwen3-235b-instruct": ("Qwen/Qwen3-235B-A22B-Instruct-2507", 'nebius'),
    # "gpt-5": ("gpt-5", 'openai'),
    #  "claude-opus-4-5": ("claude-opus-4-5-20251101", 'anthropic'),
    # "gemini-3-flash-preview": ("gemini-3-flash-preview", "gemini"),
    "gemini-3-flash-preview": ("gemini-3-flash-preview", "gemini"),
    # "gemini-3-pro-preview": ("gemini-3-pro-preview", "gemini"),

}
print(MODELS)
# Binary classifiers - explicit names matching your filenames
BINARY_CLASSIFIERS = [
    {"prompt": "Binary1-C1_t1_fs_binary-clear-reply-28-shot_Q.txt", "name": "clear-reply"},
    {"prompt": "Binary1-C2_t1_fs_binary-ambivalent-28-shot_Q.txt", "name": "ambivalent"},
    {"prompt": "Binary3-C3_t1_fs_binary-clear-non-reply-28-shot_Q.txt", "name": "clear-non-reply"},
]

# HF token
hf_token = os.environ["HF_TOKEN"]
openai_token = os.environ["OPENAI_API_KEY"]
anthropic_token = os.environ["ANTHROPIC_API_KEY"]
gemini_token = os.environ["GEMINI_API_KEY"]

# Run each model with each binary classifier
for llm_name, (model_id, provider) in MODELS.items():
    if provider == 'openai':
        client = OpenAI(api_key=openai_token)
    elif provider == 'anthropic':
        client = Anthropic(api_key=anthropic_token) 
    elif provider == "gemini":
        client = genai.Client()
    else:
        client = InferenceClient(model=model_id, token=hf_token, provider=provider)

    # Single output file for all classifiers
    output_filename = f"{llm_name}_binary-28-shot_{provider}.csv"
    out_path = os.path.join(results_path, output_filename)
    os.makedirs(results_path, exist_ok=True)

    for classifier in BINARY_CLASSIFIERS:
        prompt_file = classifier["prompt"]
        classifier_name = classifier["name"]
        pred_col = f"{llm_name}_binary-{classifier_name}"

        # Load system prompt
        with open(os.path.join(prompts_path, prompt_file), "r", encoding="utf-8") as f:
            system_msg = f.read().strip()

        def classify(target_q: str, full_q: str, answer: str) -> str:
            """Classify using binary classifier"""
            user_msg = (
                f"Target question (to evaluate): {target_q}\n"
                f"Full interviewer turn (may contain multiple questions): {full_q}\n"
                f"Answer: {answer}\n"
                f"Label:"
            )
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]

            if provider == 'openai':
                out = client.responses.create(
                    model=model_id,
                    input=messages,
                    max_output_tokens=100,
                )
                raw = out.output_text
                if not raw:
                    raise ValueError("Empty response from model")
                raw = raw.strip().split()[0].replace(".", "")
                return raw
            elif provider == 'anthropic':
                out = client.messages.create(
                    model=model_id,
                    max_tokens=100,
                    system=system_msg,
                    messages=[{"role": "user", "content": user_msg}]
                )
                raw = (out.content[0].text or "").strip()
            elif provider == 'gemini':
                contents = [
                    {
                        "role": "user",
                        "parts": [{"text": user_msg}],
                    }
                ]

                out = client.models.generate_content(
                    model=model_id,
                    contents=contents,
                    config={
                        "system_instruction": system_msg,
                        "max_output_tokens": 100,
                        "seed": 42,
                    }
                )
                raw = (out.text or "").strip()
            else:
                out = client.chat.completions.create(
                    messages=messages,
                    temperature=0,
                    max_tokens=100,
                )
                raw = (out.choices[0].message.content or "").strip()
                if not raw:
                    raise ValueError("Empty response from Gemini")

            print(f"DEBUG: Raw response = '{raw}'")  # Add this line
            raw = raw.split()[0]   # keeps "Yes" or "No"
            raw = raw.replace(".", "")
            print(raw)
            return raw

        # Add column if not exists
        if pred_col not in test_df.columns:
            test_df[pred_col] = None

        print(f"\n{'='*60}")
        print(f"Model: {llm_name}")
        print(f"Classifier: {classifier['name']}")
        print(f"Prompt: {prompt_file}")
        print(f"Column: {pred_col}")
        print(f"{'='*60}")

        # Run predictions
        for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing"):
            if pd.notna(row[pred_col]):
                continue

            target_q = row["question"]
            full_q = row["interview_question"]
            answer = row["interview_answer"]

            try:
                pred = classify(target_q, full_q, answer)
                test_df.at[i, pred_col] = pred
            except Exception as e:
                print(f"Error at index {i}: {e}. Retrying...")
                time.sleep(2)
                try:
                    pred = classify(target_q, full_q, answer)
                    test_df.at[i, pred_col] = pred
                except Exception as e2:
                    print(f"Failed again: {e2}")
                    test_df.at[i, pred_col] = None

            # Save checkpoint every 10 rows
            if (i + 1) % 10 == 0:
                test_df.to_csv(out_path, index=False)
                time.sleep(0.2)

        # Final save after this classifier completes
        test_df.to_csv(out_path, index=False)
        print(f"\n✓ Saved predictions → {out_path}")

print("\n" + "="*60)
print("All classifiers completed!")
print("="*60)