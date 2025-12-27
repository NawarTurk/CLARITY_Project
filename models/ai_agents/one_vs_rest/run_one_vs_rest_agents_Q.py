from agent import Agent
from judge import Judge
from pathlib import Path
import pandas as pd
from huggingface_hub import InferenceClient
import os
from openai import OpenAI

CLARITY_CLASSES = [
    "Clear Reply",
    "Ambivalent",
    "Clear Non-Reply"
]

llm_name = "Qwen/Qwen3-235B-A22B-Instruct-2507"

# TEST_DATASET_PATH = Path(__file__).resolve().parent.parent.parent.parent / "datasets" / "test_dataset.csv"
TEST_DATASET_PATH = "one_vs_rest_Qwen-Qwen3-235B-A22B-Instruct-2507_RUN2.csv"
# OUTPUT_PATH = f"one_vs_rest_{llm_name.replace("/","-")}.csv"
OUTPUT_PATH = "one_vs_rest_Qwen-Qwen3-235B-A22B-Instruct-2507_RUN3.csv"
PREDICTION_COL = "AI-Agents_One-vs-Rest"
YES_LABEL_COL = "Num_Yes_One-vs-Rest"
REASONING_COL = "reasoning"

client = InferenceClient(model=llm_name, provider="nebius")
hf_token = os.environ["HF_TOKEN"] 



def llm_call(prompt):
    response = client.chat.completions.create(messages=[{"role": "user", "content": prompt}], temperature=0, max_tokens=1000)
    return response.choices[0].message.content.strip()

def run_agents(question, answer, agents, judge):
    votes = {}
    rationales = {}
    for agent in agents:
        votes[agent.target_label] = agent.predict(question, answer)

    yes_labels = [k for k, v in votes.items() if v == "YES"]

    if len(yes_labels) == 1:
        print("single YES vote, assigning that label...")
        final_label = yes_labels[0]
    elif len(yes_labels) == 0:
        print("no YES votes, assigning Ambivalent...")
        final_label = "Ambivalent"
    else:
        print("using the judge for tie-breaker...")
        for agent in agents:
            if votes[agent.target_label] == "YES":
                rationales[agent.target_label] = agent.explain(question, answer)

        final_label = judge.judge(question, answer, votes, rationales).strip()

    return final_label, len(yes_labels), rationales

if __name__ == "__main__":
    print(TEST_DATASET_PATH)
    df = pd.read_csv(TEST_DATASET_PATH)

    if PREDICTION_COL not in df.columns:
        df[PREDICTION_COL] = None

    if YES_LABEL_COL not in df.columns:
        df[YES_LABEL_COL] = None
    
    if REASONING_COL not in df.columns:
        df[REASONING_COL] = None

    agents = [Agent(label, llm_call) for label in CLARITY_CLASSES]
    judge = Judge(llm_call)

    for index, row in df.iterrows():        
        if pd.notna(row[PREDICTION_COL]):
            continue
        
        question = row['question']
        answer = row['interview_answer']
        correct_label = row['clarity_label']
        print(f"Index: {index}")
        final_label, num_yes, reasoning = run_agents(question, answer, agents, judge)
        print(f"Final Label: {final_label} (YES votes: {num_yes})\n")
        print(f"Correct Label: {correct_label}")
        print("-" * 50 + "\n")
        df.loc[index, PREDICTION_COL] = final_label
        df.loc[index, YES_LABEL_COL] = num_yes
        df.loc[index, "reasoning"] = str(reasoning)

        if index % 10 == 0:
            df.to_csv(OUTPUT_PATH, index=False)
            print(f"Saved interim results to {OUTPUT_PATH}")

    df.to_csv(OUTPUT_PATH, index=False)



    
