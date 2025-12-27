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

# TEST_DATASET_PATH = Path(__file__).resolve().parent.parent.parent.parent.parent / "datasets" / "test_dataset.csv"
TEST_DATASET_PATH = "one_vs_rest_Qwen-Qwen3-235B-A22B-Instruct-2507_Diff_5_RUN2.csv"
REASONING_COL = "reasoning"
CLOSE_PREDICTION_COL = "close_precision"
PREDICTION_COL = "AI-Agents_One-vs-Rest"
CONFUSED_LABELS_COL = "confused_labels"
DIFF_THRESHOLD = 5  # confidence difference threshold
# OUTPUT_PATH = f"one_vs_rest_{llm_name.replace('/','-')}_Diff_{DIFF_THRESHOLD}.csv"
OUTPUT_PATH = "one_vs_rest_Qwen-Qwen3-235B-A22B-Instruct-2507_Diff_5_RUN3.csv"

client = InferenceClient(model=llm_name, provider="nebius")
# client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def llm_call(prompt):
    response = client.chat.completions.create(messages=[{"role": "user", "content": prompt}], temperature=0, max_tokens=1000)
    # response = client.chat.completions.create(
    #     model=llm_name,
    #     messages=[{"role": "user", "content": prompt}],
    #     max_completion_tokens=2500
    # )
    return response.choices[0].message.content.strip()

def run_agents(question, answer, agents, judge):
    scores = {}
    rationales = {}
    close_precision = False
    labels = ""

    for agent in agents:
        scores[agent.target_label] = agent.predict(question, answer)
    
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_label, top_score = sorted_scores[0]
    second_label, second_score = sorted_scores[1]
    third_label, third_score = sorted_scores[2]

    diff_top2 = top_score - second_score

    if top_score == third_score:
        print("All scores equal, defaulting to Ambivalent")
        return "Ambivalent", top_score, rationales, close_precision
    elif diff_top2 < DIFF_THRESHOLD:
        print(f"Close margin ({diff_top2}), asking judge...")
        close_precision = True
        rationales = {}
        for agent in agents:
            if agent.target_label in [top_label, second_label]:
                rationales[agent.target_label] = agent.explain(question, answer, scores[agent.target_label])
        
        final_label = judge.judge(question, answer, rationales)
        return final_label, top_score, rationales, close_precision
    else:
        return top_label, top_score, rationales, close_precision


if __name__ == "__main__":
    print(TEST_DATASET_PATH)
    df = pd.read_csv(TEST_DATASET_PATH)

    if PREDICTION_COL not in df.columns:
        df[PREDICTION_COL] = None
    
    if REASONING_COL not in df.columns:
        df[REASONING_COL] = None

    if CLOSE_PREDICTION_COL not in df.columns:
        df[CLOSE_PREDICTION_COL] = None
    
    if CONFUSED_LABELS_COL not in df.columns:
        df[CONFUSED_LABELS_COL] = None

    agents = [Agent(label, llm_call) for label in CLARITY_CLASSES]
    judge = Judge(llm_call)

    for index, row in df.iterrows():

        if pd.notna(row[PREDICTION_COL]):
            continue

        question = row['question']
        answer = row['interview_answer']
        correct_label = row['clarity_label']
        print(f"Index: {index}")
        final_label, top_score, reasoning, close_precision = run_agents(question, answer, agents, judge)
        print(f"Final Label: {final_label} | Close Prediction {close_precision} | Use of Judge: {True if reasoning else False}\n")
        print(f"Correct Label: {correct_label}")
        print("-" * 50 + "\n")
        df.loc[index, PREDICTION_COL] = final_label
        df.loc[index, REASONING_COL] = str(reasoning)
        df.loc[index, CLOSE_PREDICTION_COL] = close_precision
        df.loc[index, CONFUSED_LABELS_COL] = str(list(reasoning.keys())) if reasoning else ""

        if index % 10 == 0:
            df.to_csv(OUTPUT_PATH, index=False)
            print(f"Saved interim results to {OUTPUT_PATH}")

    df.to_csv(OUTPUT_PATH, index=False)



    
