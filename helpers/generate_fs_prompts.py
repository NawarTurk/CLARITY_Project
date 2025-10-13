from pathlib import Path
import pandas as pd
import textwrap

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
TRAIN_DATASET_PATH = Path(__file__).resolve().parents[1] / "datasets" / "train_dataset.csv"
Q_COLUMN = "interview_question"
A_COLUMN = "interview_answer"
LABEL_COLUMN = "clarity_label"
SEED = 42

def generate_random_balanced_examples(num):
    num_per_lable = num//3
    train_df = pd.read_csv(TRAIN_DATASET_PATH)
    train_df = train_df[[Q_COLUMN, A_COLUMN, LABEL_COLUMN]]
    clear_examples = train_df[train_df[LABEL_COLUMN] == 'Clear Reply'].sample(n=num_per_lable, random_state=SEED)
    nonreaply_examples = train_df[train_df[LABEL_COLUMN] == 'Clear Non-Reply'].sample(n=num_per_lable, random_state=SEED)
    ambivalent_examples = train_df[train_df[LABEL_COLUMN] == 'Ambivalent'].sample(n=num_per_lable, random_state=SEED)
    examples_df = pd.concat([clear_examples, ambivalent_examples, nonreaply_examples],
                            ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return examples_df

def generate_prompt_body(examples_df):
    prompt_body = ""
    for i, row in examples_df.iterrows():
        prompt_body += (
        f"Example {i+1}:\n"
        f"QUESTION: {row[Q_COLUMN]}\n"
        f"ANSWER: {row[A_COLUMN]}\n"
        f"Label: {row[LABEL_COLUMN]}\n\n"
        )
    return prompt_body


def main():
    print(f"Using training data from: {TRAIN_DATASET_PATH}")
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)  

    base_prompt = textwrap.dedent("""\
            You are a world-class political discourse analyst trained to detect evasive communication strategies in high-stakes interviews.

            You will judge the clarity of an answer to a journalist’s question.

            Labels (choose exactly ONE):
            - Clear Reply: the answer directly supplies what was asked.
            - Ambivalent Reply: the answer references the topic but is indirect, vague, partial, or hedged.
            - Clear Non-Reply: the answer refuses, claims not to know, asks for clarification, or ignores the question.

            The following examples illustrate each label:

            """)
    
    prompt_footer = (
        "Output ONLY the label name (no explanation, no punctuation).\n\n"
        "QUESTION: {Q}\n"
        "ANSWER: {A}\n\n"
        "Label:"
        )
    counter = 2
    for num in [3, 9, 27, 81]:
        examples_df = generate_random_balanced_examples(num)
        prompt_body = generate_prompt_body(examples_df)
        prompt = base_prompt + prompt_body +  prompt_footer

        prompt_name = f"0{counter}_t1_fs_base-{num}-shot-intvQ.txt"
        out_path = PROMPT_DIR / prompt_name

        with open (out_path, 'w', encoding='utf-8') as f:
            f.write(prompt)
        print(f"Saved: {out_path}")
        counter += 1

        
if __name__ == "__main__":
    main()


