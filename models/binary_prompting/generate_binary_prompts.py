from pathlib import Path
import pandas as pd
import textwrap

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
TRAIN_DATASET_PATH = Path(__file__).resolve().parents[2] / "datasets" / "train_dataset.csv"
Q_COLUMN = "question"
FULL_Q_COLUMN = "interview_question"
A_COLUMN = "interview_answer"
LABEL_COLUMN = "clarity_label"
SEED = 42


def generate_binary_examples(num, target_class):
    """Generate balanced binary examples for a target class vs Rest
    
    Args:
        num: total number of examples (28)
        target_class: 'Clear Reply', 'Ambivalent', or 'Clear Non-Reply'
    
    Returns:
        DataFrame with binary_label ('Yes' for target, 'No' for rest)
    """
    train_df = pd.read_csv(TRAIN_DATASET_PATH)
    
    num_positive = num // 2  # 14 examples of target class
    num_negative = num // 4  # 7 each of the other two classes
    
    # Positive class: target label
    positive = train_df[train_df[LABEL_COLUMN] == target_class].sample(
        n=num_positive, random_state=SEED
    )
    positive = positive.copy()
    positive['binary_label'] = 'Yes'
    
    # Negative class: the other two labels
    all_labels = ['Clear Reply', 'Ambivalent', 'Clear Non-Reply']
    negative_labels = [l for l in all_labels if l != target_class]
    
    neg_examples = []
    for neg_label in negative_labels:
        neg = train_df[train_df[LABEL_COLUMN] == neg_label].sample(
            n=num_negative, random_state=SEED
        )
        neg = neg.copy()
        neg['binary_label'] = 'No'
        neg_examples.append(neg)
    
    negative = pd.concat(neg_examples, ignore_index=True)
    
    # Combine and shuffle
    examples_df = pd.concat([positive, negative], ignore_index=True).sample(
        frac=1, random_state=SEED
    ).reset_index(drop=True)
    
    return examples_df


def generate_prompt_body(examples_df):
    """Generate prompt body with labeled examples"""
    prompt_body = ""
    for i, row in examples_df.iterrows():
        q_formatted = (
            f"Target question (to evaluate): {row[Q_COLUMN]}\n"
            f"Full interviewer turn (may contain multiple questions): {row[FULL_Q_COLUMN]}"
        )
        prompt_body += (
            f"Example {i+1}:\n"
            f"{q_formatted}\n"
            f"Answer: {row[A_COLUMN]}\n"
            f"Label: {row['binary_label']}\n\n"
        )
    return prompt_body


def get_prompt_header(target_class):
    """Return prompt header for the target class with definitions and sub-categories"""
    
    prompts = {
        'Clear Reply': textwrap.dedent("""\
            You are a world-class political discourse analyst trained to detect direct, answering responses in high-stakes interviews.

            Judge whether an answer is a Clear Reply to the journalist's question.

            Important: Judge ONLY whether the Target question is answered directly.
            Ignore other questions in the full interviewer turn.

            Labels (choose exactly ONE):
            
            Yes: Clear Reply
            Definition: Containing replies that admit only one interpretation.
            
            Sub-category:
            - Explicit: The information requested is explicitly stated (in the requested form)
            
            No: Not a Clear Reply

            The following examples illustrate each label:

            """),
        
        'Ambivalent': textwrap.dedent("""\
            You are a world-class political discourse analyst trained to detect indirect or hedged responses in high-stakes interviews.

            Judge whether an answer is Ambivalent to the journalist's question.

            Important: Judge ONLY whether the Target question is answered directly.
            Ignore other questions in the full interviewer turn.

            Labels (choose exactly ONE):
            
            Yes: Ambivalent Reply
            Definition: Where a response is given in the form of a valid answer but allows for multiple interpretations.
            
            Sub-categories:
            - Implicit: The information requested is given, but without being explicitly stated (not in the expected form)
            - General: The information provided is too general/lacks the requested specificity
            - Partial: Offers only a specific component of the requested information
            - Dodging: Ignoring the question altogether
            - Deflection: Starts on topic but shifts the focus and makes a different point than what is asked
            
            No: Not Ambivalent

            The following examples illustrate each label:

            """),
        
        'Clear Non-Reply': textwrap.dedent("""\
            You are a world-class political discourse analyst trained to detect evasive communication strategies in high-stakes interviews.

            Judge whether an answer is a Clear Non-Reply to the journalist's question.

            Important: Judge ONLY whether the Target question is answered directly.
            Ignore other questions in the full interviewer turn.

            Labels (choose exactly ONE):
            
            Yes: Clear Non-Reply
            Definition: Containing responses where the answerer openly refuses to share information.
            
            Sub-categories:
            - Declining to answer: Acknowledge the question but directly or indirectly refusing to answer at the moment
            - Claims ignorance: The answerer claims/admits not to know the answer themselves
            - Clarification: Does not provide the requested information and asks for clarification
            
            No: Not a Clear Non-Reply

            The following examples illustrate each label:

            """),
    }
    
    return prompts[target_class]


def get_prompt_footer():
    """Return consistent footer for all prompts - with literal placeholders"""
    return (
        "\nOutput ONLY the label (Yes or No, no explanation, no punctuation).\n\n"
        "Target question (to evaluate): {Q}\n"
        "Full interviewer turn (may contain multiple questions): {FULL_Q}\n"
        "Answer: {A}\n"
        "Label:"
    )


def main():
    print(f"Using training data from: {TRAIN_DATASET_PATH}")
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    
    num = 28
    target_classes = ['Clear Reply', 'Ambivalent', 'Clear Non-Reply']
    prompt_ids = ['Binary1-C1', 'Binary1-C2', 'Binary3-C3']
    
    for prompt_id, target_class in zip(prompt_ids, target_classes):
        print(f"\nGenerating {target_class} (ID: {prompt_id})...")
        
        # Generate examples
        examples_df = generate_binary_examples(num, target_class)
        
        # Generate components
        base_prompt = get_prompt_header(target_class)
        prompt_body = generate_prompt_body(examples_df)
        footer = get_prompt_footer()
        
        # Combine
        prompt = base_prompt + prompt_body + footer
        
        # Save
        class_name = target_class.lower().replace(' ', '-')
        prompt_name = f"{prompt_id}_t1_fs_binary-{class_name}-{num}-shot_Q.txt"
        out_path = PROMPT_DIR / prompt_name
        
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(prompt)
        
        print(f"✓ Saved: {out_path}")
        print(f"  Examples: {len(examples_df)} (Yes: {(examples_df['binary_label']=='Yes').sum()}, No: {(examples_df['binary_label']=='No').sum()})")


if __name__ == "__main__":
    main()