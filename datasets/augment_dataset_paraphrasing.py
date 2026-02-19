import pandas as pd
import re
from typing import List
from openai import OpenAI

# ------------------ helpers ------------------
def count_sentences(text: str) -> int:
    return len(re.findall(r"[.!?]+", text))

def count_words(text: str) -> int:
    return len(text.split())

def passes_constraints(generated: str, original: str, word_tol=0.20, sent_tol=1) -> bool:
    orig_words = count_words(original)
    gen_words = count_words(generated)

    if orig_words == 0:
        return False

    if not (orig_words * (1 - word_tol) <= gen_words <= orig_words * (1 + word_tol)):
        return False

    orig_sents = count_sentences(original)
    gen_sents = count_sentences(generated)

    # sentence count ±1
    if abs(orig_sents - gen_sents) > sent_tol:
        return False

    return True

# ------------------ LLM generation ------------------
def generate_paraphrases(
    answers: List[str],
    client: OpenAI,
    model: str = "gpt-4o",
    temperature: float = 0.55,
) -> List[str]:
    paraphrases = []

    system_msg = (
        "You are rewriting a political interview answer.\n"
        "The rewritten answer must remain a CLEAR NON-REPLY.\n"
        "Hard constraints:\n"
        "- Use the SAME number of sentences as the original (±1).\n"
        "- Keep the total word count within ±20% of the original.\n"
        "- Do NOT add or remove sentences.\n"
        "- Do NOT turn the answer into generic politeness or conversational filler.\n"
        "Length-specific constraints:\n"
        "- If the original answer has fewer than 15 words, preserve it almost exactly—only swap 3-4 words maximum.\n"
        "- If the original answer has 15-50 words, rephrase lightly while keeping structure identical.\n"
        "- If the original answer has 50+ words, maintain the same point-by-point structure; do NOT summarize or condense.\n"
        "- Do NOT expand short answers into full explanations.\n"
        "Behavioral constraints:\n"
        "- Do NOT answer the interviewer's question.\n"
        "- Reference the question indirectly or reframe it without addressing it.\n"
        "- Sound like a political deflection, not small talk.\n"
        "- Do NOT introduce new facts, examples, or topics.\n"
        "Style preservation:\n"
        "- Preserve hesitation, repetition, vagueness, and informal phrasing if present.\n"
        "- Preserve specific details, names, numbers, and examples from the original.\n"
        "- Do NOT clarify, summarize, or improve the structure of the answer.\n"
        "Forbidden patterns:\n"
        "- Do NOT add abstract commentary, meta-discussion, or generalized statements.\n"
        "- Do NOT replace concrete content with vague substitutes.\n"
        "- Avoid phrases such as: 'broader context', 'important to recognize', 'complex issue',\n"
        "  'it's worth noting', 'we need to consider', 'different perspectives', 'multifaceted'.\n"
        "Quality constraints:\n"
        "- Preserve the original meaning and tone.\n"
        "- The answer must remain fluent and coherent.\n"
        "- Keep the same level of formality or informality as the original.\n"
        "Before responding, silently verify that all constraints are satisfied.\n"
        "If any constraint cannot be satisfied, produce the closest possible CLEAR NON-REPLY without expanding the content."
    )


    # system_msg = (
    #     "You are rewriting a political interview answer.\n"
    #     "The rewritten answer must remain a CLEAR NON-REPLY.\n"
    #     "Hard constraints:\n"
    #     "- Use the SAME number of sentences as the original (±1).\n"
    #     "- Keep the total word count within ±20% of the original.\n"
    #     "- Do not add or remove sentences.\n"
    #     "- Do not turn the answer into generic politeness or conversational filler.\n"
    #     "Behavioral constraints:\n"
    #     "- Do NOT answer the interviewer's question.\n"
    #     "- It should reference the question indirectly or reframe it.\n"
    #     "- It should sound like a political deflection, not small talk.\n"
    #     "- Do not introduce new facts or topics.\n"
    #     "Quality constraints:\n"
    #     "- Preserve the original meaning and tone.\n"
    #     "- The answer should remain fluent and coherent.\n"
    #     "Before responding, silently verify that all constraints are satisfied."
    # )
    # system_msg = (
    #     "You are rewriting a political interview answer.\n" # we should say unclear ______
    #     "Keep the answer clearly evasive.\n"
    #     "Do NOT answer the question.\n"
    #     "Do NOT add facts.\n"
    #     "Preserve meaning and tone."
    # )

    MAX_RETRIES = 3
    for ans in answers:
        final_output = None
        for attempt in range(MAX_RETRIES):
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": ans}
                ]
            )
            output = response.choices[0].message.content.strip()
            if passes_constraints(output, ans):
                final_output = output
                break
            final_output = output

        paraphrases.append(final_output)
        print(f"Original Answer: {ans}\n----\n")
        print(f"Augmentedone: {final_output}\n\n***\n")

    return paraphrases

# ------------------ filtering ------------------
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["orig_sentences_count"] = df["source_answer"].apply(count_sentences)
    df["para_sentences_count"] = df["interview_answer"].apply(count_sentences)
    df["passes_sentence_filter"] = (
        (df["para_sentences_count"] - df["orig_sentences_count"]).abs() <= 1
    ).astype(int)
    
    df["orig_words_count"] = df["source_answer"].apply(count_words)
    df["para_words_count"] = df["interview_answer"].apply(count_words)
    df["passes_word_filter"] = (
        (df["para_words_count"] >= 0.8 * df["orig_words_count"])
        & (df["para_words_count"] <= 1.2 * df["orig_words_count"])
    ).astype(int)
    
    df["passes_all_filters"] = (
        (df["passes_sentence_filter"] == 1)
        & (df["passes_word_filter"] == 1)
    ).astype(int)
    return df

# ------------------ main ------------------
def main():
    INPUT_CSV = "../datasets/train_dataset.csv"
    AUGMENTED_CSV = "train_dataset_augmented.csv"
    FILTERED_CSV = "train_dataset_augmented_filtered.csv"
    
    df = pd.read_csv(INPUT_CSV)
    
    # add source_answer for originals (schema consistency)
    df["source_answer"] = df["interview_answer"]
    
    # mark originals
    df["is_augmented"] = 0
    df["augmentation_method"] = None
    df["source_index"] = None
    df["passes_sentence_filter"] = 1
    df["passes_word_filter"] = 1
    df["passes_all_filters"] = 1

    # extract Clear Non-Reply
    minority_df = df[df["clarity_label"] == "Clear Non-Reply"].copy()
    print(f"Found {len(minority_df)} Clear Non-Reply answers")
    
    client = OpenAI()
    
    # generate 450 paraphrases (356 base + 194 from random sample)
    print("Generating first batch (356 paraphrses)...")
    paraphrases = generate_paraphrases(
        answers=minority_df["interview_answer"].tolist(),
        client=client
    )
    
    print("Generating second batch (194 paraphrases from random sample)...")
    extra = generate_paraphrases(
        answers=minority_df["interview_answer"].sample(194, replace=False, random_state=42).tolist(),
        client=client,
    )

    paraphrases.extend(extra)
    print(f"Total paraphrases generated: {len(paraphrases)}")
    
    # build augmented rows with correct source alignment
    aug_rows = []
    next_index = df["index"].max() + 1
    
    # align sources: 356 originals + 194 sampled (with replacement, seeded)
    source_rows = (
        list(minority_df.iterrows())
        + list(minority_df.sample(194, replace=False, random_state=42).iterrows())
    )

    for (_, row), para in zip(source_rows, paraphrases):
        new_row = row.copy()
        new_row["interview_answer"] = para
        new_row["is_augmented"] = 1
        new_row["augmentation_method"] = "llm_paraphrase"
        new_row["source_index"] = row["index"] if "index" in row else row.name
        new_row["index"] = next_index
        new_row["source_answer"] = row["interview_answer"]
        next_index += 1
        aug_rows.append(new_row)
    
    aug_df = pd.DataFrame(aug_rows)
    
    # apply filters
    print("Applying filters...")
    aug_df = apply_filters(aug_df)
    
    # report filter statistics
    sentence_pass = aug_df["passes_sentence_filter"].sum()
    word_pass = aug_df["passes_word_filter"].sum()
    all_pass = aug_df["passes_all_filters"].sum()
    
    print(f"\nFilter Results:")
    print(f"  Sentence filter: {sentence_pass}/{len(aug_df)} ({100*sentence_pass/len(aug_df):.1f}%)")
    print(f"  Word filter: {word_pass}/{len(aug_df)} ({100*word_pass/len(aug_df):.1f}%)")
    print(f"  Both filters passed: {all_pass}/{len(aug_df)} ({100*all_pass/len(aug_df):.1f}%)")
    
    # combine originals + augmented (audit dataset)
    full_augmented = pd.concat([df, aug_df], ignore_index=True)
    full_augmented.to_csv(AUGMENTED_CSV, index=True)
    
    # build filtered training dataset
    passed_aug = aug_df[aug_df["passes_all_filters"] == 1].head(356)
    train_filtered = pd.concat([df, passed_aug], ignore_index=True)
    train_filtered.to_csv(FILTERED_CSV, index=False)
    
    # report final class distribution
    print(f"\nFinal Class Distribution:")
    print(train_filtered["clarity_label"].value_counts())
    print(f"\nClear Non-Reply: {len(df[df['clarity_label'] == 'Clear Non-Reply'])} → {len(train_filtered[train_filtered['clarity_label'] == 'Clear Non-Reply'])}")
    
    print(f"\n✓ Augmented dataset saved to {AUGMENTED_CSV}")
    print(f"✓ Filtered training dataset saved to {FILTERED_CSV}")
    print(f"✓ Kept {len(passed_aug)}/356 augmented samples")

if __name__ == "__main__":
    main()