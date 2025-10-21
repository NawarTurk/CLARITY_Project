import os
import random
from typing import List

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    default_data_collator,
    set_seed,
)

SEED = 0
MAX_LENGTH = 512
BATCH_SIZE = 32

DATASET_PATHS = [
    os.path.join("datasets", "train_dataset.csv"),
    os.path.join("datasets", "test_dataset.csv"),
]
OUTPUT_DIR = os.path.join("results", "embeddings")

MODEL_NAME = "distilbert-base-uncased"
ARG1_KEY = "question"
ARG2_KEY = "interview_answer"


def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    print(f"[Info] Global seed set: {seed}")


set_global_seed(SEED)


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    arg1_key: str,
    arg2_key: str,
) -> Dataset:
    def format_pair(question, answer):
        q_str = "" if question is None else str(question)
        a_str = "" if answer is None else str(answer)
        return f"Interview_question : {q_str}\nAnswer: {a_str}"

    def tokenize(examples):
        questions = examples[arg1_key]
        answers = examples.get(arg2_key, [""] * len(questions))
        if len(answers) < len(questions):
            answers = list(answers) + [""] * (len(questions) - len(answers))
        combined = [format_pair(q, a) for q, a in zip(questions, answers)]
        return tokenizer(
            combined,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

    columns: List[str] = ["input_ids", "attention_mask"]
    if "token_type_ids" in tokenized.column_names:
        columns.append("token_type_ids")

    tokenized.set_format(type="torch", columns=columns)
    return tokenized


def extract_embeddings(
    model: AutoModel,
    dataset: Dataset,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=default_data_collator)
    collected: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            if getattr(outputs, "pooler_output", None) is not None:
                pooled = outputs.pooler_output
            else:
                pooled = outputs.last_hidden_state[:, 0, :]

            collected.append(pooled.cpu())

    if not collected:
        return np.empty((0, model.config.hidden_size), dtype=np.float32)

    embeddings = torch.cat(collected, dim=0).to(torch.float32)
    return embeddings.numpy()


def generate_test_embeddings(
    model_name: str,
    test_csv_path: str,
    arg1_key: str,
    arg2_key: str, 
    ) -> None:
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test CSV not found at {test_csv_path}")

    test_df = pd.read_csv(test_csv_path)
    print(f"[Info] Loaded {len(test_df)} rows from {test_csv_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    dataset = Dataset.from_pandas(test_df[[arg1_key, arg2_key]].copy())
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, arg1_key, arg2_key)

    embeddings = extract_embeddings(model, tokenized_dataset, batch_size=BATCH_SIZE)
    print(f"[Info] Generated embeddings with shape: {embeddings.shape}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data_basename = os.path.splitext(os.path.basename(test_csv_path))[0]
    simplified_name = data_basename.replace("_dataset", "")
    base_filename = f"{simplified_name}_embeddings"

    npy_path = os.path.join(OUTPUT_DIR, f"{base_filename}.npy")
    np.save(npy_path, embeddings)
    print(f"[Info] Saved raw embeddings to: {npy_path}")

    embedding_columns = {f"dim_{i}": embeddings[:, i] for i in range(embeddings.shape[1])}
    output_df = pd.concat(
        [test_df.reset_index(drop=True), pd.DataFrame(embedding_columns)],
        axis=1,
    )

    csv_path = os.path.join(OUTPUT_DIR, f"{base_filename}.csv")
    output_df.to_csv(csv_path, index=False)
    print(f"[Info] Saved embeddings with metadata to: {csv_path}")


if __name__ == "__main__":
    for dataset_path in DATASET_PATHS:
        generate_test_embeddings(
            model_name=MODEL_NAME,
            test_csv_path=dataset_path,
            arg1_key=ARG1_KEY,
            arg2_key=ARG2_KEY,
        )
