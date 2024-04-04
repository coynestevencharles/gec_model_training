import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import T5Tokenizer

clang_8_file_path = "./data/clang8_source_target_en.tsv"
tokenized_file_path = "./data/tokenized_data"

df = pd.read_csv(
    clang_8_file_path,
    sep=r"\t",
    on_bad_lines="skip",
    header=None,
    names=["source", "target"],
    engine="python",
)

train_df, eval_df = train_test_split(df, test_size=0.05, shuffle=True)

gec_dataset = DatasetDict(
    {
        "train": Dataset.from_pandas(train_df, preserve_index=False),
        "eval": Dataset.from_pandas(eval_df, preserve_index=False),
    }
)

model_name = "google/t5-v1_1-base"
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)


def tokenize_function(examples):
    max_length = 128
    tokenized_inputs = tokenizer(
        examples["source"], padding="max_length", truncation=True, max_length=max_length
    )
    tokenized_targets = tokenizer(
        examples["target"], padding="max_length", truncation=True, max_length=max_length
    )

    labels = tokenized_targets["input_ids"]
    labels_with_ignore_index = [
        [(label if label != tokenizer.pad_token_id else -100) for label in label_seq]
        for label_seq in labels
    ]

    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": labels_with_ignore_index,
    }


tokenized_datasets = gec_dataset.map(
    tokenize_function, batched=True, remove_columns=["source", "target"]
)
tokenized_datasets.save_to_disk(tokenized_file_path)
