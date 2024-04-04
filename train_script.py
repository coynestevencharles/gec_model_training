from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
)
from datasets import load_from_disk
from accelerate import Accelerator
import os

os.environ["WANDB_PROJECT"] = "gec_t5_small"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

tokenized_file_path = "./data/tokenized_data"

model_name = "google/t5-v1_1-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)

accelerator = Accelerator(mixed_precision="fp16")

tokenized_datasets = load_from_disk(tokenized_file_path)

data_collator = DataCollatorForSeq2Seq(
    tokenizer, model=model, padding="longest", return_tensors="pt"
)

batch_size = 32
num_epochs = 3
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="steps",
    eval_steps=1000,
    logging_steps=1000,
    save_steps=1000,
    save_total_limit=3,
    learning_rate=2e-5,
    lr_scheduler_type="linear",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    report_to="wandb",
    gradient_accumulation_steps=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["eval"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer = accelerator.prepare(trainer)

trainer.train()
