import random
import numpy as np
import torch
from datasets import load_dataset, load_metric
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)


def tokenize(examples):
    return tokenizer(examples['premise'], examples['hypothesis'], padding='max_length', truncation=True)


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

dataset = load_dataset("super_glue", "rte")

model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenized_datasets = dataset.map(tokenize, batched=True)

training_args = TrainingArguments(
    output_dir="./rte",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="steps",
    eval_steps=100,
    save_total_limit=2,
    num_train_epochs=3,
    save_steps=100,
    learning_rate=2e-5,
    load_best_model_at_end=True,
)

metric = load_metric("glue", "rte")

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

results_before = trainer.predict(tokenized_datasets["validation"])
accuracy_before = compute_metrics((results_before.predictions, results_before.label_ids))["accuracy"]

trainer.train()

results_after = trainer.evaluate()

print(f"Accuracy Before Fine-Tuning: {accuracy_before:.2f}")
print(f"Accuracy After Fine-Tuning: {results_after['eval_accuracy']:.2f}")

model.save_pretrained("fine_tuned_bert")
tokenizer.save_pretrained("fine_tuned_bert")
