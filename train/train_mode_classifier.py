import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

# 1. 데이터 로드
df = pd.read_csv("data/classification_data.csv")
label_map = {"대화": 0, "지시": 1}
df["label"] = df["label"].map(label_map)

# 2. tokenizer 및 dataset
tokenizer = BertTokenizer.from_pretrained("klue/bert-base")

def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=64)

dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2)
dataset = dataset.map(preprocess)

# 3. 모델 정의
model = BertForSequenceClassification.from_pretrained("klue/bert-base", num_labels=2)

# 4. 학습 인자
training_args = TrainingArguments(
    output_dir="./mode_classifier",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    save_strategy="epoch",
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("./mode_classifier")
