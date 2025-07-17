from dataset import ModeOnlyDataset
from model import ModeClassifier  # ← 모델도 단일 분류용으로 가정

from transformers import AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import os

def train():
    model_name = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = ModeOnlyDataset("data/mode_dataset.jsonl", tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = ModeClassifier(model_name).cuda()
    optimizer = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(5):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["label"].cuda()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), "saved_models/mode_classifier.pt")

if __name__ == "__main__":
    train()
