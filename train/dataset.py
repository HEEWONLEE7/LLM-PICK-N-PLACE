import json
import torch
from torch.utils.data import Dataset

class ModeOnlyDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len=64):
        self.samples = []
        self.tokenizer = tokenizer
        self.label_map = {"대화": 0, "지시": 1}
        self.max_len = max_len

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                text = item["text"]
                mode = item["mode"]

                label = self.label_map[mode]
                encoding = tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt"
                )

                input_ids = encoding["input_ids"].squeeze(0)
                attention_mask = encoding["attention_mask"].squeeze(0)

                self.samples.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "label": torch.tensor(label),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
