# =========================================
# train_model.py
# Fine-tune DistilBERT + numeric features on phishing dataset
# =========================================

import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from extract_features import extract_features
import torch.nn as nn
from transformers import DistilBertModel
from safetensors.torch import save_file

# -----------------------------
# CONFIG
# -----------------------------
MODEL_NAME = "distilbert-base-uncased"
SAVE_MODEL_DIR = "./fine-tuned-models/final-distilbert-phishing"
RESULTS_DIR = "./results-distilbert"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUMERIC_FEATURES_DIM = 87  # must match extract_features.py

os.makedirs(SAVE_MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"🧠 Using device: {DEVICE}")

# -----------------------------
# LOAD DATA
# -----------------------------
DATA_PATH = "./data/dataset_phishing.csv"
df = pd.read_csv(DATA_PATH)
df["labels"] = df["status"].str.lower().map({"legitimate": 0, "phishing": 1})
df["text"] = df["url"].fillna("")

# -----------------------------
# EXTRACT NUMERIC FEATURES
# -----------------------------
print("📊 Extracting numeric features...")
numeric_features = df["url"].apply(lambda url: extract_features({"url": url}))
numeric_features_array = np.stack(numeric_features.values)
df_numeric = pd.DataFrame(numeric_features_array, columns=[f"f{i}" for i in range(NUMERIC_FEATURES_DIM)])
df = pd.concat([df[["text", "labels"]], df_numeric], axis=1)

# -----------------------------
# CREATE HF DATASET
# -----------------------------
dataset = Dataset.from_pandas(df)
split = dataset.train_test_split(test_size=0.1, seed=42)
train_ds, eval_ds = split["train"], split["test"]
print(f"✅ Dataset split: {len(train_ds)} train / {len(eval_ds)} eval")

# -----------------------------
# TOKENIZER
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_and_add_features(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    tokens["numeric_features"] = [example[f"f{i}"] for i in range(NUMERIC_FEATURES_DIM)]
    tokens["labels"] = example["labels"]
    return tokens

train_ds = train_ds.map(tokenize_and_add_features, batched=False)
eval_ds = eval_ds.map(tokenize_and_add_features, batched=False)

# -----------------------------
# DATA COLLATOR
# -----------------------------
def collate_fn(batch):
    return {
        "input_ids": torch.tensor([b["input_ids"] for b in batch], dtype=torch.long),
        "attention_mask": torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long),
        "numeric_features": torch.tensor([b["numeric_features"] for b in batch], dtype=torch.float),
        "labels": torch.tensor([b["labels"] for b in batch], dtype=torch.long)
    }

# -----------------------------
# MODEL
# -----------------------------
class DistilBERTWithFeatures(nn.Module):
    def __init__(self, base_model_name, num_features):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(base_model_name)
        self.num_features_fc = nn.Linear(num_features, 32)
        self.classifier = nn.Linear(self.bert.config.hidden_size + 32, 2)

    def forward(self, input_ids=None, attention_mask=None, numeric_features=None, labels=None):
        cls_emb = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,0,:]
        num_feat_proj = self.num_features_fc(numeric_features)
        logits = self.classifier(torch.cat([cls_emb, num_feat_proj], dim=1))
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}

model = DistilBERTWithFeatures(MODEL_NAME, NUMERIC_FEATURES_DIM).to(DEVICE)

# -----------------------------
# METRICS
# -----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_weighted": f1_score(labels, preds, average="weighted")
    }

# -----------------------------
# TRAINING
# -----------------------------
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=RESULTS_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_dir="./logs",
        logging_steps=100,
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        report_to="none",
    ),
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=collate_fn
)

print("🚀 Training started...")
trainer.train()

# -----------------------------
# SAVE FINAL MODEL + TOKENIZER
# -----------------------------
print("💾 Saving final model and tokenizer...")

# Save weights as safetensors
model_path = os.path.join(SAVE_MODEL_DIR, "model.safetensors")
save_file(model.state_dict(), model_path)

# Save tokenizer and config files for from_pretrained()
tokenizer.save_pretrained(SAVE_MODEL_DIR)
model.bert.config.save_pretrained(SAVE_MODEL_DIR)

print(f"✅ Model and tokenizer saved to {SAVE_MODEL_DIR}")
print("Saved files:", sorted(os.listdir(SAVE_MODEL_DIR)))
