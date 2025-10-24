# =========================================
# train_model.py
# Fine-tune DistilBERT + numeric features on phishing dataset
# =========================================

import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from extract_features import extract_features
from transformers import DistilBertModel
import torch.nn as nn

# -----------------------------
# CONFIGURATION
# -----------------------------
MODEL_NAME = "distilbert-base-uncased"
SAVE_MODEL_DIR = "./fine-tuned-models/final-distilbert-phishing"
RESULTS_DIR = "./results-distilbert"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUMERIC_FEATURES_DIM = 89  # number of extracted features

os.makedirs(SAVE_MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"🧠 Using device: {DEVICE}")

# -----------------------------
# LOAD DATASET
# -----------------------------
DATA_PATH = "./dataset_phishing.csv"
assert os.path.exists(DATA_PATH), f"❌ Dataset not found at {DATA_PATH}"

df = pd.read_csv(DATA_PATH)
label_map = {"legitimate": 0, "phishing": 1}
df["labels"] = df["status"].str.lower().map(label_map)
df["text"] = df["url"].fillna("") + " " + df["content"].fillna("")

# -----------------------------
# EXTRACT NUMERIC FEATURES
# -----------------------------
print("📊 Extracting numeric features...")
df["numeric_features"] = df.apply(lambda row: extract_features({
    "url": row["url"],
    "content": row["content"]
}), axis=1)

# -----------------------------
# CONVERT TO HF DATASET
# -----------------------------
dataset = Dataset.from_pandas(df[["text", "numeric_features", "labels"]])
split = dataset.train_test_split(test_size=0.1, seed=42)
train_ds, eval_ds = split["train"], split["test"]
print(f"✅ Dataset split: {len(train_ds)} train / {len(eval_ds)} eval")

# -----------------------------
# TOKENIZER
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    tokens["numeric_features"] = torch.tensor(example["numeric_features"], dtype=torch.float)
    tokens["labels"] = torch.tensor(example["labels"], dtype=torch.long)
    return tokens

train_ds = train_ds.map(tokenize_function, batched=False)
eval_ds = eval_ds.map(tokenize_function, batched=False)

# -----------------------------
# CUSTOM MODEL
# -----------------------------
class DistilBERTWithFeatures(nn.Module):
    def __init__(self, base_model_name, num_features):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(base_model_name)
        self.num_features_fc = nn.Linear(num_features, 32)
        self.classifier = nn.Linear(self.bert.config.hidden_size + 32, 2)

    def forward(self, input_ids=None, attention_mask=None, numeric_features=None, labels=None):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = bert_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        num_feat_proj = self.num_features_fc(numeric_features)
        combined = torch.cat([cls_emb, num_feat_proj], dim=1)
        logits = self.classifier(combined)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
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
# TRAINING ARGUMENTS
# -----------------------------
args = TrainingArguments(
    output_dir=RESULTS_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
    save_strategy="epoch",
    save_total_limit=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

# -----------------------------
# CUSTOM DATA COLLATOR
# -----------------------------
def data_collator(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    numeric_features = torch.stack([b["numeric_features"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "numeric_features": numeric_features,
        "labels": labels
    }

# -----------------------------
# TRAINER
# -----------------------------
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

# -----------------------------
# TRAIN & SAVE
# -----------------------------
print("🚀 Training started...")
trainer.train()
torch.save(model.state_dict(), os.path.join(SAVE_MODEL_DIR, "pytorch_model.bin"))
tokenizer.save_pretrained(SAVE_MODEL_DIR)
print(f"✅ Model saved to {SAVE_MODEL_DIR}")
