# =========================================
# train_model.py
# Fine-tune DistilBERT + numeric features on phishing dataset
# =========================================

import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold, train_test_split
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
NUMERIC_FEATURES_DIM = 45  # must match extract_features.py
N_FOLDS = 5

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
# SPLIT TEST SET (1000 samples)
# -----------------------------
print("🔀 Splitting dataset into train/val and test sets...")
test_size = 1000
trainval_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df["labels"])
print(f"✅ Dataset split: {len(trainval_df)} train/validation / {len(test_df)} test")

trainval_dataset = Dataset.from_pandas(trainval_df.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

# -----------------------------
# TOKENIZER
# -----------------------------
print("🔤 Initializing tokenizer...")
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
# K-FOLD CROSS VALIDATION
# -----------------------------
print("🔄 Starting K-Fold Cross-Validation...")
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
accuracies = []
f1_scores = []

trainval_dataset = trainval_dataset.shuffle(seed=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(trainval_dataset)):
    print(f"\n===== Fold {fold + 1} / {N_FOLDS} =====")

    fold_train_ds = trainval_dataset.select(train_idx).map(tokenize_and_add_features, batched=False)
    fold_val_ds = trainval_dataset.select(val_idx).map(tokenize_and_add_features, batched=False)

    model = DistilBERTWithFeatures(MODEL_NAME, NUMERIC_FEATURES_DIM).to(DEVICE)

    k_trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=os.path.join(RESULTS_DIR, f"fold_{fold+1}"),
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            learning_rate=2e-5,
            weight_decay=0.01,
            fp16=torch.cuda.is_available(),
            report_to="none",
            save_strategy="no",
        ),
        train_dataset=fold_train_ds,
        eval_dataset=fold_val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=collate_fn
    )

    k_trainer.train()
    eval_results = k_trainer.evaluate(fold_val_ds)
    accuracies.append(eval_results["eval_accuracy"])
    f1_scores.append(eval_results["eval_f1_weighted"])
    print(f"Fold {fold+1} -> Accuracy: {eval_results['eval_accuracy']:.4f}, F1: {eval_results['eval_f1_weighted']:.4f}")

# Save K-fold metrics
print("\n📊 K-Fold Cross-Validation Results:")
kfold_results_path = os.path.join(RESULTS_DIR, "kfold_results.csv")
pd.DataFrame({
    "fold": list(range(1, N_FOLDS + 1)),
    "accuracy": accuracies,
    "f1_weighted": f1_scores
}).to_csv(kfold_results_path, index=False)
print(f"\n📄 K-Fold metrics saved to {kfold_results_path}")

# -----------------------------
# FINAL EVALUATION
# -----------------------------
print("\n📈 Evaluating final model on 1000-sample test set...")

full_trainval_ds = trainval_dataset.map(tokenize_and_add_features, batched=False)
final_model = DistilBERTWithFeatures(MODEL_NAME, NUMERIC_FEATURES_DIM).to(DEVICE)

final_trainer = Trainer(
    model=final_model,
    args=TrainingArguments(
        output_dir=os.path.join(RESULTS_DIR, "final_model"),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        report_to="none",
        save_strategy="epoch",
        save_total_limit=2
    ),
    train_dataset=full_trainval_ds,
    eval_dataset=test_dataset.map(tokenize_and_add_features, batched=False),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=collate_fn
)

final_trainer.train()
test_results = final_trainer.evaluate(test_dataset.map(tokenize_and_add_features, batched=False))

print("\n✅ Final Test Results (1000 samples):")
print(f"Accuracy: {test_results['eval_accuracy']:.4f}")
print(f"F1 (Weighted): {test_results['eval_f1_weighted']:.4f}")

# -----------------------------
# SAVE FINAL MODEL + TOKENIZER
# -----------------------------
os.makedirs(SAVE_MODEL_DIR, exist_ok=True)
save_file(final_model.state_dict(), os.path.join(SAVE_MODEL_DIR, "model.safetensors"))
tokenizer.save_pretrained(SAVE_MODEL_DIR)
final_model.bert.config.save_pretrained(SAVE_MODEL_DIR)
print(f"\n✅ Final model and tokenizer saved to {SAVE_MODEL_DIR}")

# -----------------------------
# SAVE VALIDATION METRICS
# -----------------------------
metrics_path = os.path.join(RESULTS_DIR, "validation_metrics.txt")
with open(metrics_path, "w") as f:
    f.write("Final Test Set Evaluation (1000 samples)\n")
    f.write(f"Accuracy: {test_results['eval_accuracy']:.4f}\n")
    f.write(f"F1 (Weighted): {test_results['eval_f1_weighted']:.4f}\n")
print(f"📄 Validation metrics saved to {metrics_path}")
