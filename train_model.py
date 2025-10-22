# =========================================
# train_model.py
# Fine-tune DistilBERT on phishing dataset (URL + content)
# =========================================

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from datasets import load_dataset
import numpy as np
import pandas as pd
import torch
import os

# 1️⃣ CONFIGURATION
MODEL_NAME = "distilbert-base-uncased"
SAVE_MODEL_DIR = "./fine-tuned-models/final-distilbert-phishing"
RESULTS_DIR = "./results-distilbert"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🧠 Using device: {DEVICE}")

# 2️⃣ LOAD DATASET
# Make sure your dataset CSV has columns: 'url', 'content', 'status'
# status = "legitimate" or "phishing"
DATA_PATH = "./dataset_phishing.csv"
assert os.path.exists(DATA_PATH), f"❌ Dataset not found at {DATA_PATH}"

df = pd.read_csv(DATA_PATH)

# Map labels to integers
label_map = {"legitimate": 0, "phishing": 1}
df["labels"] = df["status"].str.lower().map(label_map)

# Combine URL + content
df["text"] = df["url"].fillna("") + " " + df["content"].fillna("")

# Convert to Hugging Face Dataset
from datasets import Dataset
dataset = Dataset.from_pandas(df[["text", "labels"]])

# 3️⃣ SPLIT DATASET
split = dataset.train_test_split(test_size=0.1, seed=42)
train_ds, eval_ds = split["train"], split["test"]
print(f"✅ Dataset split: {len(train_ds)} train / {len(eval_ds)} eval")

# 4️⃣ TOKENIZATION
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_ds = train_ds.map(tokenize_function, batched=True)
eval_ds = eval_ds.map(tokenize_function, batched=True)
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
eval_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 5️⃣ MODEL SETUP
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)

# 6️⃣ METRICS FUNCTION
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }

# 7️⃣ TRAINING ARGUMENTS
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

# 8️⃣ TRAINER
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 9️⃣ TRAIN & SAVE
print("🚀 Training started...")
trainer.train()
trainer.save_model(SAVE_MODEL_DIR)
tokenizer.save_pretrained(SAVE_MODEL_DIR)
print(f"✅ Fine-tuning complete — model saved to {SAVE_MODEL_DIR}")

# 🔟 EVALUATION
print("\n📊 Running evaluation on validation set...")
eval_results = trainer.evaluate()
print("\n✅ Evaluation complete:")
print(eval_results)

# Confusion Matrix
outputs = trainer.predict(eval_ds)
pred_labels = np.argmax(outputs.predictions, axis=1)
true_labels = outputs.label_ids
conf_matrix = confusion_matrix(true_labels, pred_labels)
labels = ["Benign", "Malicious"]
conf_matrix_df = pd.DataFrame(
    conf_matrix,
    index=[f"True_{l}" for l in labels],
    columns=[f"Pred_{l}" for l in labels]
)
print("\nConfusion Matrix:\n", conf_matrix_df)
