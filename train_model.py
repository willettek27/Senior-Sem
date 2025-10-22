# train_model.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from kagglehub import KaggleDatasetAdapter, dataset_load
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd

# ===============================
# 1. Choose training mode
# ===============================
# "top20" → train using Top 20 features only
# "all"   → train using all features
TRAIN_MODE = "top20"


top20_features = ['google_index', 'page_rank', 'nb_hyperlinks', 'web_traffic', 'domain_age',
    'nb_www', 'phish_hints', 'ratio_intHyperlinks', 'longest_word_path', 'safe_anchor',
    'ratio_extHyperlinks', 'ratio_digits_url', 'ratio_extRedirection', 'length_url',
    'avg_word_path', 'longest_words_raw', 'length_words_raw', 'shortest_word_host',
    'length_hostname', 'char_repeat']


# 3. Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 4. Load model and tokenizer
model_name = "r3ddkahili/final-complete-malicious-url-model"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# 5. Load dataset
label_map = {"legitimate": 0, "phishing": 1}
dataset = dataset_load(
    KaggleDatasetAdapter.HUGGING_FACE,
    "shashwatwork/web-page-phishing-detection-dataset",
    "dataset_phishing.csv"
)

# If Top 20, select only those columns + "status" for labels
if TRAIN_MODE == "top20":
    dataset = dataset.map(lambda x: {k: x[k] for k in top20_features + ["status"]})

# Map labels
dataset = dataset.map(lambda x: {"labels": label_map[x["status"].lower()]})

# Tokenize
dataset = dataset.map(
    lambda x: tokenizer(x["url"], truncation=True, padding="max_length", max_length=128),
    batched=True
)

dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


# 6. Split dataset
split = dataset.train_test_split(test_size=0.1)
train_ds, eval_ds = split["train"], split["test"]


# 7. Compute Metrics
def compute_metrics(m):
    logits = m.predictions
    labels = m.label_ids

    # Merge phishing classes (if needed)
    phishing_scores = logits[:, 1:].sum(axis=1)
    two_class_scores = np.stack([logits[:, 0], phishing_scores], axis=1)
    preds = np.argmax(two_class_scores, axis=1)

    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")

    return {
        "accuracy": round(accuracy, 4),
        "f1_weighted": round(f1, 4)
    }

# 8. Training setup
# Folder names depend on TRAIN_MODE to avoid overwriting
if TRAIN_MODE == "top20":
    save_model_dir = "./final-malicious-url-model-top20"
    results_dir = "./results-top20"
else:
    save_model_dir = "./final-malicious-url-model-all"
    results_dir = "./results-all"

args = TrainingArguments(
    output_dir=results_dir,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",

    # Logging
    logging_dir="./logs",
    logging_steps=100,

    # Saving
    save_strategy="epoch",
    save_total_limit=2,

    # Optimization
    learning_rate=2e-5,
    weight_decay=0.01,

    # Mixed precision
    fp16=torch.cuda.is_available(),

    # Reporting
    report_to="none",
)

# 9. Trainer and train
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(save_model_dir)
print(f"✅ Fine-tuning complete — model saved to {save_model_dir}")


# 10. Evaluation
print("\nRunning final evaluation on validation set...")
eval_results = trainer.evaluate()
print("\nEvaluation Complete ✅")
print("Eval results:", eval_results)

# Confusion matrix
outputs = trainer.predict(eval_ds)
pred_labels = (np.argmax(outputs.predictions, axis=1) != 0).astype(int)
true_labels = (outputs.label_ids != 0).astype(int)

conf_matrix = confusion_matrix(true_labels, pred_labels)
labels = ["benign", "malicious"]

conf_matrix_df = pd.DataFrame(
    conf_matrix,
    index=[f"True_{label}" for label in labels],
    columns=[f"Pred_{label}" for label in labels]
)

print("\nConfusion Matrix:\n")
print(conf_matrix_df)
