# train_model.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from kagglehub import KaggleDatasetAdapter, dataset_load
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model and tokenizer
model_name = "r3ddkahili/final-complete-malicious-url-model"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load dataset (2-label dataset mapped to 4-label space)
label_map = {"legitimate": 0, "phishing": 1}
dataset = dataset_load(
    KaggleDatasetAdapter.HUGGING_FACE,
    "shashwatwork/web-page-phishing-detection-dataset",
    "dataset_phishing.csv"
)
dataset = dataset.map(lambda x: {"labels": label_map[x["status"].lower()]})
dataset = dataset.map(lambda x: tokenizer(x["url"], truncation=True, padding="max_length", max_length=128), batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Split dataset
split = dataset.train_test_split(test_size=0.1)
train_ds, eval_ds = split["train"], split["test"]


#Compute Metrics to log eval metrics
def compute_metrics(m):
    logits = m.predictions
    labels = m.label_ids

    # phishing classes 1,2,3 into single phishing class
    phishing_scores = logits[:, 1:].sum(axis=1)
    two_class_scores = np.stack([logits[:, 0], phishing_scores], axis=1)

    preds = np.argmax(two_class_scores, axis=1)

    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")

    return {
        "accuracy": round(accuracy, 4),
        "f1_weighted": round(f1, 4)
    }

# Training setup
args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",

    # Logging
    logging_dir="./logs",
    logging_steps=100,              # log every 100 steps
 
    # Saving
    save_strategy="epoch",   # save at the end of each epoch
    save_total_limit=2,      # keep last 2 checkpoints

    # Optimization
    learning_rate=2e-5,
    weight_decay=0.01,

     # Mixed precision
    fp16=torch.cuda.is_available(),

    # Reporting
    report_to="none",
)

# Trainer and train
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("./final-malicious-url-model")

print("✅ Fine-tuning complete — model saved to ./final-malicious-url-model")

#Print Eval Metrics
print("\nRunning final evaluation on validation set...")
eval_results = trainer.evaluate()

print("\n Evaluation Complete ✅")
print("Eval results:", eval_results)

# Calculate confusion matrix
outputs = trainer.predict(eval_ds)
pred_labels = (np.argmax(outputs.predictions, axis=1)!=0).astype(int)
true_labels = (outputs.label_ids !=0).astype(int)

conf_matrix = confusion_matrix(true_labels, pred_labels)
labels = ["legitimate", "phishing"]

conf_matrix_df = pd.DataFrame(
    conf_matrix,
    index = [f"True_{label}" for label in labels],
    columns = [f"Pred_{label}" for label in labels]

)

print("\nConfusion Matrix:\n")
print(conf_matrix_df)
