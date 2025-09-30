# model.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import pandas as pd
import os

#Dataset
dataset_path = "data/malicious_phish.csv"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset file not found at {dataset_path}")

dataset = pd.read_csv(dataset_path)
dataset_dict = dict(zip(dataset["url"].str.strip().str.lower(), dataset["type"]))

# Safe domains
safe_domains = ["wikipedia.org", "google.com", "github.com", "stackoverflow.com", "example.com"]

# Load the model and tokenizer
model_name = "r3ddkahili/final-complete-malicious-url-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# Mapping prediction to labels
label_map = {0: "Benign", 1: "Defacement", 2: "Phishing", 3: "Malware"}
threshold = 0.6

# Example URL
url = "https://www.wikipedia.org"
preprocessed_url = url.strip().lower()

# Check Kagglehub dataset first
if preprocessed_url in dataset_dict:
    label = dataset_dict[preprocessed_url]
    confidence = 1.0 if label.lower() == "benign" else 0.9
    result = {
        "url": url,
        "prediction": label,
        "source": "dataset",
        "confidence": confidence
    }

# Safe domain
elif any(domain in preprocessed_url for domain in safe_domains):
    result = {
        "url": url,
        "prediction": "Benign",
        "source": "safe_domains",
        "confidence": 1.0
    }

# If not in dataset, use AI model
else:
    # Tokenize and predict
    inputs = tokenizer([url], return_tensors="pt", truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)[0]

    prediction = torch.argmax(probs).item()
    confidence = probs[prediction].item()
    pred_label = "Benign" if confidence < threshold else label_map.get(prediction, "Unknown")


# === Results ===
    result = {
        "url": url,
        "prediction": pred_label,
        "source": "model",
        "confidence": round(confidence, 4)
    }

#Result on separate lines
for k, v in result.items():
    print(f"{k}: {v}")