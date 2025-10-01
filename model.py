# model.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F


# Load the model and tokenizer
model_name = "r3ddkahili/final-complete-malicious-url-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# Mapping prediction to labels
label_map = {0: "Benign", 1: "Defacement", 2: "Phishing", 3: "Malware"}

thresholds = {
    "Defacement": 0.75,
    "Malware": 0.85,
    "Phishing": 0.9,
}

safe_domains = ["wikipedia.org", "github.com", "google.com", "stackoverflow.com"]

def predict_model(url: str):
    if any(domain in url for domain in safe_domains):
        return {
            "url": url,
            "prediction": "Benign",
        }

    inputs = tokenizer([url], return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1)[0] 
       
    passed = {i: probs[i] for i, label in label_map.items()
        if label != "Benign" and probs[i] >= thresholds.get(label, 0)}

    pred_idx = max(passed, key=passed.get) if passed else 0

# === Results ===
    return {
        "url": url,
        "prediction": label_map[pred_idx] 
    }
