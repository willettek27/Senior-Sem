# =========================================
# model.py
# Use fine-tuned DistilBERT + numeric features for phishing detection
# =========================================

import os
import sys
import torch
from transformers import AutoTokenizer, DistilBertModel
from extract_features import extract_features
import torch.nn as nn
import urllib.parse as urlparse
import requests
from bs4 import BeautifulSoup

# DEVICE

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🧠 Using device: {device}")

# -----------------------------
# LOAD MODEL
# -----------------------------
MODEL_PATH = "./fine-tuned-models/final-distilbert-phishing"
if not os.path.isdir(MODEL_PATH):
    print(f"❌ ERROR: Model not found at {MODEL_PATH}")
    sys.exit(1)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Define the same custom model class as in training
class DistilBERTWithFeatures(nn.Module):
    def __init__(self, base_model_name, num_features):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(base_model_name)
        self.num_features_fc = nn.Linear(num_features, 32)
        self.classifier = nn.Linear(self.bert.config.hidden_size + 32, 2)

    def forward(self, input_ids=None, attention_mask=None, numeric_features=None, labels=None):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = bert_outputs.last_hidden_state[:, 0, :]
        num_feat_proj = self.num_features_fc(numeric_features)
        combined = torch.cat([cls_emb, num_feat_proj], dim=1)
        logits = self.classifier(combined)
        return logits

NUMERIC_FEATURES_DIM = 89
model = DistilBERTWithFeatures("distilbert-base-uncased", NUMERIC_FEATURES_DIM).to(device)
model.load_state_dict(torch.load(os.path.join(MODEL_PATH, "pytorch_model.bin"), map_location=device))
model.eval()
# -----------------------------
idx_to_label = {0: "Benign", 1: "Malicious"}

# -----------------------------
# UTILITIES
# -----------------------------
def normalize_url(url: str) -> str:
    parsed = urlparse.urlparse(url.strip())
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc or parsed.path
    path = parsed.path if parsed.netloc else ""
    if path == "":
        path = "/"
    return f"{scheme}://{netloc}{path}"

def fetch_page_content(url: str) -> str:
    """Fetch visible text content from a web page."""
    try:
        response = requests.get(url, timeout=6, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code != 200:
            return ""
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "meta", "link"]):
            tag.extract()
        text = soup.get_text(separator=" ", strip=True)
        return text[:1000]
    except:
        return ""

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_model(url: str):
    url = normalize_url(url)
    page_text = fetch_page_content(url)

    # Extract numeric features
    features = extract_features({"url": url, "content": page_text})
    numeric_features = torch.tensor([list(features.values())[1:]], dtype=torch.float).to(device)  # skip original "url" field

    # Tokenize text
    combined_input = url + " " + page_text if page_text else url
    inputs = tokenizer(
        combined_input,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Model inference
    with torch.no_grad():
        logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], numeric_features=numeric_features)
        probs = torch.softmax(logits, dim=1)
        benign_prob, malicious_prob = probs[0].tolist()

    prediction = idx_to_label[int(malicious_prob > 0.5)]
    confidence = round(max(benign_prob, malicious_prob), 2)

    return {
        "url": url,
        "prediction": prediction,
        "confidence": confidence,
        "raw_probs": {
            "benign": round(benign_prob, 4),
            "malicious": round(malicious_prob, 4)
        },
        "used_content": len(page_text) > 0
    }

# -----------------------------
# QUICK TEST
# -----------------------------
if __name__ == "__main__":
    test_url = "https://chatgpt.com"
    result = predict_model(test_url)
    print("\n🔍 Prediction Result:")
    for k, v in result.items():
        print(f"{k}: {v}")