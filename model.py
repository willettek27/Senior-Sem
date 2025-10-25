# =========================================
# model.py
# Use fine-tuned DistilBERT + numeric features for phishing detection
# =========================================

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import urllib.parse as urlparse
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, DistilBertModel
from safetensors.torch import load_file
from extract_features import extract_features

# -----------------------------
# DEVICE
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🧠 Using device: {DEVICE}")

# -----------------------------
# LOAD MODEL + TOKENIZER
# -----------------------------
MODEL_PATH = "./fine-tuned-models/final-distilbert-phishing"
if not os.path.isdir(MODEL_PATH):
    print(f"❌ ERROR: Model not found at {MODEL_PATH}")
    sys.exit(1)

MODEL_NAME = "distilbert-base-uncased"
NUMERIC_FEATURES_DIM = 87  # must match extract_features.py

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

class DistilBERTWithFeatures(nn.Module):
    def __init__(self, base_model_name, num_features):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(base_model_name)
        self.num_features_fc = nn.Linear(num_features, 32)
        self.classifier = nn.Linear(self.bert.config.hidden_size + 32, 2)

    def forward(self, input_ids=None, attention_mask=None, numeric_features=None):
        cls_emb = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        num_feat_proj = self.num_features_fc(numeric_features)
        logits = self.classifier(torch.cat([cls_emb, num_feat_proj], dim=1))
        return logits

# Load model weights
model = DistilBERTWithFeatures(MODEL_NAME, NUMERIC_FEATURES_DIM).to(DEVICE)

model_file = os.path.join(MODEL_PATH, "model.safetensors")
if not os.path.exists(model_file):
    print(f"❌ ERROR: Missing model file at {model_file}")
    sys.exit(1)

state_dict = load_file(model_file)
model.load_state_dict(state_dict)
model.eval()

print(f"✅ Loaded fine-tuned model from {MODEL_PATH}")

# -----------------------------
# LABEL MAPPING
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
    if not path:
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
    except Exception:
        return ""

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_model(url: str):
    url = normalize_url(url)
    page_text = fetch_page_content(url)

    # Extract numeric features (same 87-dim order as training)
    numeric_features = np.array(extract_features({"url": url}), dtype=np.float32)
    numeric_features = torch.tensor([numeric_features], dtype=torch.float).to(DEVICE)

    # Tokenize combined input
    combined_text = url + " " + page_text if page_text else url
    inputs = tokenizer(
        combined_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Inference
    with torch.no_grad():
        logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            numeric_features=numeric_features
        )
        probs = torch.softmax(logits, dim=1)
        benign_prob, malicious_prob = probs[0].tolist()

    prediction = idx_to_label[int(malicious_prob > 0.5)]
    confidence = round(max(benign_prob, malicious_prob), 3)

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
    test_url = "https://chat.openai.com"
    result = predict_model(test_url)
    print("\n🔍 Prediction Result:")
    for k, v in result.items():
        print(f"{k}: {v}")
