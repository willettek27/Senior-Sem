# =========================================
# model.py
# Use fine-tuned DistilBERT for phishing detection
# =========================================

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from bs4 import BeautifulSoup
import requests
import urllib.parse
import os
import sys

# 1️⃣ DEVICE
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🧠 Using device: {device}")

# 2️⃣ LOAD MODEL
MODEL_PATH = "./fine-tuned-models/final-distilbert-phishing"
if not os.path.isdir(MODEL_PATH):
    print(f"❌ ERROR: Model not found at {MODEL_PATH}")
    sys.exit(1)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()

print("✅ Model and tokenizer loaded successfully!")

idx_to_label = {0: "Benign", 1: "Malicious"}

# 3️⃣ UTILITIES
def normalize_url(url: str) -> str:
        parsed = urllib.parse.urlparse(url.strip())
        scheme = parsed.scheme or "https"
        netloc = parsed.netloc or parsed.path
        path = parsed.path if parsed.netloc else ""
        if path == "":
            path = "/"
        return f"{scheme}://{netloc}{path}"

def fetch_page_content(url: str) -> str:
    """Fetch visible text content from a web page."""
    try:
        response = requests.get(
            url,
            timeout=6,
            headers={"User-Agent": "Mozilla/5.0"},
            allow_redirects=True
        )
        if response.status_code != 200:
            print(f"⚠️ Non-200 response: {response.status_code}")
            return ""
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "meta", "link"]):
            tag.extract()
        text = soup.get_text(separator=" ", strip=True)
        return text[:1000]  # first 1000 chars
    except Exception as e:
        print(f"⚠️ Error fetching content: {e}")
        return ""

# 4️⃣ PREDICTION FUNCTION
def predict_model(url: str):
    url = normalize_url(url)
    page_text = fetch_page_content(url)

    combined_input = url + " " + page_text if page_text else url
    inputs = tokenizer(
        combined_input,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 🔹 Model inference
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        benign_prob, malicious_prob = probs[0].tolist()

    prediction = idx_to_label[int(malicious_prob > 0.5)]
    confidence = round(max(benign_prob, malicious_prob), 2)

    # 🔹 Return results
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



# 5️⃣ QUICK TEST
if __name__ == "__main__":
    test_url = "https://chatgpt.com"
    result = predict_model(test_url)
    print("\n🔍 Prediction Result:")
    for k, v in result.items():
        print(f"{k}: {v}")
