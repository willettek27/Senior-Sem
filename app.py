# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

dataset_path = "data/malicious_phish.csv"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset file not found at {dataset_path}")

dataset = pd.read_csv(dataset_path)

dataset_dict = dict(zip(dataset['url'], dataset['type']))

# Load the model and tokenizer
model_name = "r3ddkahili/final-complete-malicious-url-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Mapping prediction to labels
label_map = {0: "Benign", 1: "Defacement", 2: "Phishing", 3: "Malware"}

def preprocess_url(url: str) -> str:
    url_clean = url.lower().strip()
    url_clean = url_clean.replace("https://", "").replace("http://", "")
    if url_clean.startswith("www."):
        url_clean = url_clean[4:]
    return url_clean


@app.route("/")
def home():
    return {"message": "Hello World! API running 🎉"}

@app.route("/predict", methods=["POST"])
def prediction():
    query = request.get_json()
    url = query.get("url", "")

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    preprocessed_url = preprocess_url(url)

    # Check Kagglehub dataset first
    if preprocessed_url in dataset_dict:
        return jsonify({
            "url": url,
            "prediction": dataset_dict[preprocessed_url],
        })

    # If not in dataset, use AI model
    entry = tokenizer([preprocessed_url], return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**entry)
    
    predicted_class_id = torch.argmax(outputs.logits, dim=1).item()
    predicted_label = label_map.get(predicted_class_id, "Unknown")

    return jsonify({
        "url": url,
        "prediction": predicted_label,
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)