# app.py

from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

# Load the model and tokenizer
model_name = "r3ddkahili/final-complete-malicious-url-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Mapping prediction to labels
label_map = {0: "Benign", 1: "Defacement", 2: "Phishing", 3: "Malware"}

@app.route("/")
def home():
    return {"message": "Hello World! API running 🎉"}

@app.route("/predict", methods=["POST"])
def prediction():
    query = request.get_json()
    url = query.get("url", "")

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    # Tokenize the input URL
    entry = tokenizer(url, return_tensors="pt", truncation=True, padding=True)

    # Get model predictions
    with torch.no_grad():
        outputs = model(**entry)

    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()
    predicted_label = label_map[predicted_class_id]

    return jsonify({"url": url, "prediction": predicted_label}) 