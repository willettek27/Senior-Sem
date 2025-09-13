# model.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the model and tokenizer
model_name = "r3ddkahili/final-complete-malicious-url-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Example URL
url = "http://example.com/login"

# Tokenize and predict
inputs = tokenizer(url, return_tensors="pt", truncation=True, padding=True, max_length=128)
with torch.no_grad():
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits).item()

# Mapping prediction to labels
label_map = {0: "Benign", 1: "Defacement", 2: "Phishing", 3: "Malware"}
print(f"Prediction: {label_map[prediction]}")
