# model.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import kagglehub
from kagglehub import KaggleDatasetAdapter
from safe_domains import safe_domain_check

# Load tokenizer & model
model_name = "r3ddkahili/final-complete-malicious-url-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# Label map 
label_map = {0: "Benign", 1: "Defacement", 2: "Phishing", 3: "Malware"}

def predict_model(url: str):

    if safe_domain_check(url):
        return {
            "url": url,
            "prediction": "Benign"
        }

    inputs = tokenizer(url, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_idx = torch.argmax(logits, dim=1).item()
        probs = torch.softmax(logits, dim=-1)[0]
        confidence = probs[pred_idx].item()

    pred_label = label_map[pred_idx]

    return {
        "url": url,
        "prediction": pred_label
    }

def load_dataset():
   file_path = "malicious_phish.csv"
   dataset = kagglehub.load_dataset(
       KaggleDatasetAdapter.PANDAS,
       "sid321axn/malicious-urls-dataset",
       file_path
   )
   return dataset
