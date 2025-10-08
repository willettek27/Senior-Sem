# model.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from kagglehub import KaggleDatasetAdapter, dataset_load
from safe_domains import safe_domain_check

# Load tokenizer & model
model_name = "./final-malicious-url-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# Label map integer to string
idx_to_label = {0: "Benign", 1: "Defacement", 2: "Phishing", 3: "Malware"}

def predict_model(url: str):

    # Check if URL is in safe domains
    if safe_domain_check(url):
        return {
            "url": url,
            "prediction": "Benign"
        }

    # Tokenize input URL
    inputs = tokenizer(url, return_tensors="pt", truncation=True, padding=True)

    # Model inference
    with torch.no_grad():
        logits = model(**inputs).logits
        pred_idx = torch.argmax(logits, dim=1).item()

    return {
        "url": url,
        "prediction": idx_to_label[pred_idx]
    }

# Load Hugging Face dataset
def load_dataset():
   file_path = "malicious_phish.csv"

   # Map string labels to integers
   label_to_idx = {"Benign": 0, "Defacement": 1, "Phishing": 2, "Malware": 3}

   dataset = dataset_load(
       KaggleDatasetAdapter.HUGGING_FACE,
       "sid321axn/malicious-urls-dataset",
       file_path

   )
   dataset = dataset.map(lambda x: {"labels": label_to_idx[x["type"]]})

   return dataset



