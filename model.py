# model.py 

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from kagglehub import KaggleDatasetAdapter, dataset_load
from safe_domains import safe_domain_check
import os
import sys
import requests
from bs4 import BeautifulSoup
import urllib.parse

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Path to your fine-tuned model
model_name = "./fine-tuned-models/final-malicious-url-model-all"

# ✅ Check if folder exists
if not os.path.isdir(model_name):
    print(f"❌ ERROR: Model folder not found at {model_name}")
    sys.exit(1)


# Load tokenizer & model
model_name = "./fine-tuned-models/final-malicious-url-model-all"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
model.eval()

print("✅ Model and tokenizer loaded successfully!")
# # Model predicts 4 classes, but we merge malware, phishing, and defacement into "Maliccious" for inference
idx_to_label = {
    0: "Benign",
    1: "Malicious", 
    
}
def normalize_url(url: str) -> str:
    """Ensure URL has a consistent scheme, domain, and trailing slash."""
    parsed = urllib.parse.urlparse(url.strip())
    
    # Add scheme if missing
    scheme = parsed.scheme or "https"
    
    # Use netloc (domain) + path
    netloc = parsed.netloc or parsed.path
    path = parsed.path if parsed.netloc else ""
    
    # Always add trailing slash if path is empty
    if path == "":
        path = "/"
    
    normalized = f"{scheme}://{netloc}{path}"
    return normalized


def fetch_page_content(url: str) -> str:
    """Fetch HTML content safely and extract visible text."""
    try:
        response = requests.get(
            url, 
            timeout=6, 
            headers={'User-Agent': 'Mozilla/5.0'},
            allow_redirects=True
        )
        final_url = response.url

   # Stop if too many redirects
        if len(response.history) > 3:
            print("⚠️ Too many redirects, skipping content.")
            return "", final_url

    # Extract visible text
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "meta", "link"]):
            tag.extract()
        text = soup.get_text(separator=" ", strip=True)
        
        return text[:1000], final_url  # Limit to first 1000 chars
    
    except Exception as e:
        print(f"⚠️ Error fetching content: {e}")
        return "", url

def predict_model(url: str):
    url = normalize_url(url)

    # Check if URL is in safe domains
    if safe_domain_check(url):
        return {
            "url": url,
            "prediction": "Benign",
            "confidence": 1.0,
            "used_content": False
        }

    # Fetch web page content
    page_text, final_url = fetch_page_content(url)
    url = normalize_url(final_url) 
    # Combine the URL + webpage text for better prediction
    combined_input =  url + " " + page_text if page_text else url

    # Tokenize combined input (✅ fix was here)
    inputs = tokenizer(combined_input, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Model inference
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        benign_prob = probs[0, 0].item()
        malicious_prob = probs[0, 1].item()

    # Threshold-based prediction
    MALICIOUS_THRESHOLD = 0.70
    if malicious_prob >= MALICIOUS_THRESHOLD:
        prediction = "Malicious"
        confidence = malicious_prob
    elif benign_prob >= (1.0 - MALICIOUS_THRESHOLD):
        prediction = "Benign"
        confidence = benign_prob
    else:
        prediction = "Unknown"
        confidence = max(benign_prob, malicious_prob)

    return {
        "url": url,
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "raw_probs": {
            "benign": round(benign_prob, 4),
            "malicious": round(malicious_prob, 4)
        },
        "used_content": len(page_text) > 0
    }


# Load Hugging Face dataset
def load_dataset():
   file_path = "dataset_phishing.csv"

   # 2-Label mapping
   label_to_idx = {
        "legitimate": 0,  # Benign 
        "phishing": 1     # Malicious
    }
   
   dataset = dataset_load(
       KaggleDatasetAdapter.HUGGING_FACE,
       "shashwatwork/web-page-phishing-detection-dataset",
       file_path
   )

   dataset = dataset.map(lambda x: {"labels": label_to_idx[x["status"].lower()]})
   
   return dataset


if __name__ == "__main__":
    # Load dataset
    dataset = load_dataset()

    # Test the first URL from dataset
    test_sample = dataset[1]
    url = test_sample["url"]
    correct_label_idx = test_sample["labels"]
    correct_label = idx_to_label[correct_label_idx]

    # Get prediction
    result = predict_model(url)
    predicted_label = result["prediction"]

    # Print prediction and correctness
    print("\nTest URL:", url)
    print("Correct label:", correct_label)
    print("Predicted label:", predicted_label)
    print("Used content:", result["used_content"])
    if predicted_label == correct_label:
        print("Prediction is CORRECT ✅")
    else:
        print("Prediction is INCORRECT ❌") 