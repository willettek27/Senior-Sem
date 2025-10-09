# model.py 

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from kagglehub import KaggleDatasetAdapter, dataset_load
from safe_domains import safe_domain_check

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load tokenizer & model
model_name = "r3ddkahili/final-complete-malicious-url-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
model.eval()

# # Model predicts 4 classes, but we merge malware and defacement into "Phishing" for inference
idx_to_label = {
    0: "Benign", 
    1: "Phishing",
    2: "Phishing",  #Defacement merged into Phishing
    3: "Phishing"   #Malware merged into Phishing
}

def predict_model(url: str):

    # Check if URL is in safe domains
    if safe_domain_check(url):
        return {
            "url": url,
            "prediction": "Benign"
        }

    # Tokenize input URL & moves to device
    inputs = tokenizer(url, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Model inference
    with torch.no_grad():
        logits = model(**inputs).logits
        pred_idx = torch.argmax(logits, dim=1).item()

    #Fallback to Unknown if index is unexpected
    prediction = idx_to_label.get(pred_idx,"Unknown")

    return {
        "url": url,
        "prediction": prediction,
    }

# Load Hugging Face dataset
def load_dataset():
   file_path = "dataset_phishing.csv"

   # 2-Label mapping
   label_to_idx = {
        "legitimate": 0,  # Benign 
        "phishing": 1     #Anything malicious: phishing
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
    test_sample = dataset[0]
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
    if predicted_label == correct_label:
        print("Prediction is CORRECT ✅")
    else:
        print("Prediction is INCORRECT ❌") 