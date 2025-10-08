# model.py 

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from kagglehub import KaggleDatasetAdapter, dataset_load
from safe_domains import safe_domain_check

# Load tokenizer & model
model_name = "r3ddkahili/final-complete-malicious-url-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# Label map integer to string
idx_to_label = {0: "Benign", 1: "Phishing"}

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

    prediction = idx_to_label[pred_idx]

    return {
        "url": url,
        "prediction": prediction,
    }

# Load Hugging Face dataset
def load_dataset():
   file_path = "dataset_phishing.csv"

   # Map string labels to integers
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