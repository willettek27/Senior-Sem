# download_dataset.py

''' 
model.py automatically downloads the dataset, 
but if you want to manually download it, 
use this script. 
'''

import kagglehub

# Download latest version
path = kagglehub.dataset_download("sid321axn/malicious-urls-dataset")

print("Path to dataset files:", path)