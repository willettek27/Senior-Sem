# =========================================
# download_dataset.py
# Download and move phishing dataset to /data
# =========================================

''' 
model.py automatically downloads the dataset, 
but if you want to manually download it, 
use this script. 
'''


import os
import shutil
import kagglehub

# Download latest version
print("📥 Downloading dataset...")
path = kagglehub.dataset_download("shashwatwork/web-page-phishing-detection-dataset")

# Ensure data folder exists
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# Move downloaded files to /data
print("📂 Moving files to data/ folder...")
for item in os.listdir(path):
    src = os.path.join(path, item)
    dst = os.path.join(data_dir, item)
    if os.path.isdir(src):
        shutil.move(src, dst)
    else:
        shutil.move(src, dst)

print(f"✅ Dataset ready in: {os.path.abspath(data_dir)}")
