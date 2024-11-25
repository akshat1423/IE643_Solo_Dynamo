import os
import subprocess

# Define Kaggle datasets and their target directories
datasets = {
    "rajat95gupta/hazing-images-dataset-cvpr-2019": "../datasets/hazing-images-dataset-cvpr-2019",
    "balraj98/indoor-training-set-its-residestandard": "../datasets/indoor-training-set-its-residestandard",
    "brunobelloni/synthetic-objective-testing-set-sots-reside": "../datasets/synthetic-objective-testing-set-sots-reside"
}

# Ensure Kaggle API is installed
try:
    subprocess.run(["kaggle", "--version"], check=True, stdout=subprocess.DEVNULL)
except FileNotFoundError:
    raise Exception("Kaggle CLI not found. Please install it and set up your Kaggle credentials.")

# Download and extract datasets
for dataset, target_dir in datasets.items():
    os.makedirs(target_dir, exist_ok=True)
    print(f"Downloading and extracting {dataset} to {target_dir}...")
    subprocess.run(["kaggle", "datasets", "download", "-d", dataset, "-p", target_dir], check=True)
    zip_files = [f for f in os.listdir(target_dir) if f.endswith(".zip")]
    for zip_file in zip_files:
        zip_path = os.path.join(target_dir, zip_file)
        subprocess.run(["unzip", "-o", zip_path, "-d", target_dir], check=True)
        os.remove(zip_path)
    print(f"Dataset {dataset} is ready in {target_dir}.")
