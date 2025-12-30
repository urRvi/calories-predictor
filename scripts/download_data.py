import kagglehub
import shutil
import os

def download_dataset():
    print("Downloading dataset...")
    # Download latest version
    path = kagglehub.dataset_download("ruchikakumbhar/calories-burnt-prediction")
    
    print("Path to dataset files:", path)
    
    # Move files to local data directory
    target_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(target_dir, exist_ok=True)
    
    for file_name in os.listdir(path):
        full_file_name = os.path.join(path, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, target_dir)
            print(f"Copied {file_name} to {target_dir}")

if __name__ == "__main__":
    download_dataset()
