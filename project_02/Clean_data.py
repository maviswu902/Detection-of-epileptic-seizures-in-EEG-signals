import os
import glob
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm

def clean_eeg_data(source_dir, target_dir):
    print(f"Starting data cleaning pipeline...\nSource: {source_dir}\nTarget: {target_dir}\n")
    
    # Create the target folder if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Find all npz files
    npz_files = sorted(glob.glob(os.path.join(source_dir, '*_EEGwindow_1.npz')))
    
    if not npz_files:
        print("No npz files found in the source directory. Please check the path!")
        return

    for npz_path in tqdm(npz_files, desc="Cleaning Progress"):
        filename = os.path.basename(npz_path)
        target_npz_path = os.path.join(target_dir, filename)
        
        # Corresponding parquet filename
        parquet_filename = filename.replace('_EEGwindow_1.npz', '_metadata_1.parquet')
        parquet_path = os.path.join(source_dir, parquet_filename)
        target_parquet_path = os.path.join(target_dir, parquet_filename)

        # 1. Load the raw data
        data = np.load(npz_path, allow_pickle=True)
        X_raw = data[data.files[0]].astype(np.float32)
        
        # Physical Clipping - Flatten extreme flashbang artifacts
        X_clean = np.clip(X_raw, -300.0, 300.0)
        
        # Z-score Normalization
        mean_val = np.mean(X_clean, axis=(0, 2), keepdims=True)
        std_val = np.std(X_clean, axis=(0, 2), keepdims=True)
        X_clean = (X_clean - mean_val) / (std_val + 1e-6)
        
        # 2. Save the cleaned data
        np.savez_compressed(target_npz_path, X_clean)
        
        # 3. Copy the corresponding parquet
        if os.path.exists(parquet_path):
            shutil.copy2(parquet_path, target_parquet_path)
        else:
            print(f"Warning: Cannot find corresponding parquet file: {parquet_filename}")

    print("\n Cleaning complete! Let's do a health check on the first cleaned file:")
    test_data = np.load(os.path.join(target_dir, os.path.basename(npz_files[0])), allow_pickle=True)
    X_test = test_data[test_data.files[0]]
    
    print(f"Checking file: {os.path.basename(npz_files[0])}")
    print(f"New Mean: {np.mean(X_test):.6f} (Should be extremely close to 0)")
    print(f"New Std: {np.std(X_test):.6f} (Should be very close to 1.0)")
    print(f"New Max: {np.max(X_test):.2f} (No longer terrifyingly high numbers)")
    print(f"New Min: {np.min(X_test):.2f}")

if __name__ == "__main__":
    RAW_DATA_DIR = "/hhome/ricse/Epilepsy/"
    CLEAN_DATA_DIR = "/hhome/ricse02/project_02/Epilepsy/"
    
    clean_eeg_data(RAW_DATA_DIR, CLEAN_DATA_DIR)