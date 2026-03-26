import os
import glob
import h5py
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset

def build_hdf5_database(data_dir, output_h5_path, strict=True):
    """
    Builds an HDF5 database.
    strict=True: Maps chb21 to chb01 to prevent data leakage [cite: 820-822, 1008-1014].
    strict=False: Keeps them separate for the intentional leakage experiment.
    """
    print(f"Building HDF5 Database (Strict Mode: {strict})...")
    parquet_files = sorted(glob.glob(os.path.join(data_dir, '*_metadata_1.parquet')))
    total_windows = sum(len(pd.read_parquet(p, engine='fastparquet')) for p in parquet_files)
    
    PATIENT_MAP = { "chb17a": "chb17", "chb17b": "chb17", "chb17c": "chb17" }
    if strict:
        PATIENT_MAP["chb21"] = "chb01"
    
    with h5py.File(output_h5_path, 'w') as h5f:
        dataset_X = h5f.create_dataset("X", shape=(total_windows, 21, 128), dtype='float32', chunks=(1024, 21, 128))
        dataset_y = h5f.create_dataset("y", shape=(total_windows,), dtype='int64')
        dataset_pid = h5f.create_dataset("patient_id", shape=(total_windows,), dtype=h5py.string_dtype(encoding='utf-8'))
        dataset_interval = h5f.create_dataset("global_interval", shape=(total_windows,), dtype='int64')
        
        current_idx = 0
        for p_file in tqdm(parquet_files, desc="Writing to HDF5"):
            npz_file = p_file.replace('_metadata_1.parquet', '_EEGwindow_1.npz')
            if not os.path.exists(npz_file): continue
            
            meta = pd.read_parquet(p_file, engine='fastparquet')
            data = np.load(npz_file, allow_pickle=True)
            
            pid_raw = meta['filename'].apply(lambda x: x.split('_')[0]).values
            pid_clean = [PATIENT_MAP.get(p, p) for p in pid_raw] 
            
            batch_size = len(meta)
            dataset_X[current_idx : current_idx + batch_size] = data[data.files[0]].astype(np.float32)
            dataset_y[current_idx : current_idx + batch_size] = meta['class'].values
            dataset_pid[current_idx : current_idx + batch_size] = pid_clean
            dataset_interval[current_idx : current_idx + batch_size] = meta['global_interval'].values
            current_idx += batch_size
    print("HDF5 Database build complete.")

class UniversalEEGDataset(Dataset):

    def __init__(self, data_source, indices, y_labels, intervals, model_type='CNN', seq_length=5):
        self.indices = indices
        self.y_labels = y_labels
        self.intervals = intervals
        self.model_type = model_type
        self.seq_length = seq_length
        self.is_h5 = isinstance(data_source, str)
        
        if self.is_h5:
            self.h5_path = data_source
            self.h5_file = None
        else:
            self.X_data = data_source

    def __len__(self):
        return len(self.indices)

    def _get_x_data(self, idx):
        if self.is_h5:
            if self.h5_file is None:
                self.h5_file = h5py.File(self.h5_path, 'r')
            return self.h5_file['X'][idx]
        return self.X_data[idx]

    def __getitem__(self, idx):
        # actual_idx represents the actual index that maps back to the original global dataset.
        actual_idx = self.indices[idx]
        
        if self.model_type == 'CNN':
            x_tensor = torch.tensor(self._get_x_data(actual_idx), dtype=torch.float32)
            # use the actual_idx to extract the global labels and intervals
            y_tensor = torch.tensor(self.y_labels[actual_idx], dtype=torch.long)
            interval = self.intervals[actual_idx]
            
            return x_tensor, y_tensor, interval, actual_idx
            
        elif self.model_type == 'CNN_LSTM':
            # Force LSTM to continuously extract (backtrack over a sequence of seq_length windows) along the timeline
            start_idx = actual_idx - self.seq_length + 1
            
            target_interval = self.intervals[actual_idx]
            valid_seq_indices = []
            
            for i in range(start_idx, actual_idx + 1):
                if i >= 0 and self.intervals[i] == target_interval:
                    valid_seq_indices.append(i)
                else:
                    valid_seq_indices.append(actual_idx) 
            
            if self.is_h5:
                if self.h5_file is None: 
                    self.h5_file = h5py.File(self.h5_path, 'r')
                x_seq_list = [self.h5_file['X'][i] for i in valid_seq_indices]
                x_tensor = torch.tensor(np.array(x_seq_list), dtype=torch.float32).unsqueeze(1)
            else:
                x_tensor = torch.tensor(self.X_data[valid_seq_indices], dtype=torch.float32).unsqueeze(1)
                
            y_tensor = torch.tensor(self.y_labels[actual_idx], dtype=torch.long)
            interval = target_interval
            
            return x_tensor, y_tensor, interval, actual_idx
    
    def __del__(self):
        if self.is_h5 and self.h5_file is not None:
            self.h5_file.close()