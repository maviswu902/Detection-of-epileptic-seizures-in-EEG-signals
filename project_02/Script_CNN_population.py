import os
import glob
import json
import h5py
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import torch.nn.functional as F

# 1. HDF5 Database Builder
def build_hdf5_database(data_dir, output_h5_path):
    print("Building HDF5 database...")
    parquet_files = sorted(glob.glob(os.path.join(data_dir, '*_metadata_1.parquet')))
    if not parquet_files: raise ValueError(f"No parquet files found in {data_dir}!")
    
    total_windows = sum(len(pd.read_parquet(p, engine='fastparquet')) for p in parquet_files)
    
    with h5py.File(output_h5_path, 'w') as h5f:
        dataset_X = h5f.create_dataset("X", shape=(total_windows, 21, 128), dtype='float32', chunks=(1024, 21, 128))
        dataset_y = h5f.create_dataset("y", shape=(total_windows,), dtype='int64')
        dt_str = h5py.string_dtype(encoding='utf-8')
        dataset_pid = h5f.create_dataset("patient_id", shape=(total_windows,), dtype=dt_str)
        
        current_idx = 0
        for p_file in tqdm(parquet_files, desc="Writing to HDF5"):
            npz_file = p_file.replace('_metadata_1.parquet', '_EEGwindow_1.npz')
            if not os.path.exists(npz_file): continue
            meta = pd.read_parquet(p_file, engine='fastparquet')
            data = np.load(npz_file, allow_pickle=True)
            X_batch = data[data.files[0]].astype(np.float32)
            y_batch = meta['class'].values
            pid_batch = meta['filename'].apply(lambda x: x.split('_')[0]).values
            
            batch_size = len(y_batch)
            dataset_X[current_idx : current_idx + batch_size] = X_batch
            dataset_y[current_idx : current_idx + batch_size] = y_batch
            dataset_pid[current_idx : current_idx + batch_size] = pid_batch
            current_idx += batch_size
    print(f"HDF5 Database saved to: {output_h5_path}")

# 2. PyTorch HDF5 Dataset Wrapper
class H5EEGDataset(Dataset):
    def __init__(self, h5_path, indices):
        self.h5_path = h5_path
        self.indices = indices
        self.h5_file = None
        with h5py.File(self.h5_path, 'r') as f:
            self.y_cache = f['y'][self.indices]
            
    def __len__(self): return len(self.indices)
    
    def __getitem__(self, idx):
        if self.h5_file is None: self.h5_file = h5py.File(self.h5_path, 'r')
        X = self.h5_file['X'][self.indices[idx]]
        y = self.y_cache[idx]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# 3. Optimized CNN Model
class BaselineCNN(nn.Module):
    def __init__(self, num_channels=21, time_steps=128):
        super(BaselineCNN, self).__init__()
        # 16 Spatial Filters
        self.data_fusion = nn.Conv1d(num_channels, 16, kernel_size=1)
        self.conv1 = nn.Sequential(nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool1d(4, 4))
        self.conv2 = nn.Sequential(nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool1d(4, 4))
        self.conv3 = nn.Sequential(nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool1d(4, 4))
        self.flatten = nn.Flatten()
        
        with torch.no_grad():
            dummy = torch.zeros(1, num_channels, time_steps)
            dummy = self.conv3(self.conv2(self.conv1(self.data_fusion(dummy))))
            self.feature_size = dummy.view(1, -1).size(1)
            
        # Dropout(0.5) to prevent overfitting on static features
        self.fc1 = nn.Sequential(nn.Linear(self.feature_size, 128), nn.ReLU(), nn.Dropout(0.5))
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.data_fusion(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)

# 4. Training Loop
def train_and_evaluate(model, train_loader, val_loader, test_loader, epochs=15, lr=0.001, device='cpu', output_dir='.', patience=5):
    criterion = nn.CrossEntropyLoss(); optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device); history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf'); epochs_no_improve = 0
    best_model_path = os.path.join(output_dir, 'best_cnn_model.pth')

    for epoch in range(epochs):
        model.train(); train_loss, correct_train, total_train = 0.0, 0, 0
        for batch_X, batch_y in tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1} Train"):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad(); outputs = model(batch_X); loss = criterion(outputs, batch_y)
            loss.backward(); optimizer.step()
            train_loss += loss.item() * batch_X.size(0); total_train += batch_y.size(0)
            correct_train += (torch.max(outputs, 1)[1] == batch_y).sum().item()
        
        model.eval(); val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X); loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0); total_val += batch_y.size(0)
                correct_val += (torch.max(outputs, 1)[1] == batch_y).sum().item()
        
        avg_val_loss = val_loss / total_val
        history['epoch'].append(epoch+1); history['train_loss'].append(train_loss/total_train)
        history['val_loss'].append(avg_val_loss); history['train_acc'].append(100*correct_train/total_train); history['val_acc'].append(100*correct_val/total_val)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss/total_train:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {100*correct_val/total_val:.2f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss; epochs_no_improve = 0; torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience: print("Early stopping triggered."); break

    model.load_state_dict(torch.load(best_model_path)); model.eval()
    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for batch_X, batch_y in tqdm(test_loader, leave=False, desc="Testing"):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            y_probs.extend(F.softmax(outputs, dim=1)[:, 1].cpu().numpy())
            y_true.extend(batch_y.cpu().numpy()); y_pred.extend(torch.max(outputs, 1)[1].cpu().numpy())

    pd.DataFrame(history).to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f: f.write(classification_report(y_true, y_pred, target_names=['Non-seizure (0)', 'Seizure (1)']))
    plt.figure(figsize=(6,5)); sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix'); plt.savefig(os.path.join(output_dir, 'confusion_matrix.png')); plt.close()
    fpr, tpr, _ = roc_curve(y_true, y_probs); roc_auc = auc(fpr, tpr)
    plt.figure(); plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}'); plt.legend(); plt.savefig(os.path.join(output_dir, 'roc_curve.png')); plt.close()
    precision, recall, _ = precision_recall_curve(y_true, y_probs); pr_auc = average_precision_score(y_true, y_probs)
    plt.figure(); plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.4f}'); plt.legend(); plt.savefig(os.path.join(output_dir, 'pr_curve.png')); plt.close()


# 5. Scientific Seed
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Personalized CNN+LSTM on: {device.type.upper()}")
    
    DATA_DIR = "/hhome/ricse02/Epilepsy/"
    ROOT_DIR = "/hhome/ricse02/project_02/LSTM"
    H5_PATH = os.path.join(ROOT_DIR, "epilepsy_dataset.h5")
    BASE_OUTPUT = "/hhome/ricse02/project_02/result02/CNN_Population"

    if not os.path.exists(H5_PATH): build_hdf5_database(DATA_DIR, H5_PATH)
    with h5py.File(H5_PATH, 'r') as f:
        y_all = f['y'][:]; pid_all = f['patient_id'][:].astype(str)
        
    logo = LeaveOneGroupOut()
    splits = list(logo.split(np.zeros(len(y_all)), y_all, groups=pid_all))
    MAX_FOLDS_TO_RUN = 25 

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        if fold_idx >= MAX_FOLDS_TO_RUN: break
        test_subject = pid_all[test_idx[0]]
        fold_dir = os.path.join(BASE_OUTPUT, f"fold_{fold_idx:02d}_{test_subject}")
        os.makedirs(fold_dir, exist_ok=True)

        tr_labels = y_all[train_idx]
        idx0, idx1 = np.where(tr_labels == 0)[0], np.where(tr_labels == 1)[0]
        np.random.shuffle(idx0); np.random.shuffle(idx1)
        
        # Validation: 20% Original Distribution
        v0_size, v1_size = int(len(idx0) * 0.2), int(len(idx1) * 0.2)
        v_idx0, t_idx0_raw = idx0[:v0_size], idx0[v0_size:]
        v_idx1, t_idx1_raw = idx1[:v1_size], idx1[v1_size:]
        
        # Training: Balanced Downsampling using min() to prevent crash
        min_samples = min(len(t_idx0_raw), len(t_idx1_raw))
        t_idx0_bal = np.random.choice(t_idx0_raw, size=min_samples, replace=False)
        t_idx1_bal = np.random.choice(t_idx1_raw, size=min_samples, replace=False)
        
        final_t_idx = np.sort(train_idx[np.concatenate([t_idx0_bal, t_idx1_bal])])
        final_v_idx = np.sort(train_idx[np.concatenate([v_idx0, v_idx1])])
        final_test_idx = np.sort(test_idx)

        train_loader = DataLoader(H5EEGDataset(H5_PATH, final_t_idx), batch_size=64, shuffle=True)
        val_loader = DataLoader(H5EEGDataset(H5_PATH, final_v_idx), batch_size=64, shuffle=False)
        test_loader = DataLoader(H5EEGDataset(H5_PATH, final_test_idx), batch_size=64, shuffle=False)

        config = {"model": "CNN", "subject": test_subject, "split": "Train-Bal/Val-Orig", "dropout": 0.5}
        with open(os.path.join(fold_dir, 'config.json'), 'w') as f: json.dump(config, f, indent=4)

        print(f"\nFold {fold_idx+1}/{len(splits)} | Test Subject: {test_subject}")
        train_and_evaluate(BaselineCNN(), train_loader, val_loader, test_loader, device=device, output_dir=fold_dir)