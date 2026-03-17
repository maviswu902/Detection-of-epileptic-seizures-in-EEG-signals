import os
import json
import numpy as np
import random
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import torch.nn.functional as F

# 1. Data Pipeline
class EpilepsyDataPipeline:
    def __init__(self, npz_file, parquet_file):
        print(f"Loading personalized data from {npz_file}...")
        data = np.load(npz_file, allow_pickle=True)
        self.X = data[data.files[0]].astype(np.float32)
        self.metadata = pd.read_parquet(parquet_file, engine='fastparquet')
        self.y = self.metadata['class'].values
        self.intervals = self.metadata['global_interval'].values

    def get_seizure_splits(self, n_splits=5):
        sgkf = StratifiedGroupKFold(n_splits=n_splits)
        splits = []
        for train_idx, test_idx in sgkf.split(self.X, self.y, groups=self.intervals):
            splits.append((np.sort(train_idx), np.sort(test_idx)))
        return splits

# 2. InMemory Dataset
class InMemoryDataset(Dataset):
    def __init__(self, X, y, indices):
        self.X = torch.tensor(X[indices], dtype=torch.float32)
        self.y = torch.tensor(y[indices], dtype=torch.long)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# 3. Baseline CNN
class BaselineCNN(nn.Module):
    def __init__(self, num_channels=21, time_steps=128):
        super(BaselineCNN, self).__init__()
        self.data_fusion = nn.Conv1d(num_channels, 16, kernel_size=1)
        self.conv1 = nn.Sequential(nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool1d(4, 4))
        self.conv2 = nn.Sequential(nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool1d(4, 4))
        self.conv3 = nn.Sequential(nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool1d(4, 4))
        self.flatten = nn.Flatten()
        
        with torch.no_grad():
            dummy = torch.zeros(1, num_channels, time_steps)
            dummy = self.conv3(self.conv2(self.conv1(self.data_fusion(dummy))))
            self.feature_size = dummy.view(1, -1).size(1)
            
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
    os.makedirs(output_dir, exist_ok=True)
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

    # Final Test
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

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

# 5. Main Execution
if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Personalized CNN on: {device.type.upper()}")
    
    DATA_DIR = "/hhome/ricse/Epilepsy/" 
    NPZ_FILE = os.path.join(DATA_DIR, "chb01_seizure_EEGwindow_1.npz") 
    PARQUET_FILE = os.path.join(DATA_DIR, "chb01_seizure_metadata_1.parquet")
    PROJECT_DIR = "/hhome/ricse02/project_02"
    BASE_OUTPUT = os.path.join(PROJECT_DIR, "result02/CNN_Single")
    
    pipeline = EpilepsyDataPipeline(NPZ_FILE, PARQUET_FILE)
    splits = pipeline.get_seizure_splits(n_splits=5)
    
    # Run Full Cross-Validation across all Seizure Intervals
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        FOLD_OUTPUT_DIR = os.path.join(BASE_OUTPUT, f"fold_{fold_idx:02d}")
        os.makedirs(FOLD_OUTPUT_DIR, exist_ok=True)
        
        # Stratified Splitting for Train/Val
        tr_labels = pipeline.y[train_idx]
        idx0, idx1 = np.where(tr_labels == 0)[0], np.where(tr_labels == 1)[0]
        np.random.shuffle(idx0); np.random.shuffle(idx1)
        
        v0_size = int(len(idx0) * 0.2)
        v1_size = max(1, int(len(idx1) * 0.2)) if len(idx1) > 0 else 0
        
        v_idx0, t_idx0_raw = idx0[:v0_size], idx0[v0_size:]
        v_idx1, t_idx1_raw = idx1[:v1_size], idx1[v1_size:]
        
        # Balance only the Training portion
        min_samples = min(len(t_idx0_raw), len(t_idx1_raw))
        if min_samples == 0: continue # Skip if no seizure in train
        
        t_idx0_bal = np.random.choice(t_idx0_raw, size=min_samples, replace=False)
        t_idx1_bal = np.random.choice(t_idx1_raw, size=min_samples, replace=False)
        
        final_t_idx = np.sort(train_idx[np.concatenate([t_idx0_bal, t_idx1_bal])])
        final_v_idx = np.sort(train_idx[np.concatenate([v_idx0, v_idx1])])
        final_test_idx = np.sort(test_idx)

        train_loader = DataLoader(InMemoryDataset(pipeline.X, pipeline.y, final_t_idx), batch_size=64, shuffle=True)
        val_loader = DataLoader(InMemoryDataset(pipeline.X, pipeline.y, final_v_idx), batch_size=64, shuffle=False)
        test_loader = DataLoader(InMemoryDataset(pipeline.X, pipeline.y, final_test_idx), batch_size=64, shuffle=False)
        
        config = {"model": "BaselineCNN", "patient": "chb01", "fold": fold_idx, "split": "Train-Bal/Val-Orig"}
        with open(os.path.join(FOLD_OUTPUT_DIR, 'config.json'), 'w') as f: json.dump(config, f)
        
        print(f"\n Fold {fold_idx+1}/{len(splits)} | Train: {len(final_t_idx)}, Val: {len(final_v_idx)}, Test: {len(final_test_idx)}")
        train_and_evaluate(BaselineCNN(), train_loader, val_loader, test_loader, epochs=15, lr=0.001, device=device, output_dir=FOLD_OUTPUT_DIR)