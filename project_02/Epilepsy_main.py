import os
import json
import h5py
import random
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc, 
                             precision_recall_curve, average_precision_score, accuracy_score)

from dataset import build_hdf5_database, UniversalEEGDataset
from model import BaselineCNN, TemporalCNN_LSTM
from datasplit import get_cv_splits, balance_training_set

# 1. Configuration
EXPERIMENT_STRATEGY = 'patient_leaky'  # Options: 'window', 'seizure', 'patient_strict', 'patient_leaky'
MODEL_TYPE          = 'CNN'             # Options: 'CNN', 'CNN_LSTM'

DATA_DIR        = "/hhome/ricse02/project_02/Epilepsy/"
H5_DIR        = "/hhome/ricse02/project_02/h5"
BASE_OUTPUT_DIR = f"/hhome/ricse02/project_02/result05/{MODEL_TYPE}_{EXPERIMENT_STRATEGY}"

# For Personalized ('seizure', 'window')
SINGLE_NPZ_FILE     = os.path.join(DATA_DIR, "chb01_seizure_EEGwindow_1.npz")
SINGLE_PARQUET_FILE = os.path.join(DATA_DIR, "chb01_seizure_metadata_1.parquet")

EPOCHS     = 15
LR         = 0.001
BATCH_SIZE = 64

# 2. Evaluation Functions (Strictly Sensitivity)
def event_level_evaluation(y_true, y_pred, interval_ids, original_indices, output_dir):
    interval_dict = defaultdict(list)
    
    # 1. Bind original time index with labels and predictions
    for i in range(len(y_true)):
        iid = interval_ids[i]
        interval_dict[iid].append((original_indices[i], y_true[i], y_pred[i]))

    seizure_events = 0
    detected_events = 0

    for iid, data in interval_dict.items():
        # 2. Sort explicitly by original index to prevent shuffle/loader errors
        data.sort(key=lambda x: x[0]) 
        
        labels = [x[1] for x in data]
        preds = [x[2] for x in data]
        
        # only care about intervals that actually contain a seizure
        if 1 in labels:
            seizure_events += 1
            detected = False
            
            # Check for 3 consecutive predicted positive windows
            for time_idx in range(len(preds) - 2):
                if preds[time_idx] == 1 and preds[time_idx+1] == 1 and preds[time_idx+2] == 1:
                    detected = True
                    break
                    
            if detected:
                detected_events += 1

    sensitivity = detected_events / seizure_events if seizure_events > 0 else 0.0

    print(f"CLINICAL METRICS | Event-Level Sensitivity: {sensitivity:.4f}")
    with open(os.path.join(output_dir, "event_metrics.txt"), "w") as f:
        f.write(f"Total Seizure Events: {seizure_events}\n")
        f.write(f"Detected Events: {detected_events}\n")
        f.write(f"Sensitivity (Recall): {sensitivity:.4f}\n")
        
    return sensitivity

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Unified Pipeline | Model: {MODEL_TYPE} | Strategy: {EXPERIMENT_STRATEGY} | Device: {device.type.upper()}")
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    # Load Data
    if EXPERIMENT_STRATEGY in ['patient_strict', 'patient_leaky']:
        strict_mode = (EXPERIMENT_STRATEGY == 'patient_strict')
        db_name = "epilepsy_dataset_strict.h5" if strict_mode else "epilepsy_dataset_leaky.h5"
        H5_FILE_PATH = os.path.join(H5_DIR, db_name)
        
        if not os.path.exists(H5_FILE_PATH): 
            build_hdf5_database(DATA_DIR, H5_FILE_PATH, strict=strict_mode)
            
        h5_file = h5py.File(H5_FILE_PATH, 'r')
        y_all = h5_file['y'][:]
        pid_all = h5_file['patient_id'][:].astype(str)
        intervals_all = h5_file['global_interval'][:]
        data_source = H5_FILE_PATH
    else:
        data = np.load(SINGLE_NPZ_FILE, allow_pickle=True)
        data_source = data[data.files[0]].astype(np.float32)
        meta = pd.read_parquet(SINGLE_PARQUET_FILE, engine='fastparquet')
        y_all = meta['class'].values
        intervals_all = meta['global_interval'].values
        pid_all = None

    splits = get_cv_splits(EXPERIMENT_STRATEGY, y_all, pid_all, intervals_all)
    
    all_sensitivity, all_pr_auc, all_window_acc = [], [], []
    fold_results_list = []
    global_y_true, global_y_probs = [], []
    valid_folds_count = 0

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        if EXPERIMENT_STRATEGY == 'patient_strict' and fold_idx >= 25: break
        
        test_id = pid_all[test_idx[0]] if pid_all is not None else f"Fold_{fold_idx:02d}"
        FOLD_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"fold_{fold_idx:02d}_{test_id}")
        os.makedirs(FOLD_OUTPUT_DIR, exist_ok=True)
        print(f"\n{'='*40}\n Fold {fold_idx+1}/{len(splits)} | Test Target: {test_id}\n{'='*40}")

        final_t_idx = balance_training_set(train_idx, y_all)
        if len(final_t_idx) == 0:
            print(f"Skipping Fold {fold_idx}: Insufficient samples.")
            continue
            
        train_dataset = UniversalEEGDataset(data_source, final_t_idx, y_all, intervals_all, model_type=MODEL_TYPE)
        test_dataset = UniversalEEGDataset(data_source, np.sort(test_idx), y_all, intervals_all, model_type=MODEL_TYPE)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = BaselineCNN().to(device) if MODEL_TYPE == 'CNN' else TemporalCNN_LSTM().to(device)

        with open(os.path.join(FOLD_OUTPUT_DIR, 'model_architecture.txt'), 'w') as f:
            f.write(str(model))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)
        best_model_path = os.path.join(FOLD_OUTPUT_DIR, 'final_model_weights.pth')
        
        # Training
        epoch_train_losses = []
        for epoch in range(EPOCHS):
            model.train(); train_loss = 0.0
            for batch_X, batch_y, _, _ in tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}"):
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_X.size(0)
            
            avg_train_loss = train_loss/len(final_t_idx)
            epoch_train_losses.append(avg_train_loss)
            print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {avg_train_loss:.4f}")

        torch.save(model.state_dict(), best_model_path)
            
        np.save(os.path.join(FOLD_OUTPUT_DIR, 'train_loss.npy'), np.array(epoch_train_losses))
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, EPOCHS + 1), epoch_train_losses, marker='o', linestyle='-', color='b', label='Training Loss')
        plt.title(f'Training Loss Curve - {test_id}', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Cross Entropy Loss', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(FOLD_OUTPUT_DIR, 'training_loss_curve.png'))
        plt.close()

        # Testing
        model.eval()
        y_true, y_pred, y_probs = [], [], []
        interval_test, original_idx_test = [], []
        
        with torch.no_grad():
            for batch_X, batch_y, batch_interval, batch_actual_idx in tqdm(test_loader, leave=False, desc="Final Testing"):
                outputs = model(batch_X.to(device))
                y_probs.extend(F.softmax(outputs, dim=1)[:, 1].cpu().numpy())
                y_true.extend(batch_y.numpy())
                y_pred.extend(torch.max(outputs, 1)[1].cpu().numpy())
                interval_test.extend(batch_interval.numpy())
                original_idx_test.extend(batch_actual_idx.numpy())

        global_y_true.extend(y_true)
        global_y_probs.extend(y_probs)

        with open(os.path.join(FOLD_OUTPUT_DIR, 'classification_report.txt'), 'w') as f: 
            f.write(classification_report(y_true, y_pred, target_names=['Non-seizure (0)', 'Seizure (1)']))
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Non-seizure', 'Seizure'], 
                    yticklabels=['Non-seizure', 'Seizure'])
        plt.title(f'Confusion Matrix - {test_id}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(FOLD_OUTPUT_DIR, 'confusion_matrix.png'), dpi=300)
        plt.close()

        # Metrics
        window_acc = accuracy_score(y_true, y_pred)
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        pr_auc = average_precision_score(y_true, y_probs)
        
        sens = event_level_evaluation(y_true, y_pred, interval_test, original_idx_test, FOLD_OUTPUT_DIR)
        
        all_window_acc.append(window_acc)
        all_sensitivity.append(sens)
        all_pr_auc.append(pr_auc)
        valid_folds_count += 1
        
        fold_results_list.append({
            "test_target": str(test_id),
            "window_level_accuracy": float(window_acc),
            "event_level_sensitivity": float(sens),
            "window_level_pr_auc": float(pr_auc)
        })

    # 4. Global Summary & Visualization
    if valid_folds_count > 0:
        print("\nGenerating Global Summaries...")
        fpr_all, tpr_all, _ = roc_curve(global_y_true, global_y_probs)
        roc_auc_all = auc(fpr_all, tpr_all)
        plt.figure()
        plt.plot(fpr_all, tpr_all, color='darkorange', lw=2, label=f'Overall AUC = {roc_auc_all:.4f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Overall ROC Curve ({EXPERIMENT_STRATEGY.upper()})')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(BASE_OUTPUT_DIR, 'overall_roc_curve.png'), dpi=300)
        plt.close()

        precision_all, recall_all, _ = precision_recall_curve(global_y_true, global_y_probs)
        pr_auc_all = average_precision_score(global_y_true, global_y_probs)
        plt.figure()
        plt.plot(recall_all, precision_all, color='blue', lw=2, label=f'Overall PR AUC = {pr_auc_all:.4f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Overall PR Curve ({EXPERIMENT_STRATEGY.upper()})')
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(BASE_OUTPUT_DIR, 'overall_pr_curve.png'), dpi=300)
        plt.close()

        folds_x = np.arange(1, len(all_sensitivity) + 1)
        plt.figure(figsize=(12, 6))
        width = 0.35
        plt.bar(folds_x - width/2, [a * 100 for a in all_window_acc], width, label='Window-Level Accuracy (%)', color='#87CEEB', edgecolor='black')
        plt.bar(folds_x + width/2, [s * 100 for s in all_sensitivity], width, label='Event-Level Sensitivity (%)', color='#FA8072', edgecolor='black')
        
        plt.axhline(np.mean(all_window_acc) * 100, color='blue', linestyle='--', alpha=0.6, label=f'Avg Accuracy ({np.mean(all_window_acc)*100:.1f}%)')
        plt.axhline(np.mean(all_sensitivity) * 100, color='red', linestyle='--', alpha=0.6, label=f'Avg Sensitivity ({np.mean(all_sensitivity)*100:.1f}%)')
        
        plt.xlabel('Cross-Validation Fold / Target', fontsize=12)
        plt.ylabel('Metric Percentage (%)', fontsize=12)
        plt.title(f'The Clinical Disconnect ({EXPERIMENT_STRATEGY.upper()})', fontsize=14, fontweight='bold')
        plt.xticks(folds_x, [str(f['test_target']) for f in fold_results_list], rotation=45)
        plt.ylim(0, 110) 
        plt.legend(loc='lower right', framealpha=0.9)
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_OUTPUT_DIR, 'clinical_disconnect_comparison.png'), dpi=300)
        plt.close()

        final_summary_dict = {
            "experiment_strategy": EXPERIMENT_STRATEGY,
            "model": MODEL_TYPE,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "global_metrics_mean": {
                "window_level_accuracy": float(np.mean(all_window_acc)),
                "event_level_sensitivity": float(np.mean(all_sensitivity)),
                "window_level_pr_auc": float(np.mean(all_pr_auc))
            },
            "per_fold_breakdown": fold_results_list
        }
        with open(os.path.join(BASE_OUTPUT_DIR, 'final_cross_val_summary.json'), 'w') as f:
            json.dump(final_summary_dict, f, indent=4)
            
        print(f"Pipeline finished successfully! Check {BASE_OUTPUT_DIR} for results.")

if __name__ == "__main__":
    main()