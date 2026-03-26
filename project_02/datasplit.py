import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, LeaveOneGroupOut

def get_cv_splits(strategy, y_all, pid_all=None, intervals_all=None):

    if strategy == 'window':
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        return list(skf.split(np.zeros(len(y_all)), y_all))
        
    elif strategy == 'seizure':
        sgkf = StratifiedGroupKFold(n_splits=5)
        splits = []
        for train_idx, test_idx in sgkf.split(np.zeros(len(y_all)), y_all, groups=intervals_all):
            splits.append((np.sort(train_idx), np.sort(test_idx)))
        return splits
        
    elif strategy == 'patient_strict':
        logo = LeaveOneGroupOut()
        return list(logo.split(np.zeros(len(y_all)), y_all, groups=pid_all))
        
    elif strategy == 'patient_leaky':
        # Strong Identity Leakage: Train strictly on chb01, Test on chb21
        test_idx = np.where(pid_all == "chb21")[0]
        train_idx = np.where(pid_all == "chb01")[0] 
        return [(train_idx, test_idx)]
        
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def balance_training_set(train_idx, y_all):
    """Balances the training set 50/50 for Seizure/Non-seizure."""
    tr_labels = y_all[train_idx]
    idx0 = np.where(tr_labels == 0)[0]
    idx1 = np.where(tr_labels == 1)[0]
    np.random.shuffle(idx0)
    np.random.shuffle(idx1)
    
    min_samples = min(len(idx0), len(idx1))
    if min_samples == 0:
        return np.array([])
        
    t_idx0_bal = np.random.choice(idx0, size=min_samples, replace=False)
    t_idx1_bal = np.random.choice(idx1, size=min_samples, replace=False)
    
    return np.sort(train_idx[np.concatenate([t_idx0_bal, t_idx1_bal])])