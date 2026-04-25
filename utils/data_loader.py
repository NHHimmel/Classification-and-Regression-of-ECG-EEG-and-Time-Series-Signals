import numpy as np
import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from utils.config import DATASET_CONFIGS
from utils.preprocessor import encode_label, Normalizer, Target_Normalizer

def load_raw(dataset_name: str, base_path: str):
    """Load raw .npy files from drive."""
    config = DATASET_CONFIGS[dataset_name]
    path = Path(base_path) / config['folder']

    X_train = np.load(path / 'x_train_original.npy')
    y_train = np.load(path / 'y_train.npy')
    X_test  = np.load(path / 'x_test_original.npy')
    y_test  = np.load(path / 'y_test.npy')

    return X_train, y_train, X_test, y_test


def to_tensors(X_train, y_train, X_val, y_val, X_test, y_test, task: str):
    """Convert numpy arrays to torch tensors with correct dtypes."""
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test  = torch.tensor(X_test,  dtype=torch.float32)
    X_val   = torch.tensor(X_val,   dtype=torch.float32)

    if task == 'classification':
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test  = torch.tensor(y_test,  dtype=torch.long)
        y_val   = torch.tensor(y_val,   dtype=torch.long)
    else:  # regression
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_test  = torch.tensor(y_test,  dtype=torch.float32)
        y_val   = torch.tensor(y_val,   dtype=torch.float32)

    return X_train, y_train, X_val, y_val, X_test, y_test


def make_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size: int = 32):
    """Wrap tensors into DataLoaders."""
    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_val, y_val)
    test_ds  = TensorDataset(X_test,  y_test)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_dl, val_dl, test_dl


def load_dataset(dataset_name: str, base_path: str, batch_size: int = 32, val_ratio = 0.2):
    """Main entry point."""
    config   = DATASET_CONFIGS[dataset_name]
    task     = config['task']

    X_train, y_train, X_test, y_test = load_raw(dataset_name, base_path)
    # Split the Original Train Set into Train + Validation Set
    stratify = y_train if task == 'classification' else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_ratio, stratify=stratify
    )

    if task=='classification':
        y_train, y_val, y_test = encode_label(y_train, y_val, y_test)
    if config['scale_features']:
        scaler = Normalizer(config)
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    if config['scale_target']:
        scaler = Target_Normalizer()
        scaler.fit(y_train)
        y_train = scaler.transform(y_train)
        y_val = scaler.transform(y_val)
        y_test = scaler.transform(y_test)

    X_train, y_train, X_val, y_val, X_test, y_test = to_tensors(X_train, y_train, X_val, y_val, X_test, y_test, task)
    train_dl, val_dl, test_dl                = make_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size)

    return train_dl, val_dl, test_dl