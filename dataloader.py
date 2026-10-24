import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np


class Load_Dataset(Dataset):
    """
    Standard Dataset wrapper for CoTMix.
    Handles both 4D Audio Mel-Spectrograms [N, C, F, T] and 3D Sensor Series [N, C, L].
    """
    def __init__(self, dataset):
        super(Load_Dataset, self).__init__()

        X = dataset["samples"]
        y = dataset["labels"]

        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).long()

        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)
        else:
            X = X.float()

        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=torch.long)
        else:
            y = y.long()

        if len(X.shape) < 3:
            X = X.unsqueeze(1)  # [N, 1, L]

        # For 3D sensor inputs, ensure channel is second dimension [N, C, L]
        if len(X.shape) == 3 and X.shape.index(min(X.shape)) != 1:
            X = X.permute(0, 2, 1)

        self.x_data = X
        self.y_data = y
        self.len = X.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], int(self.y_data[index])

    def __len__(self):
        return self.len


def data_generator(data_path, domain_id, dataset_configs, hparams):
    """
    Loads domain datasets from disk and returns train and test DataLoaders.
    
    Files expected:
        train_{domain_id}.pt
        test_{domain_id}.pt
    """
    train_path = os.path.join(data_path, f"train_{domain_id}.pt")
    test_path = os.path.join(data_path, f"test_{domain_id}.pt")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Dataset files not found for domain '{domain_id}' in '{data_path}'.\n"
            f"Expected: '{train_path}' and '{test_path}'.\n"
            f"For music genre recognition, run: python -m data.prepare_music_dataset"
        )

    train_data = torch.load(train_path, weights_only=False)
    test_data = torch.load(test_path, weights_only=False)

    train_dataset = Load_Dataset(train_data)
    test_dataset = Load_Dataset(test_data)

    batch_size = hparams.get("batch_size", 16) if isinstance(hparams, dict) else getattr(hparams, "batch_size", 16)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True if len(train_dataset) >= batch_size else False,
        num_workers=0
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    return train_loader, test_loader
