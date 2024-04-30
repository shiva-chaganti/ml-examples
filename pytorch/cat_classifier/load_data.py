from typing import Tuple
import h5py
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


def load_h5_dataset(train_path: str, test_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_dataset = h5py.File(train_path, "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # labels

    test_dataset = h5py.File(test_path, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # labels

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    # Normalize image vectors
    train_set_x = train_set_x_orig / 255.
    test_set_x = test_set_x_orig / 255.
    
    # Convert to torch tensors
    train_X = torch.Tensor(train_set_x).permute(0, 3, 1, 2)  # Reshape to [N, C, H, W]
    train_Y = torch.Tensor(train_set_y_orig).view(-1, 1)
    test_X = torch.Tensor(test_set_x).permute(0, 3, 1, 2)
    test_Y = torch.Tensor(test_set_y_orig).view(-1, 1)
    
    return train_X, train_Y, test_X, test_Y


def get_data_loader(train_path: str, test_path: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    train_X, train_Y, test_X, test_Y = load_h5_dataset(train_path, test_path)

    train_data = TensorDataset(train_X, train_Y)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    test_data = TensorDataset(test_X, test_Y)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
    return train_loader, test_loader
