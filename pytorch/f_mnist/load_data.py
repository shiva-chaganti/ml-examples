
from typing import Callable, Tuple
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader


def get_data_loader(get_datasets_fn: Callable[[], Tuple[TensorDataset, TensorDataset]],
                    batch_size: int) -> Tuple[DataLoader, DataLoader]:
    train_datasets, test_datasets = get_datasets_fn
    train_loader = DataLoader(dataset=train_datasets, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_datasets, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def get_fmnist_dataset(download_path: str) -> Tuple[TensorDataset, TensorDataset]:
    fmnist_transforms = transforms.Compose([
        transforms.Resize((32,32)), # Required by LeNet model
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])
    train_datasets = datasets.FashionMNIST(root=download_path, train=True, download=True, transform=fmnist_transforms)
    test_datasets = datasets.FashionMNIST(root=download_path, train=False, download=True, transform=fmnist_transforms)

    return train_datasets, test_datasets
