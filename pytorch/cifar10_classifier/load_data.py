
from typing import Callable, Tuple
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader


def get_data_loader(get_datasets_fn: Callable[[], Tuple[TensorDataset, TensorDataset]],
                    batch_size: int) -> Tuple[DataLoader, DataLoader]:
    train_datasets, test_datasets = get_datasets_fn
    train_loader = DataLoader(dataset=train_datasets, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_datasets, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def transforms_for_alexnet() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((255,255)),
        transforms.CenterCrop(224), # Crop to 224x224 (input size for AlexNet)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_cifar10_dataset(data_transforms: transforms.Compose, download_path: str) -> Tuple[TensorDataset, TensorDataset]:
    train_datasets = datasets.CIFAR10(root=download_path, train=True, download=True, transform=data_transforms)
    test_datasets = datasets.CIFAR10(root=download_path, train=False, download=True, transform=data_transforms)

    return train_datasets, test_datasets
