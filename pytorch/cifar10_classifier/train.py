from functools import partial
import torch
from torchsummary import summary
from alex_net_model import alexnet_model, train_alexnet, evaluate_model
from load_data import get_cifar10_dataset, get_data_loader, transforms_for_alexnet


def main():
    num_classes = 10
    data_download_path = 'pytorch/cifar10_classifier/datasets'
    model_save_path = 'pytorch/cifar10_classifier/model_state_dict.pth'
    batch_size = 64

    alexnet_data_transform = transforms_for_alexnet()
    get_dataset_fn = partial(get_cifar10_dataset, alexnet_data_transform, data_download_path)
    train_loader, test_loader = get_data_loader(get_datasets_fn=get_dataset_fn(), batch_size=batch_size)

    model = alexnet_model(num_classes)
    print("Summary: \n")
    print(f'{summary(model, (3, 255, 255))}')
    train_alexnet(model=model, num_epochs=10, train_loader=train_loader, model_save_path=model_save_path)
    model.load_state_dict(torch.load(model_save_path))
    evaluate_model(model=model, test_loader=test_loader)

main()
