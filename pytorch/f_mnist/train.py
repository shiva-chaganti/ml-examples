from functools import partial
import torch
from torchsummary import summary

from load_data import get_fmnist_dataset, get_data_loader
from le_net_model import FMnistClassifierLeNet, train, evaluate_model


def main():
    num_classes = 10
    data_download_path = 'f_mnist/datasets'
    model_save_path = 'f_mnist/model_state_dict.pth'
    batch_size = 64

    get_dataset_fn = partial(get_fmnist_dataset, data_download_path)
    train_loader, test_loader = get_data_loader(get_datasets_fn=get_dataset_fn(), batch_size=batch_size)


    model = FMnistClassifierLeNet(num_classes)
    print("Summary: \n")
    print(f'{summary(model, (1, 32, 32))}')
    train(model=model, num_epochs=10, train_loader=train_loader, model_save_path=model_save_path)
    model.load_state_dict(torch.load(model_save_path))
    evaluate_model(model=model, test_loader=test_loader)

main()
