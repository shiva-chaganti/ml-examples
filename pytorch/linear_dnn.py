import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class DNN(nn.Module):
    """
    Simple DNN with multiple hidden layers L, with relu activation
    for L-1 layers and sigmoid activation for the output layer
    """
    def __init__(self):
        super(DNN, self).__init__()
        # Layers
        # Usually the number of units per hidden layer depend heavily on the specific task, 
        # the complexity of the data, and the available computational resources.
        # As a common practice, powers of 2 are chosen such as 256, 512, 1024, 2048, 4096.
        self.fc1 = nn.Linear(12288, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x


def main():
    model = DNN()
    print(f"Model summary:\n {summary(model, input_size=(12288,))}")
    example_input = torch.randn(209, 12288)
    output = model(example_input)
    print(f'Output shape: \n {output.shape}')
    print(f'Output: \n {output}')

if __name__ == "__main__":
    main()
