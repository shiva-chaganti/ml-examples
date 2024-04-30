import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from load_data import get_data_loader

class CatDetectorLR(nn.Module):
    def __init__(self, input_size):
        super(CatDetectorLR, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # Single linear layer

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.linear(x))  # Sigmoid activation function for binary classification
        return x

def train(model: CatDetectorLR, num_epochs: int, train_loader: DataLoader, model_save_path: str):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
    
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


def eval(model: CatDetectorLR, test_loader: DataLoader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            predicted = outputs.round()  # Threshold at 0.5
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on the test set: {100 * correct / total}%')


def main():
    # Assuming images are of size 64x64x3, adjust if your images are of a different size
    input_size = 64 * 64 * 3
    train_data_path = 'pytorch/cat_detector/datasets/train_catvnoncat.h5'
    test_data_path = 'pytorch/cat_detector/datasets/test_catvnoncat.h5'
    model_save_path = 'pytorch/cat_detector/model_state_dict.pth'
    batch_size = 64

    model = CatDetectorLR(input_size)
    train_loader, test_loader = get_data_loader(train_path=train_data_path, 
                                                test_path=test_data_path, 
                                                batch_size=batch_size)
    
    train(model=model, num_epochs=10, train_loader=train_loader, model_save_path=model_save_path)
    model.load_state_dict(torch.load(model_save_path))
    eval(model=model, test_loader=test_loader)

main()