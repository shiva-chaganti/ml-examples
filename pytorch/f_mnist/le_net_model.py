
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, precision_score, recall_score

from load_data import get_fmnist_dataset, get_data_loader


class FMnistClassifierLeNet(nn.Module):
    """
    input -> conv -> pool -> conv -> pool -> fc -> fc -> fc -> output

    Conv1:
        input: 32, 32, 1
        filter_size: 5, 5
        num_filters: 6
        padding: 0
        output: 28, 28, 6
    MaxPool1:
        input: 28, 28, 6
        filter_size: 2, 2
        stride: 2
        output: 14, 14, 6
    Conv2:
        input: 14, 14, 6
        filter_size: 5, 5
        num_filters: 16
        padding: 0
        output: 10, 10, 16
    MaxPool2:
        input: 10, 10, 16
        filter_size: 2, 2
        stride: 2
        output: 5, 5, 16
    FC 1:
        input: 400
        output: 120
    FC 2:
        input: 120
        output: 84
    FC 3:
        input: 84
        output: num_classes

    """
    def __init__(self, num_classes):
        super(FMnistClassifierLeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 5 * 5 * 16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        
        return logits


def train(model: FMnistClassifierLeNet, train_loader: DataLoader, num_epochs: int, model_save_path: str, device: str='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.squeeze().long()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    
    print('Finished Training')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


def evaluate_model(model: FMnistClassifierLeNet, test_loader: DataLoader, device: str='cpu'):
    model.to(device)
    model.eval()
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():  # Inference mode, gradients not computed
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            predictions = torch.max(outputs, 1)[1]  # Get the index of the max log-probability
            y_pred.extend(predictions.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    
    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')

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
