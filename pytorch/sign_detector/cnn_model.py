import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score

from load_data import get_data_loader

class SignDetectorCNN(nn.Module):
    """
    The network has 2 Conv layers with ReLu activations and a fully connected layer
    feeding into softmax layer.
    Conv 1:
        input: 64, 64, 3
        filter_size: 5, 5
        num_filters: 32
        padding: 2
        output: 64, 64, 32
    MaxPool1:
        input: 64, 64, 32
        filter_size: 2, 2
        stride: 2
        output: 32, 32, 32
    Conv 2:
        input: 32, 32, 32
        filter_size: 5, 5
        num_filters: 64
        padding: 2
        output: 32, 32, 64
    MaxPool2:
        input: 32, 32, 64
        filter_size: 2, 2
        stride: 2
        output: 16, 16, 64

    Linear layer:
        input: 16*16*64, num_classes
    """
    def __init__(self, num_classes):
        super(SignDetectorCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 16 * 16, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))  # Output size: [batch_size, 32, 32, 32]
        x = self.pool2(F.relu(self.conv2(x)))  # Output size: [batch_size, 64, 16, 16]
        x = x.view(-1, 64 * 16 * 16)  # Flatten the tensor for the fully connected layer
        logits = self.fc1(x)
        return logits


def train(model: SignDetectorCNN, train_loader: DataLoader, num_epochs: int, model_save_path: str, device: str='cpu'):
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


def evaluate_model(model: SignDetectorCNN, test_loader: DataLoader, device: str='cpu'):
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
    num_classes = 6
    train_data_path = 'pytorch/sign_detector/datasets/train_signs.h5'
    test_data_path = 'pytorch/sign_detector/datasets/test_signs.h5'
    model_save_path = 'pytorch/sign_detector/model_state_dict.pth'
    batch_size = 64

    train_loader, test_loader = get_data_loader(train_path=train_data_path, 
                                                test_path=test_data_path, 
                                                batch_size=batch_size)

    model = SignDetectorCNN(num_classes)
    print("Summary: \n")
    print(f'{summary(model, (3, 64, 64))}')
    train(model=model, num_epochs=10, train_loader=train_loader, model_save_path=model_save_path)
    model.load_state_dict(torch.load(model_save_path))
    evaluate_model(model=model, test_loader=test_loader)

