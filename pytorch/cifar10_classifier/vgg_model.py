import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import accuracy_score, precision_score, recall_score


class FinetunedVgg11(nn.Module):
    """
    Model:

    ----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1         [-1, 64, 255, 255]           1,792
                ReLU-2         [-1, 64, 255, 255]               0
            MaxPool2d-3         [-1, 64, 127, 127]               0
                Conv2d-4        [-1, 128, 127, 127]          73,856
                ReLU-5        [-1, 128, 127, 127]               0
            MaxPool2d-6          [-1, 128, 63, 63]               0
                Conv2d-7          [-1, 256, 63, 63]         295,168
                ReLU-8          [-1, 256, 63, 63]               0
                Conv2d-9          [-1, 256, 63, 63]         590,080
                ReLU-10          [-1, 256, 63, 63]               0
            MaxPool2d-11          [-1, 256, 31, 31]               0
            Conv2d-12          [-1, 512, 31, 31]       1,180,160
                ReLU-13          [-1, 512, 31, 31]               0
            Conv2d-14          [-1, 512, 31, 31]       2,359,808
                ReLU-15          [-1, 512, 31, 31]               0
            MaxPool2d-16          [-1, 512, 15, 15]               0
            Conv2d-17          [-1, 512, 15, 15]       2,359,808
                ReLU-18          [-1, 512, 15, 15]               0
            Conv2d-19          [-1, 512, 15, 15]       2,359,808
                ReLU-20          [-1, 512, 15, 15]               0
            MaxPool2d-21            [-1, 512, 7, 7]               0
    AdaptiveAvgPool2d-22            [-1, 512, 7, 7]               0
            Linear-23                 [-1, 4096]     102,764,544
                ReLU-24                 [-1, 4096]               0
            Dropout-25                 [-1, 4096]               0
            Linear-26                 [-1, 4096]      16,781,312
                ReLU-27                 [-1, 4096]               0
            Dropout-28                 [-1, 4096]               0
            Linear-29                   [-1, 10]          40,970
    ================================================================
    """    
    def __init__(self, num_classes):
        super(FinetunedVgg11, self).__init__()

        vgg11_model = models.vgg11()
        self.features = vgg11_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, num_classes)
        )
        for params in self.features.parameters():
            params.requires_grad = False
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1) # Automatically calculates the size
        x = self.classifier(x)
        return x



def train(model: FinetunedVgg11, train_loader: DataLoader, num_epochs: int, model_save_path: str, device: str = 'cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

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
 

def evaluate_model(model: FinetunedVgg11, test_loader: DataLoader, device: str='cpu'):
    model.to(device)
    model.eval()
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            predictions = torch.max(outputs, 1)[1]
            y_pred.extend(predictions.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    
    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
