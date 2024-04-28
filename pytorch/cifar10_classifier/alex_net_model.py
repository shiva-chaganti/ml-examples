import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import accuracy_score, precision_score, recall_score


class FinetunedAlexNet(nn.Module):
    """
        Model summary

        ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
        ================================================================
                Conv2d-1           [-1, 64, 63, 63]          23,296
                ReLU-2           [-1, 64, 63, 63]               0
            MaxPool2d-3           [-1, 64, 31, 31]               0
                Conv2d-4          [-1, 192, 31, 31]         307,392
                ReLU-5          [-1, 192, 31, 31]               0
            MaxPool2d-6          [-1, 192, 15, 15]               0
                Conv2d-7          [-1, 384, 15, 15]         663,936
                ReLU-8          [-1, 384, 15, 15]               0
                Conv2d-9          [-1, 256, 15, 15]         884,992
                ReLU-10          [-1, 256, 15, 15]               0
            Conv2d-11          [-1, 256, 15, 15]         590,080
                ReLU-12          [-1, 256, 15, 15]               0
            MaxPool2d-13            [-1, 256, 7, 7]               0
    AdaptiveAvgPool2d-14            [-1, 256, 6, 6]               0
            Dropout-15                 [-1, 9216]               0
            Linear-16                 [-1, 4096]      37,752,832
                ReLU-17                 [-1, 4096]               0
            Dropout-18                 [-1, 4096]               0
            Linear-19                 [-1, 4096]      16,781,312
                ReLU-20                 [-1, 4096]               0
            Linear-21                   [-1, 10]          40,970
        ================================================================
    """    
    def __init__(self, num_classes):
        super(FinetunedAlexNet, self).__init__()

        alexnet = models.alexnet(pretrained=True)

        self.features = alexnet.features
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
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



def train(model: FinetunedAlexNet, train_loader: DataLoader, num_epochs: int, model_save_path: str, device: str = 'cpu'):
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
 

def evaluate_model(model: models.alexnet, test_loader: DataLoader, device: str='cpu'):
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
