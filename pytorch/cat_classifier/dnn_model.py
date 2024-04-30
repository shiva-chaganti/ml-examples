import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, accuracy_score

from load_data import get_data_loader

class CatDetectorDNN(nn.Module):
    def __init__(self, input_size):
        super(CatDetectorDNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.bn1(self.fc1(x)))
        x = torch.tanh(self.bn2(self.fc2(x)))
        x = torch.tanh(self.bn3(self.fc3(x)))
        x = torch.sigmoid(self.fc4(x))
        return x

def train(model: CatDetectorDNN, num_epochs: int, train_loader: DataLoader, model_save_path: str):
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


def evaluate_model(model: CatDetectorDNN, test_loader: DataLoader):
    model.eval()
    y_pred = []
    y_true = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            predicted = outputs.round()
            y_pred.extend(predicted.view(-1).tolist())
            y_true.extend(labels.view(-1).tolist())
    
    y_pred = torch.tensor(y_pred)
    y_true = torch.tensor(y_true)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")


def main():
    # Assuming images are of size 64x64x3, adjust if your images are of a different size
    input_size = 64 * 64 * 3
    train_data_path = 'cat_detector/datasets/train_catvnoncat.h5'
    test_data_path = 'cat_detector/datasets/test_catvnoncat.h5'
    model_save_path = 'cat_detector/model_state_dict.pth'
    batch_size = 64

    model = CatDetectorDNN(input_size)
    train_loader, test_loader = get_data_loader(train_path=train_data_path, 
                                                test_path=test_data_path, 
                                                batch_size=batch_size)
    
    train(model=model, num_epochs=10, train_loader=train_loader, model_save_path=model_save_path)
    model.load_state_dict(torch.load(model_save_path))
    evaluate_model(model=model, test_loader=test_loader)

main()