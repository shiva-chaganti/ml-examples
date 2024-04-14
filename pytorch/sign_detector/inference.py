import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
from cnn_model import SignDetectorCNN


def load_model(model_path: str):
    num_classes = 6
    model = SignDetectorCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def preprocess_image(image_path: str):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)


def predict_image(image_path: str, model_path: str):
    model = load_model(model_path)
    image_tensor = preprocess_image(image_path)
    outputs = model(image_tensor)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()


def main():
    parser = argparse.ArgumentParser(description="Predict the class of a sign from an image")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the PyTorch model file")
    args = parser.parse_args()

    predicted_class_index = predict_image(args.image_path, args.model_path)

    # Define a dictionary mapping from class indices to human-readable labels
    class_labels = {
        0: "Stop",
        1: "Yield",
        2: "Speed Limit",
        3: "No Entry",
        4: "Construction",
        5: "No Parking"
    }

    predicted_class_label = class_labels[predicted_class_index]
    print(f'Predicted Class: {predicted_class_label}')

main()