import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
from le_net_model import FMnistClassifierLeNet


def load_model(model_path: str):
    num_classes = 10
    model = FMnistClassifierLeNet(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def preprocess_image(image_path: str):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    image = Image.open(image_path).convert('L')
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
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot"
    }

    predicted_class_label = class_labels[predicted_class_index]
    print(f'Predicted Class: {predicted_class_label}')

main()