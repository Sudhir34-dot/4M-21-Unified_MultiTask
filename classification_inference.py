import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

# 🔹 Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 🔹 Define the model architecture (adjust for your dataset)
num_classes = 10  # Change this if your dataset has a different number of classes
model = models.resnet18(weights=None)  # 🔹 Use weights=None instead of pretrained=False
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# 🔹 Load the trained model weights
model_path = "C:/4m-21_image/models/cifar10_resnet18.pth"
model.load_state_dict(torch.load(model_path, map_location=device))  # 🔹 Remove weights_only=True
model.to(device)
model.eval()
print(f"Classification Model loaded from {model_path}")

# 🔹 Define transformation (match training normalization)
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])  # CIFAR-10 normalization
])

def classify_image(image_path):
    try:
        # Open and preprocess the image
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            output = model(image_tensor)
            predicted_class = torch.argmax(output, dim=1).item()

        # Map the predicted class index to a class name (if available)
        class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]  # CIFAR-10 classes
        predicted_class_name = class_names[predicted_class]

        return {"prediction": predicted_class_name}
    except Exception as e:
        return {"error": f"Error during classification: {str(e)}"}

# 🔹 Run inference
if __name__ == "__main__":
    test_image = "C:/4m-21_image/datasets/coco2017/test2017/000000000001.jpg"  # Change to an actual image path
    output = classify_image(test_image)
    print(f"Predicted Class: {output}")