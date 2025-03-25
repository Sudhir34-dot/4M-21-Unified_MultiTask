import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from PIL import Image

# Step 1: Load the trained model
model = resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10 has 10 classes
model.load_state_dict(torch.load("C:/4m-21_image/models/Classification_resnet18.pth"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print("Model loaded successfully!")

# Step 2: Define transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to CIFAR-10 input size
    transforms.ToTensor(),         # Convert to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Step 3: Test on a single image
image_path = "example_image.jpg"  # Replace with your image path
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)
    print(f"Predicted class index: {predicted.item()}")

# Step 4: Map class index to class name
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
predicted_class = class_names[predicted.item()]
print(f"Predicted class: {predicted_class}")

# Step 5: Test on the CIFAR-10 test dataset
test_dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100.0 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")