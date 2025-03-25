import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

# Define class labels for CIFAR-10
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                    'dog', 'frog', 'horse', 'ship', 'truck']

# Load model
model_path = "models/cifar10_resnet18.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
model.fc = nn.Linear(512, 10)  # Adjust final layer for CIFAR-10 (10 classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Define preprocessing transformations (same as training)
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # CIFAR-10 images are 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Same normalization as training
])

# Load and preprocess the test image
image_path = "C:/4m-21_image/datasets/coco2017/bird_image_for_test.jpeg"  # Replace with your image path
image = Image.open(image_path).convert("RGB")  # Ensure it's RGB
image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

# Make prediction
with torch.no_grad():
    output = model(image)
    probabilities = F.softmax(output, dim=1)[0]  # Apply softmax

    # Get top-3 predictions
    top3_prob, top3_classes = torch.topk(probabilities, 3)

# Print the predictions
print("\nTop 3 Predictions:")
for i in range(3):
    print(f"{i+1}. {cifar10_classes[top3_classes[i].item()]} ({top3_prob[i].item():.2%})")

# Get the highest probability class
predicted_class = cifar10_classes[top3_classes[0].item()]
print(f"\nFinal Prediction: {predicted_class}")
