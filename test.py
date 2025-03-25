import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CIFAR-10 Test Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Corrected normalization for RGB
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Define the ResNet18 Model
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=False)  # Use pretrained=True if needed
        num_ftrs = self.model.fc.in_features  # Dynamically get input size
        self.model.fc = nn.Linear(num_ftrs, 10)  # CIFAR-10 has 10 classes

    def forward(self, x):
        return self.model(x)

# Load model
model = ResNet18().to(device)
checkpoint_path = "models/cifar10_resnet18.pth"

try:
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    model.eval()
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Evaluate model accuracy
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"🎯 Model Accuracy on CIFAR-10 Test Set: {accuracy:.2f}%")

# Function to display images
def imshow(img):
    img = img * 0.5 + 0.5  # Unnormalize
    npimg = img.cpu().numpy()  # Move to CPU before conversion
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")
    plt.show()

# Get some random test images
dataiter = iter(testloader)
images, labels = next(dataiter)

# Make predictions
outputs = model(images.to(device))
_, predicted = torch.max(outputs, 1)

# Show images with predicted labels
imshow(torchvision.utils.make_grid(images[:8]))  # Display up to 8 images

print('📝 **Predicted Labels:**', ', '.join(testset.classes[predicted[j].item()] for j in range(min(8, len(predicted)))))
