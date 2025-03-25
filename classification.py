import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# 🔹 Detect GPU (or fallback to CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔹 Using device: {device}")

# 🔹 Define Data Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images for ResNet
    transforms.RandomHorizontalFlip(),  # Random flip for data augmentation
    transforms.RandomRotation(10),  # Rotate slightly
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])  # CIFAR-10 normalization
])

# 🔹 Define Model (Pretrained ResNet-18)
def create_model():
    model = models.resnet18(weights="IMAGENET1K_V1")  # Load pretrained weights
    model.fc = nn.Linear(model.fc.in_features, 10)  # Adjust final layer for 10 classes (CIFAR-10)
    return model.to(device)  # Move model to GPU if available

def train():
    print("🔹 Loading CIFAR-10 dataset...")
    train_dataset = datasets.CIFAR10(root="datasets/cifar10", train=True, transform=transform, download=True)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4 if torch.cuda.is_available() else 0)

    model = create_model()
    
    # 🔹 Define Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # Reduce LR dynamically

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        print(f"🔹 Starting Epoch {epoch+1}/{num_epochs}")

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)  # Move to GPU if available

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if batch_idx % 100 == 0:
                print(f" Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

        train_acc = 100 * correct / total
        print(f" Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_acc:.2f}%")

        scheduler.step()  # Adjust learning rate dynamically

    # 🔹 Save Trained Model
    torch.save(model.state_dict(), "models/cifar10_resnet18.pth")
    print(" Model saved as 'models/cifar10_resnet18.pth'")

def load_classification_model(model_path):
    """Load the trained classification model."""
    model = create_model()  # Create the model architecture
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load weights
    model.eval()  # Set to evaluation mode
    return model


# 🔹 Wrap everything inside the main guard (Fix for Windows)
if __name__ == "__main__":
    train()
