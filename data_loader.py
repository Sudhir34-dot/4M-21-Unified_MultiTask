import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

# 🔹 Load CIFAR-10 for Image Classification
def load_cifar10():
    print(" Loading CIFAR-10 dataset...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match ImageNet size
        transforms.ToTensor(),
    ])
    dataset = torchvision.datasets.CIFAR10(root="datasets/cifar10", train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f" CIFAR-10 Dataset Loaded Successfully! Total samples: {len(dataset)}")
    return dataloader

# 🔹 Load COCO for Object Detection
def load_coco_detection():
    print("🔹 Loading COCO Detection dataset...")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = torchvision.datasets.CocoDetection(
        root="datasets/coco",
        annFile="datasets/coco/annotations/instances_train2017.json",
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    print(f" COCO Detection Dataset Loaded Successfully! Total samples: {len(dataset)}")
    return dataloader

# 🔹 Load COCO for Image Segmentation
def load_coco_segmentation():
    print(" Loading COCO Segmentation dataset...")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = torchvision.datasets.CocoDetection(
        root="C:/4m-21_image/datasets/coco2017",
        annFile="datasets/coco/annotations/instances_train2017.json",
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    print(f" COCO Segmentation Dataset Loaded Successfully! Total samples: {len(dataset)}")
    return dataloader

# 🔹 Run dataset loading functions
if __name__ == "__main__":
    load_cifar10() 

