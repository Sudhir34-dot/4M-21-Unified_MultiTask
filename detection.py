import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, Subset
from pycocotools.coco import COCO
import os
import numpy as np
from PIL import Image
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please check your PyTorch installation and GPU setup.")

print(f"Using device: {device}")


# Define transformation
transform = T.Compose([
    T.ToTensor(),
])

# COCO Dataset Class
class COCODataset(Dataset):
    def __init__(self, image_folder, annotation_file, transform=None):
        self.image_folder = image_folder
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def is_valid_bbox(self, bbox):
        """Check if bounding box is valid (positive width & height)."""
        x, y, w, h = bbox
        return w > 0 and h > 0

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)

        # Load Image
        img_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.image_folder, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        # Filter out invalid bounding boxes
        valid_annotations = []
        for ann in annotations:
            bbox = ann["bbox"]
            if self.is_valid_bbox(bbox):
                bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]  # Convert to x_min, y_min, x_max, y_max
                ann["bbox"] = bbox
                valid_annotations.append(ann)

        if not valid_annotations:
            return self.__getitem__((idx + 1) % len(self.image_ids))  # Skip empty annotations

        # Convert to tensor
        boxes = torch.as_tensor([ann["bbox"] for ann in valid_annotations], dtype=torch.float32)
        labels = torch.as_tensor([ann["category_id"] for ann in valid_annotations], dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transform:
            image = self.transform(image)

        return image, target

# Define dataset paths
image_folder = "C:/4m-21_image/datasets/coco2017/train2017"
annotation_file = "C:/4m-21_image/datasets/coco2017/annotations/instances_train2017.json"

# Load full dataset
full_dataset = COCODataset(image_folder, annotation_file, transform=transform)

# Select 40,000 random images for training
random.seed(42)
selected_indices = random.sample(range(len(full_dataset)), 10000)
dataset = Subset(full_dataset, selected_indices)

# Create DataLoader
data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda batch: tuple(zip(*batch)))

# Load Faster R-CNN model (pre-trained on COCO)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="COCO_V1")
model.to(device)
model.train()

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    total_loss = 0

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Print loss every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f"Batch {batch_idx+1}/{len(data_loader)} - Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} completed! Average Loss: {total_loss / len(data_loader):.4f}")

# Save the trained model
model_save_path = "C:/4m-21_image/models/faster_rcnn_coco40k.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved at {model_save_path}")
