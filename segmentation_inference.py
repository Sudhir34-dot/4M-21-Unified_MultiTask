import torch
import torchvision.transforms as T
from PIL import Image
import torchvision
import os
import cv2
import numpy as np
import time


# 🔹 Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 🔹 Load the trained segmentation model
model_path = "C:/4m-21_image/models/mask_rcnn_coco10k.pth"
model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")  # 🔹 Use "DEFAULT" for latest weights
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print(f"Segmentation Model loaded from {model_path}")

# 🔹 Define transformation
transform = T.Compose([
    T.ToTensor(),
])


def segment_image(image_path):
    try:
        # Open and preprocess the image
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).to(device).unsqueeze(0)

        # Run inference
        with torch.no_grad():
            prediction = model(image_tensor)

        # Post-process output
        masks = prediction[0]["masks"].cpu().numpy()
        masks = (masks > 0.5).astype(np.uint8)  # Convert to binary mask

        # Convert PIL image to OpenCV format
        image_cv = np.array(image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        # Overlay mask on the image
        for mask in masks:
            mask = mask.squeeze()  # Remove extra dimensions
            colored_mask = np.zeros_like(image_cv)
            colored_mask[mask == 1] = [0, 255, 0]  # Green mask
            image_cv = cv2.addWeighted(image_cv, 1, colored_mask, 0.5, 0)

        # Generate a unique filename
        timestamp = int(time.time())
        output_filename = f"output_segmentation_{timestamp}.jpg"
        output_path = os.path.join("static", "images", output_filename)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save the output image
        cv2.imwrite(output_path, image_cv)
        print(f"Output image saved to: {output_path}")  # Debug

        # Return the correct path for Flask
        return {"output_image": output_path.replace("\\", "/")}
    except Exception as e:
        return {"error": f"Error during segmentation: {str(e)}"}
# 🔹 Run inference
if __name__ == "__main__":
    test_image = "C:/4m-21_image/datasets/coco2017/test2017/000000000016.jpg"  # Change to an actual image path
    output = segment_image(test_image)
    print("Segmentation Output:", output)