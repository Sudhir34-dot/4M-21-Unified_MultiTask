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

# 🔹 Load the trained object detection model
model_path = "C:/4m-21_image/models/faster_rcnn_coco40k.pth"
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")  
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print(f"Object Detection Model loaded from {model_path}")

# 🔹 Define transformation
transform = T.Compose([T.ToTensor()])

def detect_objects(image_path):
    try:
        # Open and preprocess the image
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).to(device).unsqueeze(0)

        # Run inference
        with torch.no_grad():
            prediction = model(image_tensor)

        # Post-process output
        boxes = prediction[0]["boxes"].cpu().numpy()
        labels = prediction[0]["labels"].cpu().numpy()
        scores = prediction[0]["scores"].cpu().numpy()

        print("Predictions received. Total detections:", len(scores))

        # Convert PIL image to OpenCV format
        image_cv = np.array(image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        # Get image dimensions
        height, width, _ = image_cv.shape

        # Draw bounding boxes and labels
        detected = False  # Flag to check if any box is drawn
        for box, label, score in zip(boxes, labels, scores):
            if score > 0.3:  # Reduced confidence threshold
                x1, y1, x2, y2 = map(int, box)
                
                # Check if box is valid within image dimensions
                if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                    print(f"Skipping invalid box: {x1}, {y1}, {x2}, {y2}")
                    continue

                print(f"Drawing box: {x1}, {y1}, {x2}, {y2}, Label: {label}, Score: {score:.2f}")  # Debugging
                cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image_cv, f"Class {label} ({score:.2f})", 
                            (x1, max(y1 - 10, 20)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                detected = True

        if not detected:
            print(" No high-confidence detections found!")

        # Show the image with bounding boxes
        cv2.imshow("Detected Objects", image_cv)
        cv2.waitKey(0)  # Ensure the window remains open
        cv2.destroyAllWindows()  # Close the window after key press

        # Save the output image
        timestamp = int(time.time())
        output_filename = f"output_detection_{timestamp}.jpg"
        output_path = os.path.join("static", "images", output_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image_cv)
        print(f" Output image saved to: {output_path}")

        return {"output_image": output_path.replace("\\", "/")}
    except Exception as e:
        return {"error": f"Error during detection: {str(e)}"}

# 🔹 Run inference
if __name__ == "__main__":
    test_image = "C:/4m-21_image/datasets/coco2017/test2017/000000000016.jpg"
    output = detect_objects(test_image)
    print("Detection Output:", output)
