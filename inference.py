import argparse
import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.classification_inference import classify_image
from src.detection_inference import detect_objects
from src.segmentation_inference import segment_image

def run_inference(image_path, task="classification"):
    """Run inference based on selected task."""
    if not os.path.exists(image_path):
        return {"error": f"Image path {image_path} does not exist."}

    try:
        if task == "classification":
            output = classify_image(image_path)
            return {"task": "classification", "prediction": output}
        
        elif task == "detection":
            output = detect_objects(image_path)
            return {"task": "detection", "detections": output}
        
        elif task == "segmentation":
            output = segment_image(image_path)
            return {"task": "segmentation", "segmentation_map": output}
        
        else:
            return {"error": "Invalid task selected"}
    except Exception as e:
        return {"error": f"Error during inference: {str(e)}"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on an image")
    parser.add_argument("--task", type=str, required=True, choices=["classification", "detection", "segmentation"],
                        help="Task to perform: classification, detection, segmentation")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")

    args = parser.parse_args()
    
    result = run_inference(args.image, args.task)
    print(result)