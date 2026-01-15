import cv2
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
import numpy as np
import time
import os

class WebcamStream:
    """Handles webcam frame capture using OpenCV."""
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            raise ValueError(f"Could not open webcam {src}")
        
    def get_frame(self):
        ret, frame = self.stream.read()
        if not ret:
            return None
        return frame

    def release(self):
        self.stream.release()

class ObjectDetector:
    """Handles object detection using torchvision's Faster R-CNN."""
    def __init__(self, threshold=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize model with pre-trained weights
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=weights, box_score_thresh=threshold)
        self.model.to(self.device)
        self.model.eval()
        self.categories = weights.meta["categories"]

    def detect(self, frame):
        # Convert BGR (OpenCV) to RGB (Torchvision)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to tensor and add batch dimension
        img_tensor = F.to_tensor(img_rgb).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            prediction = self.model(img_tensor)[0]
        
        return prediction

    def get_labels(self, prediction):
        labels = [self.categories[i] for i in prediction['labels']]
        scores = prediction['scores'].tolist()
        return list(zip(labels, scores))

def main():
    # Configuration
    WEBCAM_INDEX = int(os.getenv("WEBCAM_INDEX", 0))
    DETECTION_THRESHOLD = float(os.getenv("DETECTION_THRESHOLD", 0.7))
    
    print(f"Starting Sentinel Vision Module...")
    print(f"Webcam index: {WEBCAM_INDEX}")
    print(f"Detection Threshold: {DETECTION_THRESHOLD}")

    try:
        stream = WebcamStream(src=WEBCAM_INDEX)
        detector = ObjectDetector(threshold=DETECTION_THRESHOLD)
        
        print("Vision module active. Press Ctrl+C to stop.")
        
        while True:
            frame = stream.get_frame()
            if frame is None:
                print("Failed to capture frame. Retrying...")
                time.sleep(1)
                continue

            # Performance check
            start_time = time.time()
            prediction = detector.detect(frame)
            end_time = time.time()
            
            latency = end_time - start_time
            results = detector.get_labels(prediction)
            
            if results:
                print(f"[{time.strftime('%H:%M:%S')}] Detections (Latency: {latency:.3f}s):")
                for label, score in results:
                    print(f"  - {label}: {score:.2f}")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] No objects detected.")

            # Add a small delay for stability
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping vision module...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'stream' in locals():
            stream.release()

if __name__ == "__main__":
    main()
