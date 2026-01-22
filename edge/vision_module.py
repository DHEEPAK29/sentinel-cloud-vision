import cv2
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
import numpy as np
import time
import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import base64
import io

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Sentinel Vision Module")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class WebcamStream:
    """Handles webcam frame capture using OpenCV."""
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            print(f"Warning: Could not open webcam {src}. Using dummy stream.")
            self.stream = None
        
    def get_frame(self):
        if self.stream is None:
            # Return dummy black frame with noise or text to indicate no camera
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "NO CAMERA", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return frame

        ret, frame = self.stream.read()
        if not ret:
            return None
        return frame

    def release(self):
        if self.stream:
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
        boxes = prediction['boxes'].tolist()
        return list(zip(labels, scores, boxes))

# Global instances
stream = None
detector = None

@app.on_event("startup")
async def startup_event():
    global stream, detector
    WEBCAM_INDEX = int(os.getenv("WEBCAM_INDEX", 0))
    DETECTION_THRESHOLD = float(os.getenv("DETECTION_THRESHOLD", 0.7))
    stream = WebcamStream(src=WEBCAM_INDEX)
    detector = ObjectDetector(threshold=DETECTION_THRESHOLD)

@app.get("/detect")
async def detect_objects():
    """Capture frame and return detection results"""
    if stream is None or detector is None:
        return {"error": "Vision module not initialized"}
    
    frame = stream.get_frame()
    if frame is None:
        return {"error": "Failed to capture frame"}
    
    # Detect objects
    start_time = time.time()
    prediction = detector.detect(frame)
    latency = time.time() - start_time
    
    results = detector.get_labels(prediction)
    
    # Encode frame as base64
    _, buffer = cv2.imencode('.jpg', frame)
    frame_b64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "detections": [{"label": label, "score": score, "box": box} for label, score, box in results],
        "latency": round(latency, 3),
        "frame": frame_b64,
        "timestamp": time.strftime('%H:%M:%S')
    }

@app.get("/")
async def root():
    return {"status": "Vision Module Active", "endpoints": ["/detect"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

