import cv2
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
import numpy as np
import time
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse, Response
import base64
import io
import qrcode

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

class MobileFrameSource:
    """Handles frames received via WebSocket from a mobile device."""
    def __init__(self):
        self.last_frame = None
        self.active = False
        self.last_update = 0
    
    def update_frame(self, frame_bytes):
        # Decode image from bytes
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is not None:
             self.last_frame = frame
             self.last_update = time.time()
             self.active = True

    def get_frame(self):
        # Timeout if no frame for 5 seconds
        if time.time() - self.last_update > 5:
            self.active = False
            return None
        return self.last_frame

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
# Global instances
stream = None
mobile_source = None
detector = None

@app.on_event("startup")
async def startup_event():
    global stream, detector, mobile_source
    WEBCAM_INDEX = int(os.getenv("WEBCAM_INDEX", 0))
    DETECTION_THRESHOLD = float(os.getenv("DETECTION_THRESHOLD", 0.7))
    stream = WebcamStream(src=WEBCAM_INDEX)
    mobile_source = MobileFrameSource()
    detector = ObjectDetector(threshold=DETECTION_THRESHOLD)

@app.get("/detect")
async def detect_objects():
    """Capture frame and return detection results"""
    if stream is None or detector is None:
        return {"error": "Vision module not initialized"}
    
    # Prioritize mobile source if active
    frame = None
    if mobile_source and mobile_source.active:
        frame = mobile_source.get_frame()
    
    if frame is None:
        frame = stream.get_frame()
        

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
        "latency": round(latency, 3),
        "frame": frame_b64,
        "timestamp": time.strftime('%H:%M:%S'),
        "source": "mobile" if (mobile_source and mobile_source.active) else "webcam"
    }

@app.websocket("/mobile-stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            mobile_source.update_frame(data)
    except WebSocketDisconnect:
        print("Mobile client disconnected")
        pass

@app.get("/qr-code")
async def get_qr_code():
    # Generate QR code for the mobile capture page
    # Assuming the server is reachable via local network IP or localhost for demo
    # In a real scenario, this IP needs to be the LAN IP of the host machine
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    # Try to find a non-loopback IP (simple heuristic)
    try:
         s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
         s.connect(("8.8.8.8", 80))
         local_ip = s.getsockname()[0]
         s.close()
    except:
         pass

    url = f"http://{local_ip}:8001/mobile-capture"
    
    img = qrcode.make(url)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")

@app.get("/mobile-capture", response_class=HTMLResponse)
async def mobile_capture_page():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sentinel Mobile Camera</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <style>
            body { margin: 0; background: #000; color: white; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; font-family: sans-serif; }
            video { width: 100%; max-height: 80vh; object-fit: contain; }
            button { padding: 15px 30px; font-size: 1.2rem; background: #ff0000; color: white; border: none; border-radius: 5px; margin-top: 20px; }
            #status { margin-top: 10px; color: #aaa; }
        </style>
    </head>
    <body>
        <video id="video" autoplay playsinline muted></video>
        <div id="status">Ready to connect</div>
        <button onclick="startStreaming()">START STREAMING</button>
        <script>
            const video = document.getElementById('video');
            const status = document.getElementById('status');
            let ws;
            let canvas = document.createElement('canvas');
            
            async function startStreaming() {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
                    video.srcObject = stream;
                    
                    // Connect WS
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    ws = new WebSocket(`${protocol}//${window.location.host}/mobile-stream`);
                    
                    ws.onopen = () => {
                        status.textContent = "CONNECTED - STREAMING";
                        status.style.color = "#00ff00";
                        sendFrame();
                    };
                    
                    ws.onclose = () => {
                        status.textContent = "DISCONNECTED";
                        status.style.color = "red";
                    };
                    
                } catch (e) {
                    status.textContent = "Error: " + e.message;
                }
            }
            
            function sendFrame() {
                if (ws.readyState === WebSocket.OPEN) {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(video, 0, 0);
                    
                    canvas.toBlob(blob => {
                        if (ws.readyState === WebSocket.OPEN) ws.send(blob);
                        requestAnimationFrame(sendFrame); // Loop
                    }, 'image/jpeg', 0.5); // Compress to 0.5 quality
                }
            }
        </script>
    </body>
    </html>
    """

@app.get("/")
async def root():
    return {"status": "Vision Module Active", "endpoints": ["/detect"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

