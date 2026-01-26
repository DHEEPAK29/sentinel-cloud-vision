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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import StreamingResponse, HTMLResponse, Response
import base64
import io
import qrcode
import shutil
import sqlite3
import json
from pathlib import Path
from datetime import datetime
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Check if jax_train is available (dependencies installed)
try:
    import jax_train
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    print("Warning: JAX/Flax dependencies not found. Finetuning will be disabled.")

app = FastAPI(title="Sentinel Vision Module")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Prometheus Metrics ---
REQUEST_COUNT = Counter(
    "vision_request_count", "Total number of vision inference requests", ["method", "endpoint", "status"]
)
INFERENCE_LATENCY = Histogram(
    "vision_inference_latency_seconds", "Latency of object detection inference", ["source"]
)
FINETUNE_COUNT = Counter(
    "vision_finetune_total", "Total number of finetuning jobs started"
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
    # Ensure finetune storage directory exists
    FINETUNE_DIR = Path("finetuned_models")
    FINETUNE_DIR.mkdir(exist_ok=True)
    # Initialize SQLite DB
    DB_PATH = Path("finetune.db")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS finetune_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        target_object TEXT,
        dataset_path TEXT,
        result_path TEXT,
        timestamp TEXT
    )""")
    conn.commit()
    conn.close()

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
    
    if frame is None:
        REQUEST_COUNT.labels(method="GET", endpoint="/detect", status="400").inc()
        return {"error": "Failed to capture frame"}
    
    # Detect objects
    start_time = time.time()
    source_type = "mobile" if (mobile_source and mobile_source.active) else "webcam"
    with INFERENCE_LATENCY.labels(source=source_type).time():
        prediction = detector.detect(frame)
    
    latency = time.time() - start_time
    
    REQUEST_COUNT.labels(method="GET", endpoint="/detect", status="200").inc()
    
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

@app.post("/finetune")
async def finetune_model(dataset: UploadFile = File(...), target_object: str = Form(...)):
    FINETUNE_COUNT.inc()
    if not JAX_AVAILABLE:
        return {"status": "error", "message": "JAX/Flax libraries not installed on server."}
    
    contents = await dataset.read()
    
    # Save dataset file locally
    FINETUNE_DIR = Path("finetuned_models")
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_filename = f"{timestamp_str}_{target_object}{Path(dataset.filename).suffix}"
    dataset_path = FINETUNE_DIR / dataset_filename
    with open(dataset_path, "wb") as f:
        f.write(contents)

    # Run finetuning
    result = jax_train.run_finetuning(contents, target_object=target_object)
    
    # Save result JSON locally
    result_path = FINETUNE_DIR / f"{timestamp_str}_{target_object}_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    # Record in DB
    DB_PATH = Path("finetune.db")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO finetune_requests (target_object, dataset_path, result_path, timestamp) VALUES (?,?,?,?)",
        (target_object, str(dataset_path), str(result_path), timestamp_str)
    )
    conn.commit()
    conn.close()
    
    return {
        "target": target_object,
        "dataset_saved": str(dataset_path),
        "result_saved": str(result_path),
        "result": result,
        "message": f"Finetuning complete and saved locally for: {target_object}"
    }

@app.get("/admin/data")
async def get_admin_data():
    """Fetch all finetune records from DB."""
    DB_PATH = Path("finetune.db")
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM finetune_requests ORDER BY id DESC")
    rows = c.fetchall()
    data = [dict(row) for row in rows]
    conn.close()
    return data

@app.get("/admin", response_class=HTMLResponse)
async def admin_console():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Nova Sentinel | Admin Command Center</title>
        <style>
            :root {
                --primary-red: #ff0000;
                --bg-dark: #0a0a0a;
                --card-bg: #121212;
                --text-main: #f8f9fa;
                --accent: #E63946;
            }
            body { 
                background: var(--bg-dark); 
                color: var(--text-main); 
                font-family: 'Outfit', 'Inter', sans-serif; 
                margin: 0;
                padding: 20px;
                display: flex;
                flex-direction: column;
                gap: 20px;
            }
            .header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 2px solid var(--primary-red);
                padding-bottom: 15px;
            }
            h1 { margin: 0; color: var(--primary-red); letter-spacing: 2px; font-size: 1.8rem; }
            .nav-tabs {
                display: flex;
                gap: 10px;
            }
            .tab-btn {
                background: transparent;
                border: 1px solid var(--primary-red);
                color: var(--primary-red);
                padding: 8px 16px;
                cursor: pointer;
                transition: 0.3s;
                font-weight: bold;
            }
            .tab-btn.active {
                background: var(--primary-red);
                color: white;
            }
            .content-section {
                display: none;
                animation: fadeIn 0.5s ease-in-out;
            }
            .content-section.active { display: block; }
            @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
            
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }
            .nova-card {
                background: var(--card-bg);
                border: 1px solid #333;
                padding: 20px;
                border-radius: 4px;
                position: relative;
            }
            .nova-card::before {
                content: '';
                position: absolute;
                top: 0; left: 0; width: 4px; height: 100%;
                background: var(--primary-red);
            }
            
            table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
            th, td { padding: 12px; text-align: left; border: 1px solid #222; }
            th { background: #1a1a1a; color: var(--primary-red); font-size: 0.9rem; text-transform: uppercase; }
            tr:hover { background: #181818; }
            .badge { background: var(--primary-red); padding: 4px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: bold; }
            .path { font-family: 'Courier New', monospace; font-size: 0.8rem; color: #888; }
            
            .monitor-frame {
                width: 100%;
                height: 600px;
                border: 1px solid #333;
                background: #000;
            }
            .monitor-links {
                display: flex;
                gap: 15px;
                margin-bottom: 15px;
            }
            .monitor-link {
                color: #00ff00;
                text-decoration: none;
                font-family: monospace;
                border: 1px solid #00ff00;
                padding: 5px 10px;
                font-size: 0.9rem;
            }
            .monitor-link:hover { background: #00ff00; color: #000; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>NOVA SENTINEL // COMMAND CENTER</h1>
            <div class="nav-tabs">
                <button class="tab-btn active" onclick="showTab('logs')">FINETUNE LOGS</button>
                <button class="tab-btn" onclick="showTab('metrics')">SYSTEM MONITOR</button>
            </div>
        </div>

        <div id="logs" class="content-section active">
            <div class="nova-card">
                <h3>ACTIVE TRAINING REQUESTS</h3>
                <div id="loading">Initialising data stream...</div>
                <table id="logs-table" style="display:none;">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Target Object</th>
                            <th>Storage Path</th>
                            <th>Result Artifact</th>
                            <th>Timestamp</th>
                        </tr>
                    </thead>
                    <tbody id="logs-body"></tbody>
                </table>
            </div>
        </div>

        <div id="metrics" class="content-section">
            <div class="nova-card">
                <h3>REAL-TIME METRICS & TELEMETRY</h3>
                <div class="monitor-links">
                    <a href="http://localhost:9090" target="_blank" class="monitor-link">> PROMETHEUS_UI</a>
                    <a href="http://localhost:3000" target="_blank" class="monitor-link">> GRAFANA_DASHBOARD</a>
                    <a href="/metrics" target="_blank" class="monitor-link">> RAW_METRICS</a>
                </div>
                <iframe src="http://localhost:3000/d-solo/sentinel-dash?refresh=5s&theme=dark" class="monitor-frame" frameborder="0"></iframe>
                <p style="font-size: 0.8rem; color: #555; margin-top: 10px;">
                    * Note: Grafana iframe requires a predefined dashboard 'sentinel-dash'. If not found, please access the main UI via the link above.
                </p>
            </div>
        </div>

        <script>
            function showTab(tabId) {
                document.querySelectorAll('.content-section').forEach(s => s.classList.remove('active'));
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                document.getElementById(tabId).classList.add('active');
                event.target.classList.add('active');
            }

            async function loadData() {
                try {
                    const res = await fetch('/admin/data');
                    const data = await res.json();
                    const body = document.getElementById('logs-body');
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('logs-table').style.display = 'table';
                    body.innerHTML = '';
                    
                    data.forEach(row => {
                        const tr = document.createElement('tr');
                        tr.innerHTML = `
                            <td>${row.id}</td>
                            <td><span class="badge">${row.target_object}</span></td>
                            <td class="path">${row.dataset_path}</td>
                            <td class="path">${row.result_path}</td>
                            <td>${row.timestamp}</td>
                        `;
                        body.appendChild(tr);
                    });
                } catch (e) {
                    document.getElementById('loading').textContent = "DATA_STREAM_ERROR: Failed to connect to backend.";
                }
            }
            loadData();
            // Refresh logs every 30s
            setInterval(loadData, 30000);
        </script>
    </body>
    </html>
    """


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


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

