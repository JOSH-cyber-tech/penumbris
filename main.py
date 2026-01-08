"""
PENUMBRIS Backend - Production FastAPI WebSocket Server
Streams OpenCV frames with real-time object detection and enhancement
"""

import asyncio
import base64
import json
import logging
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Detection:
    """Single object detection"""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str
    threat_level: float = 0.0

@dataclass
class FramePayload:
    """Complete frame data sent to frontend"""
    timestamp: float
    frame_id: int
    original_frame: str  # base64 encoded
    enhanced_frame: str  # base64 encoded
    detections: List[Dict]
    alert: Optional[Dict] = None

# ============================================================================
# COMPUTER VISION PIPELINE
# ============================================================================

class SimpleEnhancer:
    """Lightweight low-light enhancement (placeholder for Zero-DCE)"""
    
    @staticmethod
    def enhance(frame: np.ndarray) -> np.ndarray:
        """
        Fast CLAHE-based enhancement as Zero-DCE alternative
        Replace with actual Zero-DCE model for production
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Merge and convert back
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # Boost brightness slightly
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=10)
        
        return enhanced

class YOLODetector:
    """YOLOv8-based object detector"""
    
    # COCO class names (80 classes)
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    # High-risk objects for threat detection
    THREAT_OBJECTS = {
        'knife': 0.9,
        'scissors': 0.7,
        'baseball bat': 0.6,
        'person': 0.1,  # Base threat for person detection
    }
    
    def __init__(self, model_path: str = 'yolov8n.pt', confidence: float = 0.25):
        """Initialize YOLO detector"""
        self.confidence = confidence
        self.model_loaded = False
        
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.model_loaded = True
            logger.info(f"✅ YOLO model loaded: {model_path}")
        except ImportError:
            logger.warning("⚠️  ultralytics not installed, using mock detections")
            self.model = None
        except Exception as e:
            logger.error(f"❌ Failed to load YOLO: {e}")
            self.model = None
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on frame"""
        if not self.model_loaded:
            return self._mock_detections(frame)
        
        try:
            # Run inference
            results = self.model(frame, verbose=False, conf=self.confidence)[0]
            
            detections = []
            boxes = results.boxes
            
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                cls_name = self.COCO_CLASSES[cls_id]
                
                # Calculate threat level
                threat = self._calculate_threat(cls_name, conf)
                
                detections.append(Detection(
                    x1=box[0], y1=box[1], x2=box[2], y2=box[3],
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                    threat_level=threat
                ))
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    def _calculate_threat(self, class_name: str, confidence: float) -> float:
        """Calculate threat level for detected object"""
        base_threat = self.THREAT_OBJECTS.get(class_name, 0.0)
        return min(base_threat * confidence, 1.0)
    
    def _mock_detections(self, frame: np.ndarray) -> List[Detection]:
        """Generate mock detections for demo (when YOLO unavailable)"""
        h, w = frame.shape[:2]
        
        # Simulate 1-3 detections
        num_dets = np.random.randint(1, 4)
        detections = []
        
        mock_classes = ['person', 'car', 'backpack', 'bottle']
        
        for _ in range(num_dets):
            x1 = np.random.randint(0, w - 200)
            y1 = np.random.randint(0, h - 200)
            x2 = x1 + np.random.randint(100, 200)
            y2 = y1 + np.random.randint(150, 250)
            
            cls_name = np.random.choice(mock_classes)
            conf = 0.65 + np.random.random() * 0.3
            
            detections.append(Detection(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=conf,
                class_id=0,
                class_name=cls_name,
                threat_level=self._calculate_threat(cls_name, conf)
            ))
        
        return detections

class PenumbrisPipeline:
    """Main CV pipeline orchestrator"""
    
    def __init__(self):
        self.enhancer = SimpleEnhancer()
        self.detector = YOLODetector()
        self.frame_count = 0
        logger.info("🚀 PENUMBRIS Pipeline initialized")
    
    async def process_frame(self, frame: np.ndarray) -> FramePayload:
        """Process single frame through pipeline"""
        self.frame_count += 1
        
        # Run enhancement and detection in parallel
        loop = asyncio.get_event_loop()
        
        enhanced_task = loop.run_in_executor(None, self.enhancer.enhance, frame)
        detection_task = loop.run_in_executor(None, self.detector.detect, frame)
        
        enhanced_frame, detections = await asyncio.gather(
            enhanced_task, 
            detection_task
        )
        
        # Encode frames to base64
        original_b64 = self._encode_frame(frame)
        enhanced_b64 = self._encode_frame(enhanced_frame)
        
        # Convert detections to dict
        detections_dict = [asdict(d) for d in detections]
        
        # Generate alert if high threat
        alert = self._generate_alert(detections)
        
        return FramePayload(
            timestamp=time.time(),
            frame_id=self.frame_count,
            original_frame=original_b64,
            enhanced_frame=enhanced_b64,
            detections=detections_dict,
            alert=alert
        )
    
    @staticmethod
    def _encode_frame(frame: np.ndarray, quality: int = 80) -> str:
        """Encode frame to base64 JPEG"""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buffer).decode('utf-8')
    
    @staticmethod
    def _generate_alert(detections: List[Detection]) -> Optional[Dict]:
        """Generate alert if threat detected"""
        high_threat = [d for d in detections if d.threat_level > 0.5]
        
        if high_threat:
            max_threat = max(high_threat, key=lambda x: x.threat_level)
            return {
                'type': 'CRITICAL' if max_threat.threat_level > 0.7 else 'WARNING',
                'message': f'{max_threat.class_name.upper()} detected in Sector A',
                'threat_level': max_threat.threat_level,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            }
        return None

# ============================================================================
# VIDEO SOURCE MANAGER
# ============================================================================

class VideoSource:
    """Manages video input (camera or file)"""
    
    def __init__(self, source: str = "0", fps: int = 30):
        """
        Initialize video source
        source: camera index (0, 1, 2) or video file path
        """
        self.source = source
        self.target_fps = fps
        self.frame_delay = 1.0 / fps
        
        # Try to parse as camera index
        try:
            self.source = int(source)
        except ValueError:
            pass  # It's a file path
        
        self.cap = None
        self.is_running = False
        logger.info(f"📹 Video source configured: {self.source} @ {fps}fps")
    
    def start(self) -> bool:
        """Start video capture"""
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            logger.error(f"❌ Failed to open video source: {self.source}")
            return False
        
        # Set properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        self.is_running = True
        logger.info("✅ Video capture started")
        return True
    
    async def read_frame(self) -> Optional[np.ndarray]:
        """Read next frame asynchronously"""
        if not self.is_running or self.cap is None:
            return None
        
        # Read in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        ret, frame = await loop.run_in_executor(None, self.cap.read)
        
        if not ret:
            # Loop video if file
            if isinstance(self.source, str):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = await loop.run_in_executor(None, self.cap.read)
            
            if not ret:
                logger.warning("⚠️  Failed to read frame")
                return None
        
        return frame
    
    def stop(self):
        """Stop video capture"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        logger.info("⏹️  Video capture stopped")

# ============================================================================
# WEBSOCKET CONNECTION MANAGER
# ============================================================================

class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Register new connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"✅ Client connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove connection"""
        self.active_connections.remove(websocket)
        logger.info(f"❌ Client disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        """Send message to all connected clients"""
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send to client: {e}")
                disconnected.append(connection)
        
        # Clean up dead connections
        for conn in disconnected:
            self.disconnect(conn)

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(title="PENUMBRIS", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
manager = ConnectionManager()
pipeline = PenumbrisPipeline()
video_source = VideoSource(source="test_video.mp4", fps=15)  # Change to 0 for webcam

# ============================================================================
# ROUTES
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "SYSTEM ACTIVE",
        "service": "PENUMBRIS",
        "version": "1.0.0",
        "active_connections": len(manager.active_connections)
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for video streaming"""
    await manager.connect(websocket)
    
    try:
        # Keep connection alive and listen for client messages
        while True:
            try:
                # Receive any client messages (optional)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                logger.info(f"Received from client: {data}")
            except asyncio.TimeoutError:
                pass  # No message, continue
            
            await asyncio.sleep(0.01)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# ============================================================================
# BACKGROUND TASKS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    logger.info("🚀 PENUMBRIS system starting...")
    
    # Start video source
    if not video_source.start():
        logger.error("⚠️  Using demo mode - no video source")
    
    # Start frame processing loop
    asyncio.create_task(frame_processing_loop())
    
    logger.info("✅ System ready")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("⏹️  Shutting down PENUMBRIS...")
    video_source.stop()

async def frame_processing_loop():
    """Main loop: capture frames and broadcast"""
    logger.info("🎬 Frame processing loop started")
    
    while True:
        try:
            # Only process if clients connected (save resources)
            if len(manager.active_connections) == 0:
                await asyncio.sleep(1)
                continue
            
            # Capture frame
            frame = await video_source.read_frame()
            
            if frame is None:
                logger.warning("⚠️  No frame available")
                await asyncio.sleep(0.5)
                continue
            
            # Process through pipeline
            payload = await pipeline.process_frame(frame)
            
            # Broadcast to all clients
            await manager.broadcast(asdict(payload))
            
            # Control frame rate
            await asyncio.sleep(video_source.frame_delay)
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            await asyncio.sleep(1)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )