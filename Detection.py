# Detection.py
# Modern YOLOv5n object detection and tracking for Pi Zero 2 W
# Updated for Raspberry Pi Zero 2W with latest Raspbian and libcamera
import cv2
import torch
import time
import os
import sys
# Light CPU optimizations for Raspberry Pi CPUs
try:
    # Limit thread usage to avoid thrashing on small CPUs
    torch.set_num_threads(max(1, min(2, torch.get_num_threads())))
except Exception:
    pass
try:
    if hasattr(cv2, 'setNumThreads'):
        cv2.setNumThreads(1)
except Exception:
    pass
from collections import deque
import numpy as np
import subprocess
import threading

# --- CONFIGURABLE TOGGLES ---
REPORT_TO_TERMINAL = True  # Set to False to disable terminal reporting
SAVE_IMAGES = True         # Set to False to disable saving detected images
SAVE_DIR = 'detections'    # Directory to save images
CONFIDENCE_THRESHOLD = 0.3  # Increased threshold for better performance
TRACKER_BUFFER = 30        # Number of frames to keep track of objects
SHOW_LIVE_FEED = False     # Set to False for headless operation (Raspbian Lite)
USE_PICAMERA2 = True       # Use PiCamera2 if available (modern Pi camera library)
USE_OPENCV_FALLBACK = True # Use OpenCV as fallback for USB webcams

"""
Model loading strategy (Pi-friendly):
1) Prefer Ultralytics YOLOv8 if 'yolov8n.pt' exists and ultralytics is available
2) Fallback to YOLOv5n via torch.hub
"""
print("Loading detection model...")
MODEL_BACKEND = None  # 'v8' or 'v5'
model = None
try:
    from ultralytics import YOLO  # Prefer modern v8 when available
    if os.path.exists('yolov8n.pt'):
        model = YOLO('yolov8n.pt')
        MODEL_BACKEND = 'v8'
        print("YOLOv8n model loaded successfully!")
    else:
        raise ImportError("yolov8n.pt not found; using YOLOv5 fallback")
except Exception as e:
    print(f"YOLOv8 not used ({e}). Falling back to YOLOv5n via torch.hub...")
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True, force_reload=False)
        MODEL_BACKEND = 'v5'
        print("Pretrained YOLOv5n model loaded successfully!")
    except Exception as e2:
        print(f"Failed to load YOLOv5n: {e2}")
        sys.exit(1)

# Configure model thresholds/class filters where applicable
if MODEL_BACKEND == 'v5':
    model.conf = CONFIDENCE_THRESHOLD
    model.iou = 0.45  # NMS IoU threshold
    model.classes = [0, 15, 16]
    model.agnostic = False
    model.multi_label = False
    model.max_det = 20

# Only detect person, dog, cat (COCO classes: 0=person, 15=cat, 16=dog)
TARGET_CLASSES = {0: 'person', 15: 'cat', 16: 'dog'}

# --- SIMPLE TRACKER (centroid-based) ---
class CentroidTracker:
    def __init__(self, max_disappeared=TRACKER_BUFFER):
        self.next_object_id = 0
        self.objects = dict()  # object_id: centroid
        self.disappeared = dict()  # object_id: frames disappeared
        self.max_disappeared = max_disappeared
        self.seen_ids = set()

    def update(self, detections):
        # detections: list of (x1, y1, x2, y2, class_id, conf)
        input_centroids = []
        for det in detections:
            x1, y1, x2, y2, class_id, conf = det
            cX = int((x1 + x2) / 2)
            cY = int((y1 + y2) / 2)
            input_centroids.append((cX, cY))

        if len(input_centroids) == 0:
            # No detections, mark all as disappeared
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    del self.objects[oid]
                    del self.disappeared[oid]
            return list(self.objects.keys())

        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.objects[self.next_object_id] = centroid
                self.disappeared[self.next_object_id] = 0
                self.seen_ids.add(self.next_object_id)
                self.next_object_id += 1
            return list(self.objects.keys())

        # Try to match input centroids to existing objects
        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())
        used_object_ids = set()
        assigned_ids = []
        for centroid in input_centroids:
            min_dist = float('inf')
            min_id = None
            for oid, ocentroid in zip(object_ids, object_centroids):
                if oid in used_object_ids:
                    continue
                dist = np.linalg.norm(np.array(ocentroid) - np.array(centroid))
                if dist < min_dist and dist < 50:
                    min_dist = dist
                    min_id = oid
            if min_id is not None:
                self.objects[min_id] = centroid
                self.disappeared[min_id] = 0
                used_object_ids.add(min_id)
                assigned_ids.append(min_id)
            else:
                self.objects[self.next_object_id] = centroid
                self.disappeared[self.next_object_id] = 0
                self.seen_ids.add(self.next_object_id)
                assigned_ids.append(self.next_object_id)
                self.next_object_id += 1

        # Mark disappeared for unmatched objects
        for oid in list(self.objects.keys()):
            if oid not in assigned_ids:
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    del self.objects[oid]
                    del self.disappeared[oid]
        return assigned_ids

def check_camera_availability():
    """Check what camera options are available on the system"""
    print("Checking camera availability...")
    
    # Check if PiCamera2 is available (modern Pi camera library)
    try:
        import picamera2
        print("✓ PiCamera2 is available")
        return 'picamera2'
    except ImportError:
        pass
    
    # Check if v4l2 is available for USB webcams
    try:
        result = subprocess.run(['v4l2-ctl', '--list-devices'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and 'video' in result.stdout:
            print("✓ v4l2 camera devices found")
            return 'v4l2'
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    print("⚠ No camera interfaces found, trying OpenCV fallback")
    return 'opencv'



def setup_picamera2():
    """Setup PiCamera2 for modern Raspberry Pi"""
    try:
        from picamera2 import Picamera2
        from picamera2.encoders import JpegEncoder
        from picamera2.outputs import FileOutput
        
        picam2 = Picamera2()
        
        # Configure camera
        config = picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"},
            controls={"FrameDurationLimits": (100000, 100000)}  # 10 FPS
        )
        picam2.configure(config)
        picam2.start()
        
        print("✓ PiCamera2 initialized successfully")
        return picam2
    except Exception as e:
        print(f"✗ PiCamera2 setup failed: {e}")
        return None

def setup_opencv_camera():
    """Setup OpenCV camera as fallback"""
    print("Setting up OpenCV camera...")
    
    # Try different camera backends
    backends = [
        (cv2.CAP_V4L2, "V4L2"),
        (cv2.CAP_ANY, "ANY"),
        (cv2.CAP_GSTREAMER, "GStreamer")
    ]
    
    for backend, name in backends:
        try:
            cap = cv2.VideoCapture(0, backend)
            
            if not cap.isOpened():
                continue
                
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 10)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Test reading a frame
            ret, test_frame = cap.read()
            if ret:
                print(f"✓ OpenCV camera initialized with {name} backend")
                return cap
            else:
                cap.release()
                
        except Exception as e:
            print(f"✗ {name} backend failed: {e}")
            continue
    
    print("✗ All OpenCV backends failed")
    return None

def setup_camera():
    """Setup camera using the best available method"""
    camera_type = check_camera_availability()
    
    if camera_type == 'picamera2' and USE_PICAMERA2:
        return setup_picamera2()
    elif camera_type == 'v4l2' and USE_OPENCV_FALLBACK:
        return setup_opencv_camera()
    else:
        return setup_opencv_camera()



def read_frame_from_picamera2(picam2):
    """Read frame from PiCamera2"""
    try:
        frame = picam2.capture_array()
        return True, frame
    except Exception as e:
        print(f"Error reading from PiCamera2: {e}")
        return False, None

def read_frame_from_opencv(cap):
    """Read frame from OpenCV camera"""
    try:
        ret, frame = cap.read()
        return ret, frame
    except Exception as e:
        print(f"Error reading from OpenCV camera: {e}")
        return False, None

def main(show_live_feed=SHOW_LIVE_FEED):
    print("Starting modern detection system for Raspberry Pi Zero 2W...")
    
    if SAVE_IMAGES and not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"Created directory: {SAVE_DIR}")

    # Setup camera
    camera = setup_camera()
    if camera is None:
        print("ERROR: No camera available!")
        return

    tracker = CentroidTracker()
    frame_count = 0
    last_report_time = time.time()
    report_interval = 1.0  # Report every second to avoid spam
    
    # Determine camera type for reading frames
    camera_type = 'opencv'
    if hasattr(camera, 'capture_array'):  # PiCamera2
        camera_type = 'picamera2'

    print(f"Detection system ready using {camera_type}! Press Ctrl+C to stop.")
    
    try:
        while True:
            # Read frame based on camera type
            if camera_type == 'picamera2':
                ret, frame = read_frame_from_picamera2(camera)
            else:
                ret, frame = read_frame_from_opencv(camera)
            
            if not ret:
                print("Failed to read frame from camera")
                time.sleep(0.1)
                continue
                
            frame_count += 1
            
            # Process frame with YOLO (supports YOLOv8 and YOLOv5 backends)
            detections = []
            try:
                if MODEL_BACKEND == 'v8':
                    v8_results = model.predict(
                        frame,
                        conf=CONFIDENCE_THRESHOLD,
                        classes=list(TARGET_CLASSES.keys()),
                        verbose=False
                    )
                    if v8_results and len(v8_results) > 0:
                        boxes = v8_results[0].boxes
                        if boxes is not None and boxes.xyxy is not None:
                            for i in range(len(boxes)):
                                x1, y1, x2, y2 = map(int, boxes.xyxy[i].cpu().numpy())
                                conf = float(boxes.conf[i].cpu().numpy()) if boxes.conf is not None else 0.0
                                class_id = int(boxes.cls[i].cpu().numpy()) if boxes.cls is not None else -1
                                if class_id in TARGET_CLASSES and conf >= CONFIDENCE_THRESHOLD:
                                    detections.append((x1, y1, x2, y2, class_id, conf))
                else:
                    v5_results = model(frame)
                    for *xyxy, conf, cls in v5_results.xyxy[0].cpu().numpy():
                        class_id = int(cls)
                        if class_id in TARGET_CLASSES and conf >= CONFIDENCE_THRESHOLD:
                            x1, y1, x2, y2 = map(int, xyxy)
                            detections.append((x1, y1, x2, y2, class_id, conf))
            except Exception as infer_e:
                print(f"Inference error: {infer_e}")
                detections = []

            object_ids = tracker.update(detections)
            current_time = time.time()

            # Report detections (with rate limiting)
            if REPORT_TO_TERMINAL and detections and (current_time - last_report_time) >= report_interval:
                print(f"\n--- Detection Report (Frame {frame_count}) ---")
                for i, det in enumerate(detections):
                    x1, y1, x2, y2, class_id, conf = det
                    label = f"{TARGET_CLASSES[class_id]} {conf:.2f}"
                    print(f"Detected {label} at [{x1},{y1},{x2},{y2}] (ID: {object_ids[i] if i < len(object_ids) else 'N/A'})")
                last_report_time = current_time

            # Save images
            if SAVE_IMAGES and detections:
                for i, det in enumerate(detections):
                    x1, y1, x2, y2, class_id, conf = det
                    if i < len(object_ids):
                        img_path = os.path.join(SAVE_DIR, f"{TARGET_CLASSES[class_id]}_{object_ids[i]}_{frame_count}.jpg")
                        try:
                            cv2.imwrite(img_path, frame[y1:y2, x1:x2])
                            if REPORT_TO_TERMINAL:
                                print(f"Saved image: {img_path}")
                        except Exception as e:
                            print(f"Failed to save image: {e}")

            # Draw bounding boxes and labels
            for i, det in enumerate(detections):
                x1, y1, x2, y2, class_id, conf = det
                label = f"{TARGET_CLASSES[class_id]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            # Show live feed if enabled
            if show_live_feed:
                cv2.imshow('Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Add small delay to prevent overwhelming the Pi
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nStopping detection system...")
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        # Clean up camera resources
        if camera_type == 'picamera2':
            if hasattr(camera, 'close'):
                camera.close()
        else:
            if hasattr(camera, 'release'):
                camera.release()
        
        if show_live_feed:
            cv2.destroyAllWindows()
        print("Detection system stopped.")

if __name__ == '__main__':
    main()
