# Detection.py - OPTIMIZED FOR RASPBERRY PI ZERO 2W
# Ultra-lightweight YOLOv5n object detection with aggressive optimizations
# Specifically tuned for Pi Zero 2W's limited resources (1GB RAM, 4-core ARM)

import cv2
import torch
import time
import os
import sys
import subprocess
import threading
from collections import deque
import numpy as np
import gc

# AGGRESSIVE CPU/MEMORY OPTIMIZATIONS FOR PI ZERO 2W
try:
    # Limit PyTorch to single thread to prevent CPU thrashing
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

try:
    # Force OpenCV to single thread
    cv2.setNumThreads(1)
    cv2.setUseOptimized(True)
except Exception:
    pass

# Force garbage collection more frequently
gc.set_threshold(100, 5, 5)

# --- PI ZERO 2W OPTIMIZED SETTINGS ---
REPORT_TO_TERMINAL = True
SAVE_IMAGES = False  # Disabled by default - I/O is expensive on Pi Zero 2W
SAVE_DIR = 'detections'
CONFIDENCE_THRESHOLD = 0.4  # Higher threshold = less processing
TRACKER_BUFFER = 15  # Reduced buffer to save memory
SHOW_LIVE_FEED = False  # Disabled for headless operation
USE_PICAMERA2 = True
USE_OPENCV_FALLBACK = True

# ULTRA-CONSERVATIVE CAMERA SETTINGS FOR PI ZERO 2W
CAMERA_WIDTH = 320   # Smaller resolution = faster processing
CAMERA_HEIGHT = 240
CAMERA_FPS = 3       # Very low FPS to prevent overwhelming the CPU
FRAME_SKIP = 2       # Process every Nth frame only

# Memory management
MAX_DETECTION_HISTORY = 5  # Limit detection history
FORCE_GC_INTERVAL = 20     # Force garbage collection every N frames

"""
Model loading strategy optimized for Pi Zero 2W:
Prefer smaller, faster models and aggressive caching
"""
print("Loading lightweight detection model for Pi Zero 2W...")
MODEL_BACKEND = None
model = None

# Try YOLOv8n first (usually smaller and faster)
try:
    from ultralytics import YOLO
    if os.path.exists('yolov8n.pt'):
        model = YOLO('yolov8n.pt')
        MODEL_BACKEND = 'v8'
        print("YOLOv8n model loaded - optimizing for Pi Zero 2W...")
        
        # Ultra-aggressive YOLOv8 optimizations
        model.fuse()  # Fuse model layers for speed
        model.model.eval()  # Set to evaluation mode
        
    else:
        raise ImportError("yolov8n.pt not found; using YOLOv5 fallback")
except Exception as e:
    print(f"YOLOv8 not available ({e}). Trying YOLOv5n...")
    try:
        # Load YOLOv5 with minimal memory footprint
        model = torch.hub.load('ultralytics/yolov5', 'yolov5n', 
                              pretrained=True, force_reload=False,
                              _verbose=False)
        MODEL_BACKEND = 'v5'
        print("YOLOv5n loaded - applying Pi Zero 2W optimizations...")
        
        # Aggressive YOLOv5 optimizations
        model.eval()
        model.half()  # Use half precision if supported
        
    except Exception as e2:
        print(f"Failed to load any YOLO model: {e2}")
        sys.exit(1)

# Configure model for Pi Zero 2W performance
if MODEL_BACKEND == 'v5':
    model.conf = CONFIDENCE_THRESHOLD
    model.iou = 0.5  # Higher IoU = fewer detections = faster
    model.classes = [0]  # Only detect people to reduce processing
    model.agnostic = False
    model.multi_label = False
    model.max_det = 5  # Limit to 5 detections max

# Only detect people for Pi Zero 2W (cats/dogs removed to save processing)
TARGET_CLASSES = {0: 'person'}

# --- ULTRA-LIGHTWEIGHT TRACKER ---
class MinimalTracker:
    def __init__(self, max_disappeared=TRACKER_BUFFER):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.last_cleanup = time.time()

    def update(self, detections):
        current_time = time.time()
        
        # Cleanup old objects more frequently
        if current_time - self.last_cleanup > 2.0:
            self._cleanup_objects()
            self.last_cleanup = current_time
        
        if not detections:
            # Mark all as disappeared
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.objects.pop(oid, None)
                    self.disappeared.pop(oid, None)
            return []

        # Simple centroid calculation
        input_centroids = [(int((x1 + x2) / 2), int((y1 + y2) / 2)) 
                          for x1, y1, x2, y2, _, _ in detections]

        if not self.objects:
            # Create new objects
            result_ids = []
            for centroid in input_centroids:
                self.objects[self.next_object_id] = centroid
                self.disappeared[self.next_object_id] = 0
                result_ids.append(self.next_object_id)
                self.next_object_id += 1
            return result_ids

        # Simple nearest neighbor matching (faster than scipy)
        object_ids = list(self.objects.keys())
        used_ids = set()
        result_ids = []
        
        for centroid in input_centroids:
            best_id = None
            min_dist = float('inf')
            
            for oid in object_ids:
                if oid in used_ids:
                    continue
                old_centroid = self.objects[oid]
                dist = ((centroid[0] - old_centroid[0]) ** 2 + 
                       (centroid[1] - old_centroid[1]) ** 2) ** 0.5
                
                if dist < min_dist and dist < 30:  # Reduced distance threshold
                    min_dist = dist
                    best_id = oid
            
            if best_id is not None:
                self.objects[best_id] = centroid
                self.disappeared[best_id] = 0
                used_ids.add(best_id)
                result_ids.append(best_id)
            else:
                # New object
                self.objects[self.next_object_id] = centroid
                self.disappeared[self.next_object_id] = 0
                result_ids.append(self.next_object_id)
                self.next_object_id += 1

        # Mark unused objects as disappeared
        for oid in object_ids:
            if oid not in used_ids:
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.objects.pop(oid, None)
                    self.disappeared.pop(oid, None)

        return result_ids

    def _cleanup_objects(self):
        """Force cleanup of old objects"""
        to_remove = [oid for oid, disappeared in self.disappeared.items() 
                    if disappeared > self.max_disappeared]
        for oid in to_remove:
            self.objects.pop(oid, None)
            self.disappeared.pop(oid, None)

# --- PI ZERO 2W SPECIFIC CAMERA FUNCTIONS ---
def detect_pi_zero_camera():
    """Detect camera specifically for Pi Zero 2W"""
    print("Detecting camera on Raspberry Pi Zero 2W...")
    
    # Check if we're actually on a Pi Zero 2W
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpu_info = f.read()
            if 'BCM2837' not in cpu_info and 'Raspberry Pi Zero 2' not in cpu_info:
                print("Warning: Not detected as Pi Zero 2W")
    except:
        pass

    # Get camera status
    try:
        result = subprocess.run(['vcgencmd', 'get_camera'], 
                              capture_output=True, text=True, timeout=3)
        print(f"Camera status: {result.stdout.strip()}")
    except:
        print("Could not get camera status")

    # Find video devices (Pi Zero 2W typically uses /dev/video0 and /dev/video10-12)
    pi_zero_devices = []
    priority_devices = [0, 10, 11, 12, 1, 2]  # Common Pi Zero 2W device numbers
    
    for device_id in priority_devices:
        device_path = f"/dev/video{device_id}"
        if os.path.exists(device_path):
            pi_zero_devices.append(device_id)
            print(f"Found video device: {device_path}")

    return pi_zero_devices

def setup_picamera2_pi_zero():
    """PiCamera2 setup optimized for Pi Zero 2W"""
    if not USE_PICAMERA2:
        return None
        
    try:
        from picamera2 import Picamera2
        print("Setting up PiCamera2 for Pi Zero 2W...")
        
        # Check for cameras
        cameras = Picamera2.global_camera_info()
        if not cameras:
            print("No cameras found by PiCamera2")
            return None
        
        print(f"Found cameras: {len(cameras)}")
        
        picam2 = Picamera2()
        
        # Ultra-lightweight config for Pi Zero 2W
        config = picam2.create_preview_configuration(
            main={
                "size": (CAMERA_WIDTH, CAMERA_HEIGHT), 
                "format": "RGB888"
            },
            controls={
                "FrameRate": CAMERA_FPS,
                "ExposureTime": 15000,  # Faster exposure
                "AnalogueGain": 1.0,
                "NoiseReductionMode": 0,  # Disable noise reduction
                "Sharpness": 0.5,
                "Brightness": 0.0,
                "Contrast": 1.0
            }
        )
        
        picam2.configure(config)
        picam2.start()
        
        # Test with longer timeout for Pi Zero 2W
        print("Waiting for camera to initialize...")
        time.sleep(4)
        
        # Test frame capture
        frame = picam2.capture_array()
        if frame is not None and frame.shape[0] > 0:
            print(f"✓ PiCamera2 working: {frame.shape}")
            return picam2
        else:
            print("PiCamera2 test capture failed")
            picam2.close()
            return None
            
    except ImportError:
        print("PiCamera2 not installed")
        return None
    except Exception as e:
        print(f"PiCamera2 failed: {e}")
        return None

def setup_opencv_pi_zero():
    """OpenCV setup optimized for Pi Zero 2W"""
    if not USE_OPENCV_FALLBACK:
        return None
        
    print("Setting up OpenCV for Pi Zero 2W...")
    
    pi_devices = detect_pi_zero_camera()
    
    # Pi Zero 2W specific backends (in order of preference)
    backends = [
        (cv2.CAP_V4L2, "V4L2"),
        (cv2.CAP_V4L, "V4L"),
        (cv2.CAP_ANY, "ANY")
    ]
    
    # Try each device with each backend
    for device_id in pi_devices:
        print(f"Trying /dev/video{device_id}...")
        
        for backend, name in backends:
            try:
                cap = cv2.VideoCapture(device_id, backend)
                
                if not cap.isOpened():
                    continue
                
                # Pi Zero 2W optimized settings
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
                
                # Try MJPEG for better performance on Pi Zero 2W
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
                
                # Test capture
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✓ OpenCV {name} working on /dev/video{device_id}")
                    print(f"Frame shape: {frame.shape}")
                    return cap
                
                cap.release()
                
            except Exception as e:
                print(f"Backend {name} failed: {e}")
    
    # Try GStreamer pipelines specific to Pi Zero 2W
    print("Trying Pi Zero 2W GStreamer pipelines...")
    gst_pipelines = [
        f"libcamerasrc ! video/x-raw,width={CAMERA_WIDTH},height={CAMERA_HEIGHT},framerate={CAMERA_FPS}/1 ! videoconvert ! appsink drop=1",
        f"v4l2src device=/dev/video0 ! video/x-raw,width={CAMERA_WIDTH},height={CAMERA_HEIGHT},framerate={CAMERA_FPS}/1 ! videoconvert ! appsink drop=1",
        f"v4l2src device=/dev/video10 ! video/x-raw,width={CAMERA_WIDTH},height={CAMERA_HEIGHT},framerate={CAMERA_FPS}/1 ! videoconvert ! appsink drop=1",
    ]
    
    for pipeline in gst_pipelines:
        try:
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✓ GStreamer working: {frame.shape}")
                    return cap
                cap.release()
        except Exception as e:
            print(f"GStreamer failed: {e}")
    
    print("✗ All OpenCV methods failed")
    return None

def setup_camera():
    """Main camera setup for Pi Zero 2W"""
    print("=== RASPBERRY PI ZERO 2W CAMERA SETUP ===")
    
    # Try PiCamera2 first (best performance on Pi Zero 2W)
    camera = setup_picamera2_pi_zero()
    if camera:
        return camera, 'picamera2'
    
    # Fallback to OpenCV
    camera = setup_opencv_pi_zero()
    if camera:
        return camera, 'opencv'
    
    print("ERROR: No camera method worked!")
    print("\nTroubleshooting for Pi Zero 2W:")
    print("1. sudo raspi-config -> Interface Options -> Camera -> Enable")
    print("2. Add 'start_x=1' to /boot/config.txt")
    print("3. Add 'gpu_mem=128' to /boot/config.txt") 
    print("4. sudo reboot")
    print("5. Check: vcgencmd get_camera")
    
    return None, None

def read_frame_optimized(camera, camera_type):
    """Optimized frame reading for Pi Zero 2W"""
    try:
        if camera_type == 'picamera2':
            frame = camera.capture_array()
            return True, frame
        else:
            ret, frame = camera.read()
            return ret, frame
    except Exception as e:
        print(f"Frame read error: {e}")
        return False, None

def process_detection_lightweight(model, frame):
    """Ultra-lightweight detection processing for Pi Zero 2W"""
    detections = []
    
    try:
        if MODEL_BACKEND == 'v8':
            # YOLOv8 with minimal settings
            results = model.predict(
                frame,
                conf=CONFIDENCE_THRESHOLD,
                classes=[0],  # Person only
                verbose=False,
                save=False,
                show=False,
                stream=False,
                device='cpu',
                half=False,  # Avoid half precision issues on Pi Zero 2W
                imgsz=320,   # Small input size
                max_det=3    # Maximum 3 detections
            )
            
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and boxes.xyxy is not None:
                    for i in range(min(len(boxes), 3)):  # Limit to 3 detections
                        x1, y1, x2, y2 = map(int, boxes.xyxy[i].cpu().numpy())
                        conf = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())
                        
                        if class_id == 0 and conf >= CONFIDENCE_THRESHOLD:
                            detections.append((x1, y1, x2, y2, class_id, conf))
        
        else:
            # YOLOv5 processing
            results = model(frame, size=320)  # Small input size
            for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
                class_id = int(cls)
                if class_id == 0 and conf >= CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, xyxy)
                    detections.append((x1, y1, x2, y2, class_id, conf))
                    if len(detections) >= 3:  # Limit detections
                        break
    
    except Exception as e:
        print(f"Detection error: {e}")
    
    return detections

def main(show_live_feed=SHOW_LIVE_FEED):
    print("=== STARTING PI ZERO 2W OPTIMIZED DETECTION SYSTEM ===")
    print(f"Camera resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS}FPS")
    print(f"Processing every {FRAME_SKIP} frames")
    print(f"Detecting: {list(TARGET_CLASSES.values())}")
    
    if SAVE_IMAGES and not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Setup camera
    camera, camera_type = setup_camera()
    if camera is None:
        return

    tracker = MinimalTracker()
    frame_count = 0
    process_count = 0
    last_report_time = time.time()
    last_gc_time = time.time()
    
    print(f"System ready using {camera_type}! Press Ctrl+C to stop.")
    
    try:
        while True:
            # Read frame
            ret, frame = read_frame_optimized(camera, camera_type)
            
            if not ret:
                print("Failed to read frame")
                time.sleep(0.2)
                continue
            
            frame_count += 1
            current_time = time.time()
            
            # Skip frames for performance (process every Nth frame)
            if frame_count % FRAME_SKIP != 0:
                time.sleep(0.1)
                continue
            
            process_count += 1
            
            # Force garbage collection periodically
            if current_time - last_gc_time > 5.0:
                gc.collect()
                last_gc_time = current_time
            
            # Process detection
            detections = process_detection_lightweight(model, frame)
            object_ids = tracker.update(detections)
            
            # Report detections (rate limited)
            if REPORT_TO_TERMINAL and detections and (current_time - last_report_time) >= 2.0:
                print(f"\n--- Frame {frame_count} (Processed: {process_count}) ---")
                for i, det in enumerate(detections):
                    x1, y1, x2, y2, class_id, conf = det
                    label = f"{TARGET_CLASSES[class_id]} {conf:.2f}"
                    obj_id = object_ids[i] if i < len(object_ids) else 'N/A'
                    print(f"Detected {label} at [{x1},{y1},{x2},{y2}] (ID: {obj_id})")
                
                # Performance stats
                fps = process_count / (current_time - (time.time() - current_time + process_count * 0.1))
                print(f"Processing FPS: {fps:.1f}")
                last_report_time = current_time
            
            # Optional image saving (disabled by default for Pi Zero 2W)
            if SAVE_IMAGES and detections:
                for i, det in enumerate(detections):
                    x1, y1, x2, y2, class_id, conf = det
                    if i < len(object_ids):
                        filename = f"person_{object_ids[i]}_{frame_count}.jpg"
                        img_path = os.path.join(SAVE_DIR, filename)
                        cv2.imwrite(img_path, frame[y1:y2, x1:x2])
            
            # Draw boxes (minimal processing)
            if show_live_feed:
                for det in detections:
                    x1, y1, x2, y2, class_id, conf = det
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv2.putText(frame, f"Person {conf:.2f}", (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                cv2.imshow('Pi Zero 2W Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Longer sleep for Pi Zero 2W
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\nStopping Pi Zero 2W detection system...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup
        if camera_type == 'picamera2':
            if hasattr(camera, 'close'):
                camera.close()
        else:
            if hasattr(camera, 'release'):
                camera.release()
        
        if show_live_feed:
            cv2.destroyAllWindows()
        
        print("Pi Zero 2W detection system stopped.")

if __name__ == '__main__':
    main()
