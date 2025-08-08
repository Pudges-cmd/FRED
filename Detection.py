# Detection.py
# Lightweight YOLOv5n object detection and tracking for Pi Zero 2 W
# Optimized for Raspberry Pi Zero W with Raspbian Lite
import cv2
import torch
import time
import os
import sys
from collections import deque
import numpy as np

# --- CONFIGURABLE TOGGLES ---
REPORT_TO_TERMINAL = True  # Set to False to disable terminal reporting
SAVE_IMAGES = True         # Set to False to disable saving detected images
SAVE_DIR = 'detections'    # Directory to save images
CONFIDENCE_THRESHOLD = 0.3  # Increased threshold for better performance
TRACKER_BUFFER = 30        # Number of frames to keep track of objects
SHOW_LIVE_FEED = False     # Set to False for headless operation (Raspbian Lite)
USE_PICAMERA = False        # Set to True for Pi Camera, False for USB webcam

# --- LOAD YOLOv5n MODEL ---
print("Loading YOLOv5 model...")
try:
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov8n.pt', force_reload=False)
    model.conf = CONFIDENCE_THRESHOLD
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Trying alternative model path...")
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True, force_reload=False)
        model.conf = CONFIDENCE_THRESHOLD
        print("Alternative model loaded successfully!")
    except Exception as e2:
        print(f"Failed to load model: {e2}")
        sys.exit(1)

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

def setup_camera():
    """Setup camera for Raspberry Pi"""
    if USE_PICAMERA:
        # Try Pi Camera first
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 10)  # Lower FPS for Pi Zero W
            if not cap.isOpened():
                raise Exception("Pi Camera not available")
            print("Pi Camera initialized successfully")
            return cap
        except Exception as e:
            print(f"Pi Camera failed: {e}")
            print("Trying USB webcam...")
    
    # Fallback to USB webcam
    try:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 10)
        if not cap.isOpened():
            raise Exception("USB webcam not available")
        print("USB webcam initialized successfully")
        return cap
    except Exception as e:
        print(f"Failed to initialize camera: {e}")
        return None

def main(show_live_feed=SHOW_LIVE_FEED):
    print("Starting detection system...")
    
    if SAVE_IMAGES and not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"Created directory: {SAVE_DIR}")

    # Setup camera
    cap = setup_camera()
    if cap is None:
        print("ERROR: No camera available!")
        return

    tracker = CentroidTracker()
    frame_count = 0
    last_report_time = time.time()
    report_interval = 1.0  # Report every second to avoid spam

    print("Detection system ready! Press Ctrl+C to stop.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera")
                time.sleep(0.1)
                continue
                
            frame_count += 1
            
            # Process frame with YOLO
            results = model(frame)
            detections = []
            
            for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
                class_id = int(cls)
                if class_id in TARGET_CLASSES and conf >= CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, xyxy)
                    detections.append((x1, y1, x2, y2, class_id, conf))

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
        cap.release()
        if show_live_feed:
            cv2.destroyAllWindows()
        print("Detection system stopped.")

if __name__ == '__main__':
    main()
