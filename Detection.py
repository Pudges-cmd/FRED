# Detection.py cum
# Lightweight YOLOv5n object detection and tracking for Pi Zero 2 W
# Future integration points: SMS messaging, website data sending (see main() for hooks)
import cv2
import torch
import time
import os
from collections import deque
import numpy as np

# --- CONFIGURABLE TOGGLES ---
REPORT_TO_TERMINAL = True  # Set to False to disable terminal reporting
SAVE_IMAGES = True         # Set to False to disable saving detected images
SAVE_DIR = 'detections'    # Directory to save images
CONFIDENCE_THRESHOLD = 0.2  # Minimum confidence for detection
TRACKER_BUFFER = 30        # Number of frames to keep track of objects
SHOW_LIVE_FEED = True     # Set to True to show live video feed (for testing)

# --- LOAD YOLOv5n MODEL ---
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5n.pt', force_reload=False)
model.conf = CONFIDENCE_THRESHOLD

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

def main(show_live_feed=SHOW_LIVE_FEED):
    if SAVE_IMAGES and not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Use webcam for testing; for Pi, change to cv2.VideoCapture(0, cv2.CAP_V4L2) or PiCamera
    cap = cv2.VideoCapture(0)
    tracker = CentroidTracker()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        results = model(frame)
        detections = []
        for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
            class_id = int(cls)
            if class_id in TARGET_CLASSES and conf >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, xyxy)
                detections.append((x1, y1, x2, y2, class_id, conf))

        object_ids = tracker.update(detections)

        for i, det in enumerate(detections):
            x1, y1, x2, y2, class_id, conf = det
            label = f"{TARGET_CLASSES[class_id]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            if REPORT_TO_TERMINAL:
                print(f"Detected {label} at [{x1},{y1},{x2},{y2}] (ID: {object_ids[i]})")
            if SAVE_IMAGES:
                img_path = os.path.join(SAVE_DIR, f"{TARGET_CLASSES[class_id]}_{object_ids[i]}_{frame_count}.jpg")
                cv2.imwrite(img_path, frame[y1:y2, x1:x2])

        if show_live_feed:
            cv2.imshow('Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if show_live_feed:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
