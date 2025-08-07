import cv2
import os
from ultralytics import YOLO

# --- CONFIGURABLE TOGGLES ---
REPORT_TO_TERMINAL = True   # Terminal print of detections
SAVE_IMAGES = False          # Save cropped detections
SAVE_DIR = 'detections'     # Folder for cropped images
CONFIDENCE_THRESHOLD = 0.2  # Confidence threshold
SHOW_LIVE_FEED = True       # Show video output

# Only detect: person (0), cat (15), dog (16) in COCO
TARGET_CLASSES = {0: 'person', 15: 'cat', 16: 'dog'}

def main(show_live_feed=SHOW_LIVE_FEED):
    if SAVE_IMAGES and not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    model = YOLO('yolov8n.pt')  # Load Ultralytics YOLOv8n
    cap = cv2.VideoCapture(0)   # Use webcam; adjust for Pi

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_resized = cv2.resize(frame, (640, 480))

        # Run detection and tracking
        results = model.track(
            frame_resized,
            persist=True,
            conf=CONFIDENCE_THRESHOLD,
            classes=list(TARGET_CLASSES.keys())
        )

        if not results or results[0].boxes is None:
            continue

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            if class_id not in TARGET_CLASSES:
                continue

            conf = float(box.conf[0])
            if conf < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{TARGET_CLASSES[class_id]} {conf:.2f}"

            # Draw bounding box
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_resized, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Optional: print to terminal
            track_id = int(box.id[0]) if hasattr(box, "id") and box.id is not None else -1
            if REPORT_TO_TERMINAL:
                print(f"Detected {label} at [{x1},{y1},{x2},{y2}] (ID: {track_id})")

            # Save cropped image
            if SAVE_IMAGES:
                cropped = frame_resized[y1:y2, x1:x2]
                filename = f"{TARGET_CLASSES[class_id]}_ID{track_id}_F{frame_count}.jpg"
                save_path = os.path.join(SAVE_DIR, filename)
                cv2.imwrite(save_path, cropped)

        if show_live_feed:
            cv2.imshow('YOLOv8 Detection', frame_resized)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if show_live_feed:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
