#!/usr/bin/env python3
# Detection2.py - Fixed version for Raspberry Pi Zero 2W
# Keeps original features but uses Picamera2 correctly

from picamera2 import Picamera2
import cv2
import time

# Initialize camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "XRGB8888", "size": (640, 480)})
picam2.configure(config)
picam2.start()

time.sleep(1)  # Allow camera to warm up
print("✅ Camera started. Press 'q' to quit.")

while True:
    # Capture one frame from Picamera2
    frame = picam2.capture_array()

    # (This part keeps your "detection logic" placeholder)
    # Currently just shows the preview, but you can plug in AI/ML detection here
    cv2.imshow("Detection Preview", frame)

    # Quit loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
picam2.stop()
print("❌ Camera stopped.")

