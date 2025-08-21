print(">>> Running UPDATED Detection2.py with Picamera2")

from picamera2 import Picamera2
import cv2
import time

# Initialize camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "XRGB8888", "size": (640, 480)})
picam2.configure(config)
picam2.start()

time.sleep(2)  # Let camera warm up

# Load a sample Haar cascade (you can replace with your model)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print(">>> Starting detection loop. Press CTRL+C to exit.")

try:
    while True:
        frame = picam2.capture_array()

        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Run detection
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Print how many faces (or objects) were detected
        if len(faces) > 0:
            print(f"Detected {len(faces)} face(s)")

        # OPTIONAL: show preview window if you’re running with a display
        # Comment out if headless
        cv2.imshow("Detection2 Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n>>> Detection stopped by user.")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
