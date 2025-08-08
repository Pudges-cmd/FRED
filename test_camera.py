#!/usr/bin/env python3
"""
Simple camera test script for Raspberry Pi Zero W
Run this first to verify your camera is working before running the main detection script
"""

import cv2
import time
import sys

def test_camera():
    print("Testing camera on Raspberry Pi Zero W...")
    
    # Try Pi Camera first
    print("Attempting to initialize Pi Camera...")
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 10)
        
        if not cap.isOpened():
            raise Exception("Pi Camera not available")
        
        print("✓ Pi Camera initialized successfully!")
        
        # Test reading a few frames
        for i in range(5):
            ret, frame = cap.read()
            if ret:
                print(f"✓ Frame {i+1} captured successfully - Size: {frame.shape}")
            else:
                print(f"✗ Failed to capture frame {i+1}")
            time.sleep(0.5)
        
        cap.release()
        print("✓ Pi Camera test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Pi Camera failed: {e}")
        print("Trying USB webcam...")
        
        try:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 10)
            
            if not cap.isOpened():
                raise Exception("USB webcam not available")
            
            print("✓ USB webcam initialized successfully!")
            
            # Test reading a few frames
            for i in range(5):
                ret, frame = cap.read()
                if ret:
                    print(f"✓ Frame {i+1} captured successfully - Size: {frame.shape}")
                else:
                    print(f"✗ Failed to capture frame {i+1}")
                time.sleep(0.5)
            
            cap.release()
            print("✓ USB webcam test completed successfully!")
            return True
            
        except Exception as e2:
            print(f"✗ USB webcam failed: {e2}")
            print("✗ No camera available!")
            return False

if __name__ == "__main__":
    success = test_camera()
    if success:
        print("\nCamera test PASSED! You can now run Detection.py")
    else:
        print("\nCamera test FAILED! Please check your camera setup.")
        print("For Pi Camera: sudo raspi-config -> Interface Options -> Camera -> Enable")
        print("For USB webcam: Make sure it's properly connected and recognized")
    sys.exit(0 if success else 1) 
