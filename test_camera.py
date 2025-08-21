#!/usr/bin/env python3
"""
Comprehensive Camera Test for Raspberry Pi Zero 2W
Tests all available camera methods: libcamera, PiCamera2, and OpenCV
"""

import cv2
import time
import subprocess
import sys
import os



def test_picamera2():
    """Test PiCamera2 library"""
    print("Testing PiCamera2...")
    try:
        from picamera2 import Picamera2
        
        picam2 = Picamera2()
        
        # Create a simple configuration
        config = picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"},
            controls={"FrameDurationLimits": (100000, 100000)}  # 10 FPS
        )
        
        picam2.configure(config)
        picam2.start()
        
        # Wait a moment for camera to start
        time.sleep(1)
        
        # Try to capture a frame
        frame = picam2.capture_array()
        
        if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
            print(f"✓ PiCamera2 is working - captured frame: {frame.shape}")
            picam2.close()
            return True
        else:
            print("✗ PiCamera2 captured empty frame")
            picam2.close()
            
    except ImportError:
        print("✗ PiCamera2 library not installed")
    except Exception as e:
        print(f"✗ PiCamera2 failed: {e}")
    return False

def test_opencv():
    """Test OpenCV camera access"""
    print("Testing OpenCV camera...")
    
    # Try different camera backends
    backends = [
        (cv2.CAP_V4L2, "V4L2"),
        (cv2.CAP_ANY, "ANY"),
        (cv2.CAP_GSTREAMER, "GStreamer")
    ]
    
    for backend, name in backends:
        try:
            print(f"  Trying {name} backend...")
            cap = cv2.VideoCapture(0, backend)
            
            if not cap.isOpened():
                print(f"    ✗ {name} backend: Camera not opened")
                continue
                
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 10)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Try to read a frame
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"    ✓ {name} backend: Working - frame shape: {frame.shape}")
                cap.release()
                return True
            else:
                print(f"    ✗ {name} backend: Camera opened but can't read frames")
                cap.release()
                
        except Exception as e:
            print(f"    ✗ {name} backend failed: {e}")
            continue
    
    print("  ✗ All OpenCV backends failed")
    return False

def test_camera_devices():
    """Check what camera devices are available"""
    print("Checking camera devices...")
    
    # Check v4l2 devices
    try:
        result = subprocess.run(['v4l2-ctl', '--list-devices'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("V4L2 devices found:")
            for line in result.stdout.split('\n'):
                if 'video' in line.lower():
                    print(f"  {line.strip()}")
        else:
            print("No V4L2 devices found")
    except:
        print("v4l2-ctl not available")
    
    # Check /dev/video* devices
    video_devices = []
    for i in range(10):
        if os.path.exists(f'/dev/video{i}'):
            video_devices.append(f'/dev/video{i}')
    
    if video_devices:
        print(f"Video devices found: {', '.join(video_devices)}")
    else:
        print("No video devices found in /dev/")

def test_system_info():
    """Display system information"""
    print("System Information:")
    
    # Check if running on Raspberry Pi
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            if 'Raspberry Pi' in cpuinfo:
                print("✓ Running on Raspberry Pi")
                # Extract model info
                for line in cpuinfo.split('\n'):
                    if 'Model' in line:
                        print(f"  {line.strip()}")
                        break
            else:
                print("⚠ Not running on Raspberry Pi")
    except:
        print("⚠ Could not read CPU info")
    
    # Check camera status
    try:
        result = subprocess.run(['vcgencmd', 'get_camera'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"Camera status: {result.stdout.strip()}")
        else:
            print("Could not get camera status")
    except:
        print("vcgencmd not available")

def main():
    print("=== Comprehensive Camera Test for Raspberry Pi Zero 2W ===")
    print("")
    
    # Display system information
    test_system_info()
    print("")
    
    # Check camera devices
    test_camera_devices()
    print("")
    
    # Test different camera methods
    picamera2_ok = test_picamera2()
    print("")
    
    opencv_ok = test_opencv()
    print("")
    
    # Summary
    print("=== Test Results Summary ===")
    print(f"PiCamera2: {'✓' if picamera2_ok else '✗'}")
    print(f"OpenCV: {'✓' if opencv_ok else '✗'}")
    print("")
    
    if picamera2_ok or opencv_ok:
        print("✓ At least one camera method is working!")
        print("")
        print("Recommendations:")
        if picamera2_ok:
            print("- Use PiCamera2 for best performance on modern Pi")
        if opencv_ok:
            print("- Use OpenCV as fallback option")
        return 0
    else:
        print("✗ No camera methods are working!")
        print("")
        print("Troubleshooting steps:")
        print("1. Check camera connections and ribbon cable")
        print("2. Enable camera in raspi-config: sudo raspi-config")
        print("3. Reboot: sudo reboot")
        print("4. Check camera status: sudo vcgencmd get_camera")
        print("5. Install missing packages: sudo apt install picamera2")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 