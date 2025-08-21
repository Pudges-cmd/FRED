#!/bin/bash
# Modern Raspberry Pi Zero 2W Setup Script
# Updated for latest Raspbian and current software versions

set -e  # Exit on any error

echo "=== Modern Raspberry Pi Zero 2W Setup ==="
echo "This script will install all necessary dependencies for the detection system"
echo ""

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo; then
    echo "Warning: This script is designed for Raspberry Pi. Continue anyway? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update system packages
echo "1. Updating system packages..."
sudo apt update
sudo apt upgrade -y

# Install system dependencies (Raspberry Pi OS Bookworm)
echo "2. Installing system dependencies..."
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    pkg-config \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev v4l-utils \
    libgtk-3-dev \
    libatlas-base-dev \
    git wget curl unzip htop vim \
    python3-opencv \
    python3-picamera2

# Enable camera interface
echo "3. Enabling camera interface..."
sudo raspi-config nonint do_camera 0

# Enable I2C if needed for sensors
echo "4. Enabling I2C interface..."
sudo raspi-config nonint do_i2c 0

# Enable SPI if needed for displays
echo "5. Enabling SPI interface..."
sudo raspi-config nonint do_spi 0

# Create Python virtual environment
echo "6. Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "7. Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch CPU wheels (select appropriate version for Pi)
echo "8. Installing PyTorch (CPU-only) ..."
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision

# OpenCV comes from apt for stability on Pi (python3-opencv)
echo "9. Using system OpenCV (python3-opencv)"

# Install other Python dependencies
echo "10. Installing other Python dependencies..."
pip install numpy pillow psutil

# Install Ultralytics for YOLOv8 (includes v5 compatibility)
echo "11. Installing Ultralytics (YOLOv8)..."
pip install ultralytics

# PiCamera2 installed via apt above for Bookworm
echo "12. PiCamera2 provided by system package"



# Download YOLOv8n model if not present
echo "13. Downloading YOLOv8n model..."
if [ ! -f "yolov8n.pt" ]; then
    echo "Downloading YOLOv8n model..."
    wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
    echo "Model downloaded successfully!"
else
    echo "YOLOv8n model already exists."
fi

# Create detection directory
echo "14. Creating detection directory..."
mkdir -p detections

# Set up camera permissions
echo "15. Setting up camera permissions..."
sudo usermod -a -G video $USER
sudo usermod -a -G gpio $USER

# Configure camera settings
echo "16. Configuring camera settings..."
if [ -f "/boot/config.txt" ]; then
    # Add camera configuration to config.txt
    if ! grep -q "camera_auto_detect=1" /boot/config.txt; then
        echo "camera_auto_detect=1" | sudo tee -a /boot/config.txt
    fi
    
    if ! grep -q "dtoverlay=imx219" /boot/config.txt; then
        echo "dtoverlay=imx219" | sudo tee -a /boot/config.txt
    fi
fi

# Create a simple test script
echo "17. Creating camera test script..."
cat > test_camera.py << 'EOF'
#!/usr/bin/env python3
"""
Simple camera test for Raspberry Pi Zero 2W
"""

import cv2
import time
import subprocess
import sys

def test_libcamera():
    """Test libcamera"""
    print("Testing libcamera...")
    try:
        result = subprocess.run(['libcamera-still', '--help'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ libcamera-still is available")
            return True
    except:
        pass
    print("✗ libcamera-still not available")
    return False

def test_picamera2():
    """Test PiCamera2"""
    print("Testing PiCamera2...")
    try:
        from picamera2 import Picamera2
        picam2 = Picamera2()
        config = picam2.create_preview_configuration()
        picam2.configure(config)
        picam2.start()
        frame = picam2.capture_array()
        picam2.close()
        print("✓ PiCamera2 is working")
        return True
    except Exception as e:
        print(f"✗ PiCamera2 failed: {e}")
        return False

def test_opencv():
    """Test OpenCV camera"""
    print("Testing OpenCV camera...")
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print("✓ OpenCV camera is working")
                return True
            else:
                print("✗ OpenCV camera opened but can't read frames")
        else:
            print("✗ OpenCV camera not opened")
    except Exception as e:
        print(f"✗ OpenCV camera failed: {e}")
    return False

def main():
    print("=== Camera Test for Raspberry Pi Zero 2W ===")
    
    # Test different camera methods
    libcamera_ok = test_libcamera()
    picamera2_ok = test_picamera2()
    opencv_ok = test_opencv()
    
    print("\n=== Test Results ===")
    print(f"libcamera: {'✓' if libcamera_ok else '✗'}")
    print(f"PiCamera2: {'✓' if picamera2_ok else '✗'}")
    print(f"OpenCV: {'✓' if opencv_ok else '✗'}")
    
    if libcamera_ok or picamera2_ok or opencv_ok:
        print("\n✓ At least one camera method is working!")
        return 0
    else:
        print("\n✗ No camera methods are working!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

chmod +x test_camera.py

# Create activation script
echo "18. Creating activation script..."
cat > activate.sh << 'EOF'
#!/bin/bash
# Activate the Python virtual environment
source venv/bin/activate
echo "Virtual environment activated!"
echo "Run 'python Detection.py' to start the detection system"
echo "Run 'python test_camera.py' to test the camera"
EOF

chmod +x activate.sh

# Final instructions
echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Next steps:"
echo "1. Reboot your Raspberry Pi: sudo reboot"
echo "2. Activate the virtual environment: source activate.sh"
echo "3. Test the camera: python test_camera.py"
echo "4. Run the detection system: python Detection.py"
echo ""
echo "Troubleshooting:"
echo "- If camera doesn't work, check: sudo vcgencmd get_camera"
echo "- Enable camera in raspi-config if needed"
echo "- Check camera connections and ribbon cable"
echo "- Try different camera methods in Detection.py"
echo ""
echo "The system is now ready for modern Raspberry Pi Zero 2W operation!" 