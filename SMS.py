import serial
import time
import glob

# Serial port for SIM7600X G-H (adjust if needed)
SIM7600_PORT = '/dev/ttyUSB2'  # Default; will auto-detect if not present
BAUDRATE = 115200

# Serial port for GPS (SIM7600X G-H shares port, but GPS NMEA often on /dev/ttyUSB1)
GPS_PORT = '/dev/ttyUSB1'
# --- SMS SENDING ---
def _autodetect_port(preferred_ports):
    """Find first existing serial device from candidate list or glob.
    """
    for candidate in preferred_ports:
        if '*' in candidate:
            for dev in glob.glob(candidate):
                return dev
        else:
            try:
                with open(candidate, 'rb'):
                    return candidate
            except Exception:
                continue
    return None
GPS_BAUDRATE = 115200

# Timeout for serial operations
SERIAL_TIMEOUT = 2

# --- SMS SENDING ---
def send_sms(phone_number, message):
    """
    Send an SMS using the SIM7600X G-H HAT.
    Args:
        phone_number (str): Recipient phone number (international format, e.g. '+1234567890')
        message (str): Message to send
    Returns:
        bool: True if sent, False otherwise
    """
    try:
        port = SIM7600_PORT
        if not port:
            port = _autodetect_port(['/dev/ttyUSB2', '/dev/ttyUSB3', '/dev/ttyUSB1', '/dev/ttyS0', '/dev/serial0', '/dev/ttyAMA0'])
        else:
            # If default is missing, attempt auto
            try:
                with open(port, 'rb'):
                    pass
            except Exception:
                port = _autodetect_port(['/dev/ttyUSB2', '/dev/ttyUSB3', '/dev/ttyUSB1', '/dev/ttyS0', '/dev/serial0', '/dev/ttyAMA0'])

        if not port:
            raise serial.SerialException('No SIM7600 serial port found')

        ser = serial.Serial(port, BAUDRATE, timeout=SERIAL_TIMEOUT)
        time.sleep(0.5)
        ser.write(b'AT+CMGF=1\r')  # Set SMS text mode
        time.sleep(0.5)
        ser.write(('AT+CMGS="{}"\r'.format(phone_number)).encode())
        time.sleep(0.5)
        ser.write((message + '\x1A').encode())  # End with Ctrl+Z
        time.sleep(3)
        ser.close()
        return True
    except serial.SerialException as e:
        print(f"[SMS] Serial port error: {e}")
        return False
    except Exception as e:
        print(f"[SMS] General error: {e}")
        return False

# --- GPS GRABBING ---
def get_gps_location():
    """
    Get GPS location from SIM7600X G-H HAT.
    Returns:
        dict: {'lat': float, 'lon': float, 'alt': float, 'fix': bool} or None if not available
    """
    try:
        port = GPS_PORT
        if not port:
            port = _autodetect_port(['/dev/ttyUSB1', '/dev/ttyUSB2', '/dev/ttyUSB3', '/dev/ttyS0', '/dev/serial0', '/dev/ttyAMA0'])
        else:
            try:
                with open(port, 'rb'):
                    pass
            except Exception:
                port = _autodetect_port(['/dev/ttyUSB1', '/dev/ttyUSB2', '/dev/ttyUSB3', '/dev/ttyS0', '/dev/serial0', '/dev/ttyAMA0'])

        if not port:
            raise serial.SerialException('No GPS serial port found')

        ser = serial.Serial(port, GPS_BAUDRATE, timeout=SERIAL_TIMEOUT)
        time.sleep(0.5)
        ser.write(b'AT+CGPS=1,1\r')  # Turn on GPS
        time.sleep(2)
        ser.write(b'AT+CGPSINFO\r')
        time.sleep(1)
        nmea = ser.readlines()
        ser.close()
        for line in nmea:
            if b'+CGPSINFO:' in line:
                parts = line.decode().split(':')[1].strip().split(',')
                if len(parts) >= 6 and parts[0] and parts[2]:
                    lat = nmea_to_decimal(parts[0], parts[1])
                    lon = nmea_to_decimal(parts[2], parts[3])
                    alt = float(parts[4]) if parts[4] else 0.0
                    fix = True
                    return {'lat': lat, 'lon': lon, 'alt': alt, 'fix': fix}
        return {'lat': 0.0, 'lon': 0.0, 'alt': 0.0, 'fix': False}
    except serial.SerialException as e:
        print(f"[GPS] Serial port error: {e}")
        return {'lat': 0.0, 'lon': 0.0, 'alt': 0.0, 'fix': False}
    except Exception as e:
        print(f"[GPS] General error: {e}")
        return {'lat': 0.0, 'lon': 0.0, 'alt': 0.0, 'fix': False}

def nmea_to_decimal(coord, direction):
    """
    Convert NMEA coordinate to decimal degrees.
    """
    if not coord or not direction:
        return 0.0
    try:
        deg = float(coord[:2])
        min = float(coord[2:])
        decimal = deg + min / 60
        if direction in ['S', 'W']:
            decimal = -decimal
        return decimal
    except Exception:
        return 0.0

# --- HIGH-LEVEL REPORT FUNCTION ---
def report_detection_via_sms(phone_number, detected_counts):
    """
    Send an SMS with detection info and GPS location.
    Args:
        phone_number (str): Recipient phone number
        detected_counts (dict): {'person': int, 'dog': int, 'cat': int}
    Returns:
        bool: True if sent, False otherwise
    """
    gps = get_gps_location()
    msg = (
        f"Detection Alert!\n"
        f"People: {detected_counts.get('person', 0)}\n"
        f"Dogs: {detected_counts.get('dog', 0)}\n"
        f"Cats: {detected_counts.get('cat', 0)}\n"
    )
    if gps and gps['fix']:
        msg += f"Location: {gps['lat']:.6f},{gps['lon']:.6f} Alt:{gps['alt']}m"
    else:
        msg += "Location: Not available"
    return send_sms(phone_number, msg) 