from SMS import get_gps_location

if __name__ == '__main__':
    gps = get_gps_location()
    if gps and gps['fix']:
        print(f"GPS Fix: {gps['lat']:.6f}, {gps['lon']:.6f}, Altitude: {gps['alt']}m")
    else:
        print('No GPS fix or GPS data unavailable.') 