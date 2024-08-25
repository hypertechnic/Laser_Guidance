#gpt

#!/usr/bin/env python3

import sys
import argparse
import ctypes
from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, Log

# Define point structure
class HeliosPoint(ctypes.Structure):
    _fields_ = [
        ('x', ctypes.c_uint16),
        ('y', ctypes.c_uint16),
        ('r', ctypes.c_uint8),
        ('g', ctypes.c_uint8),
        ('b', ctypes.c_uint8),
        ('i', ctypes.c_uint8)
    ]

# Load and initialize Helios DAC library
def init_helios_dac():
    helios_lib = ctypes.cdll.LoadLibrary("./libHeliosDacAPI.so")
    num_devices = helios_lib.OpenDevices()
    if num_devices < 1:
        raise Exception("No Helios DAC devices found")
    print(f"Found {num_devices} Helios DAC(s)")
    return helios_lib, num_devices

## notes on scaling between camera and laser
# (x, y)

## camera TOP LEFT = (0, 0)
## camera TOP RIGHT = (1280, 0)
## camera BOTTOM LEFT = (0, 720)
## camera BOTTOM RIGHT = (1280, 720)

## laser TOP LEFT = (0, 4095)
## laser TOP RIGHT = (4095, 4095)
## laser BOTTOM LEFT = (0, 0)
## laser BOTTOM RIGHT = (4095, 0)

## x just needs to scale, factors are 1280/4095
## y needs to flip and scale. Factors are 720/4095

# Function to calculate laser coordinates
#def calculate_laser_coords(center_x, center_y):
#    x = int(1.6 * (center_x * 3.415) - 1600)
#    y = int(4096 - (center_y * 5.68))
#    return x, y


## in practice, I find x needs to be scaled more
x_boost = 1.8

def calculate_laser_coords(center_x, center_y, camera_width=1280, camera_height=720, laser_max=4095):
    """Calculate laser coordinates based on detection center with proper scaling from camera coordinates."""
    # Scale the camera coordinates (0-1280 for x, 0-720 for y) to the laser coordinates (0-4095 for both x and y)
    x = int((center_x / (camera_width * x_boost)) * laser_max)
    y = int((center_y / camera_height) * laser_max)
    
    # Optionally, adjust the y-coordinate if the coordinate systems differ in origin (e.g., invert y-axis)
    # Uncomment the following line if the laser y-coordinate needs to be inverted
    y = laser_max - y

    return x, y



# Send a frame to the laser DAC
def send_laser_frame(helios_lib, num_devices, x, y):
    frame = HeliosPoint(x, y, 50, 0, 0, 63)
    pps = 64000
    for device_index in range(num_devices):
        status_attempts = 0
        while status_attempts < 512 and helios_lib.GetStatus(device_index) != 1:
            status_attempts += 1
        helios_lib.WriteFrame(device_index, pps, 64, ctypes.pointer(frame), 1)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Locate objects in a live camera stream using an object detection DNN.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=detectNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage()
    )

    parser.add_argument("input", type=str, default="/dev/video0", nargs='?', help="URI of the input stream")
    parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
    parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load")
    parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags")
    parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use")

    args = parser.parse_args()

    # Initialize video sources and outputs
    input = videoSource(args.input, argv=sys.argv)
    output = videoOutput(args.output, argv=sys.argv)
    
    # Load object detection network
    net = detectNet(args.network, sys.argv, args.threshold)

    # Initialize Helios DAC
    helios_lib, num_devices = init_helios_dac()

    try:
        # Process frames until EOS or the user exits
        #while input.IsStreaming() and output.IsStreaming():
        while True:
            img = input.Capture()
            if img is None:  # timeout
                print("image timed out!!!!")
                continue

            # Detect objects in the image
            detections = net.Detect(img, overlay=args.overlay)

            for detection in detections:
                det_center_x = detection.Center[0]
                det_center_y = detection.Center[1]
                
                x, y = calculate_laser_coords(det_center_x, det_center_y)
                print(f"det_Center_x: {det_center_x}, sent x: {x}")
                print(f"det_Center_y: {det_center_y}, sent y: {y}")

                send_laser_frame(helios_lib, num_devices, x, y)

            # Render the image
            output.Render(img)
            output.SetStatus(f"{args.network} | Network {net.GetNetworkFPS():.0f} FPS")

            # Print out performance info
            net.PrintProfilerTimes()
    finally:
        helios_lib.CloseDevices()

if __name__ == "__main__":
    main()
