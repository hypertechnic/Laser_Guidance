#!/usr/bin/env python3

import sys
import argparse
import ctypes
import numpy as np
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

def calculate_laser_coords(center_x, center_y, camera_width=1280, camera_height=720, laser_max=4095, 
                            exp_x=0.93, exp_y=0.5):
    """Calculate laser coordinates based on detection center using exponential scaling."""
    
    # Normalize the camera coordinates to a range of 0 to 1
    norm_x = center_x / camera_width
    norm_y = center_y / camera_height
    
    # Apply exponential scaling
    scaled_x = laser_max * (np.power(norm_x, exp_x))
    scaled_y = laser_max * (np.power(norm_y, exp_y))
    
    # Clip the values to ensure they fall within the valid laser range
    x = int(np.clip(scaled_x, 0, laser_max))
    y = int(np.clip(scaled_y, 0, laser_max))
    
    return x, y

# Create a set of points that represent the corners of the detection box
def create_line_points(detection, camera_width, camera_height):
    # Corners of the detection bounding box
    top_left = (detection.Left, detection.Top)
    top_right = (detection.Right, detection.Top)
    bottom_right = (detection.Right, detection.Bottom)
    bottom_left = (detection.Left, detection.Bottom)
    
    # Convert these corners to laser coordinates
    corners = [
        calculate_laser_coords(*top_left, camera_width=camera_width, camera_height=camera_height),
        calculate_laser_coords(*top_right, camera_width=camera_width, camera_height=camera_height),
        calculate_laser_coords(*bottom_right, camera_width=camera_width, camera_height=camera_height),
        calculate_laser_coords(*bottom_left, camera_width=camera_width, camera_height=camera_height),
        calculate_laser_coords(*top_left, camera_width=camera_width, camera_height=camera_height)  # Close the loop
    ]
    
    return corners

# Send a frame to the laser DAC with multiple points to draw lines
def send_laser_frame(helios_lib, num_devices, points):
    pps = 64000
    frame = (HeliosPoint * len(points))(*[
        HeliosPoint(x, y, 50, 0, 0, 63) for x, y in points
    ])
    
    for device_index in range(num_devices):
        status_attempts = 0
        while status_attempts < 512 and helios_lib.GetStatus(device_index) != 1:
            status_attempts += 1
        helios_lib.WriteFrame(device_index, pps, len(frame), frame, 1)

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
    parser.add_argument("--width", type=int, default=800, help="camera input width")
    parser.add_argument("--height", type=int, default=600, help="camera input height")

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
        while True:
            img = input.Capture()
            if img is None:  # timeout
                continue

            # Detect objects in the image
            detections = net.Detect(img, overlay=args.overlay)

            for detection in detections:
                points = create_line_points(detection, args.width, args.height)
                send_laser_frame(helios_lib, num_devices, points)

            # Render the image
            output.Render(img)
            output.SetStatus(f"{args.network} | Network {net.GetNetworkFPS():.0f} FPS")

            # Print out performance info
            net.PrintProfilerTimes()
    finally:
        helios_lib.CloseDevices()

if __name__ == "__main__":
    main()
