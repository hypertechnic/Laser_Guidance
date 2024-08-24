#!/usr/bin/env python3

import sys
import argparse

####################################################################################
# -*- coding: utf-8 -*-
import ctypes
####################################################################################

from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, Log

####################################################################################
#Define point structure
class HeliosPoint(ctypes.Structure):
    #_pack_=1
    _fields_ = [('x', ctypes.c_uint16),
                ('y', ctypes.c_uint16),
                ('r', ctypes.c_uint8),
                ('g', ctypes.c_uint8),
                ('b', ctypes.c_uint8),
                ('i', ctypes.c_uint8)]

#Load and initialize library
HeliosLib = ctypes.cdll.LoadLibrary("./libHeliosDacAPI.so")
numDevices = HeliosLib.OpenDevices()
print("Found ", numDevices, "Helios DACs")

#Create sample frames
frame = HeliosPoint
#4095 IS MAX to scale to
x = 2048
y = 2048

#x,y,R,G,B,i

        

####################################################################################


# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=detectNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="/dev/video0", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# create video sources and outputs
input = videoSource(args.input, argv=sys.argv)
output = videoOutput(args.output, argv=sys.argv+is_headless)
	
# load the object detection network
net = detectNet(args.network, sys.argv, args.threshold)

# note: to hard-code the paths to load a model, the following API can be used:
#
# net = detectNet(model="model/ssd-mobilenet.onnx", labels="model/labels.txt", 
#                 input_blob="input_0", output_cvg="scores", output_bbox="boxes", 
#                 threshold=args.threshold)

# process frames until EOS or the user exits
while True:
    # capture the next image
    img = input.Capture()

    if img is None: # timeout
        continue  
        
    # detect objects in the image (with overlay)
    detections = net.Detect(img, overlay=args.overlay)

    # print the detections
    #print("detected {:d} objects in image".format(len(detections)))
#real x = 3.415
#real y = 5.68
    for detection in detections:
        det_Center_x = detection.Center[0]
        det_Center_y = detection.Center[1]
        det_Top = detection.Top
        det_Right = detection.Right
        det_Bottom = detection.Bottom
        det_Left = detection.Left
        x=1.6*(detection.Center[0]*3.415)-1600
        y=(4096-(detection.Center[1]*5.68))
        

        print("det_Center_x: ", det_Center_x)
        print("sent x: ", int(x))
        print("det_Center_y: ", det_Center_y)
        print("sent y: ", int(y))
        
        print("det_Top: ", det_Top)
        print("det_Right: ", det_Right)
        print("det_Bottom: ", det_Bottom)
        print("det_Left: ", det_Left)
        
        frame = HeliosPoint(int(x),int(y),50,0,0,63)

    # render the image
    output.Render(img)

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format(args.network, net.GetNetworkFPS()))

    # print out performance info
    net.PrintProfilerTimes()



##################################################################################################

    pps = 64000

    #attempt to only push frame when detection
    if detections:
        for j in range(numDevices):
            statusAttempts = 0
            # Make 512 attempts for DAC status to be ready. After that, just give up and try to write the frame anyway
            while (statusAttempts < 512 and HeliosLib.GetStatus(j) != 1):
                statusAttempts += 1
            HeliosLib.WriteFrame(j, pps, 64, ctypes.pointer(frame), 1) #Send the frame object of i frames

##################################################################################################

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        ####
        HeliosLib.CloseDevices()
        ####
        break
