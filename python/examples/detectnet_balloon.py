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
animationFrames = 1
framePointLength = 8
max_detections = 8

frames = [0 for x in range(max_detections)]
frameType = HeliosPoint * max_detections


#4095 IS MAX to scale to

#x,y,R,G,B,i

        

####################################################################################


# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=detectNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="/dev/video0", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-inception-v2", help="pre-trained model to load (see below for options)")   

#"ssd-mobilenet-v2"
#ssd-inception-v2 --fast
#monodepth-fcn-resnet50  --seems fastest
 
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use, 0.5 is def") 

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
#net = detectNet(args.network, sys.argv, args.threshold)

# note: to hard-code the paths to load a model, the following API can be used:
#
net = detectNet(model="balloon_detector.onnx", labels="balloon_labels.txt", 
                 input_blob="input_0", output_cvg="scores", output_bbox="boxes", 
                 threshold=args.threshold)

# process frames until EOS or the user exits

low = 0.7
med = 0.8
high = 0.9

while True:
    statusAttempts = 0
    # capture the next image
    img = input.Capture()

    if img is None: # timeout
        continue  

    # detect objects in the image (with overlay)
    detections = net.Detect(img, overlay=args.overlay)
    r = 0
    b = 0
    g = 0
    det_counter = 1
    for detection in detections:
        det_Center_x = detection.Center[0]
        det_Center_y = detection.Center[1]
        det_Top = detection.Top
        det_Right = detection.Right
        det_Bottom = detection.Bottom
        det_Left = detection.Left
        det_conf = detection.Confidence
        if det_conf < low:
            r = 10
            g = 0
            b = 0
        elif low < det_conf < med:
            r = 0
            g = 0
            b = 10
        elif (med < det_conf < high):
            r = 0
            g = 10
            b = 0

		#incoming is 1280x 720
		#detecting x: 0 (left)-1280
		#detecting y: 0(top) - 720
		#outgoing is 4096 x 4096
		#x:0 - 4095
		#y:0 - 4095
		#x needs gain of 3.2
		#y needs gain of 5.7

        for i in range(0,animationFrames):
            #y = round(det_Bottom)
            frames[i] = frameType()
            for j in range(max_detections):
              print("det_center_X: ", int(detection.Center[0]))
              print("det_center_Y: ", int(detection.Center[1]))
              if (1000 > detection.Center[0] > 240 ):
                x= 5.4*detection.Center[0] - 1300
                y = 2000-(6*detection.Center[1]-1900)
                print("sent_X: ", int(x))
                print("sent_Y: ", int(y))
                frames[i][j] = HeliosPoint(int(x),int(y),3,0,0,3)
                
        #write frame
            pps = 64000        
            for j in range(0,1):
                statusAttempts = 0
                # Make 512 attempts for DAC status to be ready
                while (statusAttempts < 512 and HeliosLib.GetStatus(j) != 1):
                    statusAttempts += 1
                HeliosLib.WriteFrame(j, pps, 0, ctypes.pointer(frames[i % animationFrames]), framePointLength)  
                

            
        det_counter += 1
    #if not detections:
    frames[1] = HeliosPoint(int(100),int(100),0,0,0,0)
    HeliosLib.WriteFrame(0, 3000, 0, ctypes.pointer(frames[1]), 1)
        #print("here!")

    # render the image
    output.Render(img)

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format(args.network, net.GetNetworkFPS()))

    # print out performance info
    net.PrintProfilerTimes()



##################################################################################################

 #   pps = 30000

    #attempt to only push frame when detection
  #  if detections:
   #     for j in range(numDevices):
    #        statusAttempts = 0
     #       # Make 512 attempts for DAC status to be ready
      #      while (statusAttempts < 512 and HeliosLib.GetStatus(j) != 1):
       #         statusAttempts += 1
        #    HeliosLib.WriteFrame(j, pps, 0, ctypes.pointer(frames[i % animationFrames]), framePointLength)

##################################################################################################

    # exit on input/output EOS
    #if not input.IsStreaming() or not output.IsStreaming():
        ####
     #   HeliosLib.CloseDevices()
        ####
      #  break
