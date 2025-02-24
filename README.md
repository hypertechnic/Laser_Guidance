# Laser Tracking System Project

A Python-based system that uses a HeliosDAC digital-to-analog interface, a Lasercube scanning laser, and a USB camera (via PyTorch) to track and highlight 2D objects in real-time. Designed to run on an Nvidia Jetson but adaptable to other devices, the system can point a laser at detected objects, change laser colors based on detection confidence or object type, and handle multiple object tracking.

![download](https://github.com/user-attachments/assets/999108de-f563-44cb-929b-b12796b73a9a)

Demonstration Video:
https://youtu.be/aSx_q8-XqPE

# Features

Real-Time Object Tracking: Detect and follow multiple 2D objects.

Dynamic Laser Control: Adjust laser color based on detection confidence or object type.

Flexible Hardware Compatibility: Optimized for Nvidia Jetson but usable with other devices.

Configurable Detection Settings: Easily adjust detection parameters and laser behaviors.

# Hardware Requirements

HeliosDAC: Digital-to-analog interface for laser control. (https://bitlasers.com/helios-laser-dac/)

Lasercube: Scanning laser for object highlighting.

USB Camera: For capturing live video feed.

Optional: Nvidia Jetson for optimized performance.

# Software Requirements

Python 3.x

PyTorch (for object detection)

HeliosDAC Python library (https://github.com/Grix/helios_dac)

# Installation

# Clone the repository
git clone <repository-url>
cd <repository-folder>


# Usage

cd examples
python detectnet_centerLaser.py --camera_id 0 --laser_port /dev/ttyUSB0 

Example Configuration

laser:
  color_mode: confidence  # Options: confidence, object_type
  default_color: [255, 0, 0]  # RGB for red

detection:
  confidence_threshold: 0.5
  max_objects: 5

# Acknowledgments

Helios_dac Team: For their open-source Python library.

PyTorch: For the object detection backbone.

License

MIT License

Contributing

Contributions are welcome! Please open an issue or submit a pull request.

