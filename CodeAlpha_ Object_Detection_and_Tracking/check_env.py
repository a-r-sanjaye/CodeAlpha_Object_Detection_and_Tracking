import torch
import cv2
import torchvision
import numpy as np

print("Checking environment...")
print(f"Torch version: {torch.__version__}")
print(f"OpenCV version: {cv2.__version__}")
try:
    import filterpy
    print(f"FilterPy installed.")
except ImportError:
    print("FilterPy NOT installed.")

print("Environment check passed.")
