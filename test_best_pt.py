from ultralytics import YOLO
import sys

try:
    model = YOLO('best.pt')
    print("Model loaded successfully!")
    print("Model classes (names):", model.names)
    print("Model task:", model.task)
    sys.exit(0)
except Exception as e:
    print("Error loading model:", e)
    sys.exit(1)
