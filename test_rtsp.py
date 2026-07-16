import cv2
import sys
import os

url = "rtsp://100.86.84.22:8554/live.sdp"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
print("Trying FFMPEG UDP...")
cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print("SUCCESS with UDP! Resolution:", frame.shape)
        cap.release()
        sys.exit(0)
cap.release()

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
print("Trying FFMPEG TCP...")
cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print("SUCCESS with TCP! Resolution:", frame.shape)
        cap.release()
        sys.exit(0)
cap.release()

print("FAILED BOTH.")
sys.exit(1)
