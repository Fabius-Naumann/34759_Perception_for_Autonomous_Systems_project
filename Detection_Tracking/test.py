from ultralytics import YOLO
import cv2

# Load a pre-trained YOLOv8 model
# 'yolov8n.pt' is the nano model, which is fast and accurate enough for many tasks
model = YOLO('yolov8n.pt')

# Path to your input video file
INPUT_VIDEO_PATH = "./Detection_Tracking/Sequence 1.mp4"

# Run object tracking on the video
# The 'tracker' argument automatically enables Multi-Object Tracking (MOT)
# 'bytetrack.yaml' is a good, modern tracking algorithm
results = model.track(source=INPUT_VIDEO_PATH, 
                      tracker="bytetrack.yaml", 
                      show=True, 
                      conf=0.3, # Confidence threshold (adjust this!)
                      classes=[0]) # Class 0 in COCO is 'person'

# The 'show=True' will display the video with bounding boxes and tracking IDs.
# If you want to save the output video, use the 'save=True' argument:
# results = model.track(source=INPUT_VIDEO_PATH, tracker="bytetrack.yaml", save=True)