import os
import cv2
import numpy as np
from detection import (
    detect_moving_objects,
    draw_movement_mask,
    draw_optical_flow,
    draw_detections
)
from tracking import TrackManager

# ---------------- Video Setup ----------------
video_path = "./Detection_Tracking/inputs/seq2_image_02_video.mp4"  # Run from project root
cap = cv2.VideoCapture(video_path)
ret, frame1 = cap.read()
if not ret:
    raise RuntimeError(f"Cannot read first frame from {video_path}")

gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
h, w = frame1.shape[:2]

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0 or np.isnan(fps):
    # Fallback FPS if OpenCV fails to read it
    fps = 30.0
    print("Warning: FPS from video was invalid. Using fallback fps = 30.0")

# Make sure output directory exists
out_dir = "./Detection_Tracking/out"
os.makedirs(out_dir, exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"XVID")

out_boxes_path = os.path.join(out_dir, "people_boxes.avi")
out_flow_path  = os.path.join(out_dir, "people_flow.avi")
out_det_path   = os.path.join(out_dir, "people_detection.avi")

out_boxes = cv2.VideoWriter(out_boxes_path, fourcc, fps, (w, h))
out_flow  = cv2.VideoWriter(out_flow_path,  fourcc, fps, (w, h))
out_det   = cv2.VideoWriter(out_det_path,   fourcc, fps, (w, h))

if not out_boxes.isOpened():
    raise RuntimeError(f"Failed to open VideoWriter for {out_boxes_path}")
if not out_flow.isOpened():
    raise RuntimeError(f"Failed to open VideoWriter for {out_flow_path}")
if not out_det.isOpened():
    raise RuntimeError(f"Failed to open VideoWriter for {out_det_path}")


# ---------------- Main Loop ----------------
frame_count = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 10 == 0:
        print(f"Processed {frame_count} / {total_frames} frames")

    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Optical flow (Farneback)
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        0.5, 3, 20, 5, 5, 1.2, 0
    )

    # -------- Detection --------
    detections, movement_mask = detect_moving_objects(flow, frame1, frame2)

   # -------- Detection: visual mask --------
    det_vis = draw_movement_mask(frame2, movement_mask)
    out_det.write(det_vis)

    # -------- Optical flow visualization --------  
    flow_vis = draw_optical_flow(frame2, flow)
    out_flow.write(flow_vis)

    # -------- Bounding boxes --------
    frame_boxes = draw_detections(frame2, detections)
    out_boxes.write(frame_boxes)


    # Prepare next frame
    gray1 = gray2.copy()
    frame1 = frame2.copy()

cap.release()
out_boxes.release()
out_flow.release()
out_det.release()

print("Saved people_boxes.avi, people_flow.avi and people_detection.avi")
