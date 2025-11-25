import os
import cv2
import numpy as np
from detection import (
    detect_moving_objects,
    draw_movement_mask,
    draw_optical_flow,
    draw_detections
)
from tracking import TrackManager, draw_tracks

# ---------------- Video Setup ----------------
SEQ = "seq2"
video_path = f"./Detection_Tracking/inputs/{SEQ}_image_03_video.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame1 = cap.read()
if not ret:
    raise RuntimeError(f"Cannot read first frame from {video_path}")

gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
h, w = frame1.shape[:2]

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0 or np.isnan(fps):
    fps = 30.0
    print("Warning: FPS invalid → using fallback 30 FPS")

# Output dirs
out_dir = f"./Detection_Tracking/out/{SEQ}/"
os.makedirs(out_dir, exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"XVID")

out_boxes   = cv2.VideoWriter(os.path.join(out_dir, "people_boxes.avi"),     fourcc, fps, (w, h))
out_flow    = cv2.VideoWriter(os.path.join(out_dir, "people_flow.avi"),      fourcc, fps, (w, h))
out_det     = cv2.VideoWriter(os.path.join(out_dir, "people_detection.avi"), fourcc, fps, (w, h))
out_tracks  = cv2.VideoWriter(os.path.join(out_dir, "people_tracks.avi"),    fourcc, fps, (w, h))
# ---------------- Tracking Setup ----------------
tracker = TrackManager(
    max_invisible=60,
    dist_thresh=120.0,
    min_confirm_frames=3
)

dt = 1.0 / fps
F = np.eye(6, dtype=np.float32)
F[0, 2] = dt
F[1, 3] = dt
u = np.zeros((6, 1), dtype=np.float32)
Q = np.eye(6, dtype=np.float32) * 2.0

H = np.zeros((2, 6), dtype=np.float32)
H[0, 0] = 1.0
H[1, 1] = 1.0
R = np.eye(2, dtype=np.float32) * 5.0

# ---------------- Main Loop ----------------
frame_count = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 10 == 0:
        print(f"Processed {frame_count}/{total_frames}")

    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # -------- Optical flow --------
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        0.5, 3, 20, 5, 5, 1.2, 0
    )

    # -------- Detection --------
    detections, movement_mask = detect_moving_objects(flow, frame1, frame2)

    # -------- Track prediction --------
    tracker.predict_tracks(F, u, Q)

    # -------- Association --------
    assigned_tracks, assigned_dets = tracker.associate(detections, H, R)
    assigned_pending = tracker.associate_pending(detections, H, R, assigned_dets)

    # -------- Create missing tracks --------
    tracker.create_pending_tracks(detections, assigned_dets)

    # -------- Promote pending tracks --------
    tracker.promote_tracks()

    # -------- Remove stale --------
    tracker.remove_stale()

    # -------- Visualization: Movement Mask --------
    det_vis = draw_movement_mask(frame2, movement_mask)
    out_det.write(det_vis)

    # -------- Visualization: Optical Flow --------
    flow_vis = draw_optical_flow(frame2, flow)
    out_flow.write(flow_vis)

    # -------- Visualization: Raw Detections --------
    box_vis = draw_detections(frame2, detections)
    out_boxes.write(box_vis)

    # -------- Visualization: Tracks --------
    track_vis = draw_tracks(frame2, tracker.tracks)
    out_tracks.write(track_vis)

    # Prepare next frame
    gray1 = gray2.copy()
    frame1 = frame2.copy()

cap.release()
out_boxes.release()
out_flow.release()
out_det.release()
out_tracks.release()

print("Saved boxes, flow, detection mask, and track videos.")