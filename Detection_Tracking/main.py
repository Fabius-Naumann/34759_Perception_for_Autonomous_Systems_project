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

# ---------------- Sequence / paths ----------------
SEQ = "seq1"  # use a stereo sequence: *_image_02 & *_image_03

LEFT_VIDEO  = f"./Detection_Tracking/inputs/{SEQ}_image_02_video.mp4"
RIGHT_VIDEO = f"./Detection_Tracking/inputs/{SEQ}_image_03_video.mp4"

capL = cv2.VideoCapture(LEFT_VIDEO)
capR = cv2.VideoCapture(RIGHT_VIDEO)

retL, frameL1 = capL.read()
retR, frameR1 = capR.read()
if not (retL and retR):
    raise RuntimeError(f"Cannot read first stereo frames from {LEFT_VIDEO} and {RIGHT_VIDEO}")

# We'll do detection + tracking on the RIGHT camera
frame1 = frameR1.copy()
gray1  = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
h, w = frame1.shape[:2]

fps = capR.get(cv2.CAP_PROP_FPS)
if fps <= 0 or np.isnan(fps):
    fps = 30.0
    print("Warning: FPS invalid → using fallback 30 FPS")

# ---------------- Output dirs / writers ----------------
out_dir = f"./Detection_Tracking/out/{SEQ}/"
os.makedirs(out_dir, exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"XVID")

out_boxes   = cv2.VideoWriter(os.path.join(out_dir, "people_boxes.avi"),     fourcc, fps, (w, h))
out_flow    = cv2.VideoWriter(os.path.join(out_dir, "people_flow.avi"),      fourcc, fps, (w, h))
out_det     = cv2.VideoWriter(os.path.join(out_dir, "people_detection.avi"), fourcc, fps, (w, h))
out_tracks  = cv2.VideoWriter(os.path.join(out_dir, "people_tracks.avi"),    fourcc, fps, (w, h))
out_depth   = cv2.VideoWriter(os.path.join(out_dir, "people_depth.avi"),     fourcc, fps, (w, h))

# ---------------- Stereo SGBM + calibration (KITTI) ----------------
min_disp = 10
num_disp = 16 * 12  # 192
block_size = 11

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=8 * 3 * block_size * block_size,
    P2=32 * 3 * block_size * block_size,
    uniquenessRatio=12,
    speckleWindowSize=200,
    speckleRange=3,
    disp12MaxDiff=0
)

# KITTI-like calibration (same as in stereo.py)
f = 707.0493        # pixels
B = 0.4727          # meters
MAX_DEPTH = 80.0    # max depth we trust in meters

# ---------------- Tracking Setup (3D state) ----------------
tracker = TrackManager(
    max_invisible=60,
    dist_thresh=70.0,      # tighter gate in pixels
    min_confirm_frames=4   # less spurious tracks
)


dt = 1.0 / fps

# State: [cx, cy, z, vx, vy, vz]
F = np.eye(6, dtype=np.float32)
F[0, 3] = dt
F[1, 4] = dt
F[2, 5] = dt

u = np.zeros((6, 1), dtype=np.float32)
Q = np.eye(6, dtype=np.float32) * 2.0

# Measurement: [cx, cy, z]
H = np.zeros((3, 6), dtype=np.float32)
H[0, 0] = 1.0
H[1, 1] = 1.0
H[2, 2] = 1.0

R = np.eye(3, dtype=np.float32)
R[0, 0] = 5.0   # pixel noise X
R[1, 1] = 5.0   # pixel noise Y
R[2, 2] = 2.0   # depth noise (meters^2) – tune if needed

# ---------------- Main Loop ----------------
frame_count = 0
total_frames = int(capR.get(cv2.CAP_PROP_FRAME_COUNT))

while True:
    retL, frameL2 = capL.read()
    retR, frameR2 = capR.read()
    if not (retL and retR):
        break

    frame_count += 1
    if frame_count % 10 == 0:
        print(f"Processed {frame_count}/{total_frames}")

    # Right camera frames: for detection + tracking
    frame2 = frameR2
    gray2  = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # -------- Optical flow (right camera) --------
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        0.5, 3, 20, 5, 5, 1.2, 0
    )

    # -------- Detection (2D, right camera) --------
    detections, movement_mask = detect_moving_objects(flow, frame1, frame2)

    # -------- Stereo depth (left & right current frames) --------
    grayL2 = cv2.cvtColor(frameL2, cv2.COLOR_BGR2GRAY)
    grayR2 = gray2  # already computed

    disp = stereo.compute(grayL2, grayR2).astype(np.float32) / 16.0
    disp_safe = np.where(disp > min_disp, disp, np.nan)
    depth = (f * B) / disp_safe  # meters

    # Depth visualization (for debug)
    depth_vis = depth.copy()
    depth_vis[~np.isfinite(depth_vis)] = 0
    depth_vis = np.clip(depth_vis, 0, MAX_DEPTH)
    depth_norm = (depth_vis / MAX_DEPTH * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_PLASMA)
    out_depth.write(depth_color)

    # -------- Attach depth to detections and filter invalid ones --------
    detections_3d = []
    for det in detections:
        cx_det = int(round(det["cx"]))
        cy_det = int(round(det["cy"]))

        if cx_det < 0 or cx_det >= w or cy_det < 0 or cy_det >= h:
            continue

        z = depth[cy_det, cx_det]
        if (not np.isfinite(z)) or (z <= 0) or (z > MAX_DEPTH):
            continue

        det3d = det.copy()
        det3d["depth"] = float(z)
        detections_3d.append(det3d)

    # -------- Track prediction --------
    tracker.predict_tracks(F, u, Q)

    # -------- Association in 3D (measurement includes depth) --------
    assigned_tracks, assigned_dets = tracker.associate(detections_3d, H, R)
    assigned_pending = tracker.associate_pending(detections_3d, H, R, assigned_dets)

    # -------- Create missing tracks --------
    tracker.create_pending_tracks(detections_3d, assigned_dets)

    # -------- Promote pending tracks --------
    tracker.promote_tracks()

    # -------- Remove stale --------
    tracker.remove_stale()

    # -------- Visualization: Movement Mask (right) --------
    det_vis = draw_movement_mask(frame2, movement_mask)
    out_det.write(det_vis)

    # -------- Visualization: Optical Flow (right) --------
    flow_vis = draw_optical_flow(frame2, flow)
    out_flow.write(flow_vis)

    # -------- Visualization: Raw Detections (right) --------
    box_vis = draw_detections(frame2, detections_3d)
    out_boxes.write(box_vis)

    # -------- Visualization: 3D Tracks (right, with depth) --------
    track_vis = draw_tracks(frame2, tracker.tracks)
    out_tracks.write(track_vis)

    # Prepare next frame (right camera)
    gray1 = gray2.copy()
    frame1 = frame2.copy()

capL.release()
capR.release()
out_boxes.release()
out_flow.release()
out_det.release()
out_tracks.release()
out_depth.release()

print("Saved boxes, flow, detection mask, depth, and 3D track videos.")
