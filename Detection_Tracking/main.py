# main.py
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from detection import (
    detect_moving_objects,
    draw_movement_mask,
    draw_optical_flow,
    draw_detections,
)
from tracking import TrackManager, draw_tracks_left, draw_tracks_right
from calibration_stereo import P1, P2


# ---------------- Stereo helpers ----------------
def match_stereo_detections(detL, detR,
                            max_disp=150.0,
                            min_disp=2.0,
                            max_vertical=2.0,
                            max_height_ratio=1.5):
    """
    Left–right matching for rectified stereo, more robust:
      - y alignment (same scanline)
      - positive disparity (uL > uR)
      - similar bbox heights
      - each right detection used at most once
    """
    pairs = []
    usedR = set()

    for iL, dl in enumerate(detL):
        best_j = None
        best_score = 1e9

        for jR, dr in enumerate(detR):
            if jR in usedR:
                continue

            # Epipolar: y coordinates should match fairly well
            if abs(dl["cy"] - dr["cy"]) > max_vertical:
                continue

            # Disparity: rectified, so uL > uR
            disp = dl["cx"] - dr["cx"]
            if disp <= min_disp or disp > max_disp:
                continue

            # Height similarity
            hL = dl["h"]
            hR = dr["h"]
            if hL <= 0 or hR <= 0:
                continue
            ratio = max(hL, hR) / min(hL, hR)
            if ratio > max_height_ratio:
                continue

            # Simple score: disparity + vertical & height differences
            score = abs(disp) + 0.5 * abs(dl["cy"] - dr["cy"]) + 0.1 * abs(hL - hR)

            if score < best_score:
                best_score = score
                best_j = jR

        if best_j is not None:
            pairs.append((iL, best_j))
            usedR.add(best_j)

    return pairs


def triangulate_pair(dl, dr):
    """
    Triangulate a single stereo pair (center points) into 3D.
    dl, dr: detection dicts from left and right with 'cx', 'cy'.
    Returns: X, Y, Z in camera coordinates.
    """
    ptL = np.array([[dl["cx"]], [dl["cy"]]], dtype=np.float32)
    ptR = np.array([[dr["cx"]], [dr["cy"]]], dtype=np.float32)

    X_h = cv2.triangulatePoints(P1, P2, ptL, ptR)
    X_h /= X_h[3, 0]

    return float(X_h[0, 0]), float(X_h[1, 0]), float(X_h[2, 0])


# ---------------- Video Setup ----------------
video_left  = "./Detection_Tracking/inputs/seq2_image_02_video.mp4"
video_right = "./Detection_Tracking/inputs/seq2_image_03_video.mp4"

capL = cv2.VideoCapture(video_left)
capR = cv2.VideoCapture(video_right)

retL, frameL_1 = capL.read()
retR, frameR_1 = capR.read()
if not retL or not retR:
    raise RuntimeError(f"Cannot read first frames from {video_left} / {video_right}")

grayL_1 = cv2.cvtColor(frameL_1, cv2.COLOR_BGR2GRAY)
grayR_1 = cv2.cvtColor(frameR_1, cv2.COLOR_BGR2GRAY)
h, w = frameL_1.shape[:2]

fps = capL.get(cv2.CAP_PROP_FPS)
if fps <= 0 or np.isnan(fps):
    fps = 30.0
    print("Warning: FPS invalid → using fallback 30 FPS")

# Make sure output directory exists
out_dir = "./Detection_Tracking/out"
os.makedirs(out_dir, exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"XVID")

# -------- Output videos (Left | Right) --------
out_det_LR   = cv2.VideoWriter(os.path.join(out_dir, "people_det_LR.avi"),
                               fourcc, fps, (2 * w, h))
out_flow_LR  = cv2.VideoWriter(os.path.join(out_dir, "people_flow_LR.avi"),
                               fourcc, fps, (2 * w, h))
out_boxes_LR = cv2.VideoWriter(os.path.join(out_dir, "people_boxes_LR.avi"),
                               fourcc, fps, (2 * w, h))
out_tracks_LR = cv2.VideoWriter(os.path.join(out_dir, "people_tracks_LR.avi"),
                                fourcc, fps, (2 * w, h))


# ---------------- Tracking Setup ----------------
tracker = TrackManager(
    max_invisible=60,
    dist_thresh_px=120.0,
    min_confirm_frames=3
)

# 3D constant-velocity model
dt = 1.0 / fps
F = np.eye(6, dtype=np.float32)
F[0, 3] = dt
F[1, 4] = dt
F[2, 5] = dt

u = np.zeros((6, 1), dtype=np.float32)
Q = np.eye(6, dtype=np.float32) * 0.01  # process noise, tune

# Measurement: 3D position (X,Y,Z)
H = np.zeros((3, 6), dtype=np.float32)
H[0, 0] = 1.0
H[1, 1] = 1.0
H[2, 2] = 1.0

R = np.eye(3, dtype=np.float32) * 0.05  # measurement noise, tune

# 3D path history: track_id -> list of (X,Y,Z)
track_history = {}

# ---------------- Main Loop ----------------
frame_count = 0
total_frames = int(min(
    capL.get(cv2.CAP_PROP_FRAME_COUNT),
    capR.get(cv2.CAP_PROP_FRAME_COUNT)
))

frameL_prev = frameL_1.copy()
frameR_prev = frameR_1.copy()
grayL_prev = grayL_1.copy()
grayR_prev = grayR_1.copy()

while True:
    retL, frameL_2 = capL.read()
    retR, frameR_2 = capR.read()
    if not retL or not retR:
        break

    frame_count += 1
    # -------- Progress print (keep) --------
    if frame_count % 10 == 0:
        print(f"Processed {frame_count}/{total_frames}")

    grayL_2 = cv2.cvtColor(frameL_2, cv2.COLOR_BGR2GRAY)
    grayR_2 = cv2.cvtColor(frameR_2, cv2.COLOR_BGR2GRAY)

    # -------- Optical flow (left & right) --------
    flowL = cv2.calcOpticalFlowFarneback(
        grayL_prev, grayL_2, None,
        0.5, 3, 20, 5, 5, 1.2, 0
    )
    flowR = cv2.calcOpticalFlowFarneback(
        grayR_prev, grayR_2, None,
        0.5, 3, 20, 5, 5, 1.2, 0
    )

    # -------- Detection (left & right, independent) --------
    detL, maskL = detect_moving_objects(flowL, frameL_prev, frameL_2)
    detR, maskR = detect_moving_objects(flowR, frameR_prev, frameR_2)

    # -------- Stereo matching + triangulation --------
    pairs = match_stereo_detections(detL, detR)
    detections_3d = []
    for iL, iR in pairs:
        dl = detL[iL]
        dr = detR[iR]
        X, Y, Z = triangulate_pair(dl, dr)
        detections_3d.append({
            "X": X,
            "Y": Y,
            "Z": Z,
            "detL": dl,
            "detR": dr
        })

    # -------- Track prediction --------
    tracker.predict_tracks(F, u, Q)

    # -------- Association --------
    assigned_tracks, assigned_dets = tracker.associate(detections_3d, H, R)
    assigned_pending = tracker.associate_pending(detections_3d, H, R, assigned_dets)

    # -------- Create missing tracks --------
    tracker.create_pending_tracks(detections_3d, assigned_dets)

    # -------- Promote pending tracks --------
    tracker.promote_tracks()

    # -------- Remove stale --------
    tracker.remove_stale()

    # -------- Detection visualization video (red semitransparent mask) --------
    det_vis_L = draw_movement_mask(frameL_2, maskL)
    det_vis_R = draw_movement_mask(frameR_2, maskR)
    det_vis_LR = np.hstack((det_vis_L, det_vis_R))
    out_det_LR.write(det_vis_LR)

    # -------- Draw optical flow visualization --------
    flow_vis_L = draw_optical_flow(frameL_2, flowL)
    flow_vis_R = draw_optical_flow(frameR_2, flowR)
    flow_vis_LR = np.hstack((flow_vis_L, flow_vis_R))
    out_flow_LR.write(flow_vis_LR)

    # -------- Raw detection boxes visualization --------
    box_vis_L = draw_detections(frameL_2, detL)
    box_vis_R = draw_detections(frameR_2, detR)
    box_vis_LR = np.hstack((box_vis_L, box_vis_R))
    out_boxes_LR.write(box_vis_LR)

    # -------- Tracks visualization (left & right) --------
    tracks_vis_L = draw_tracks_left(frameL_2, tracker.tracks)
    tracks_vis_R = draw_tracks_right(frameR_2, tracker.tracks)
    tracks_vis_LR = np.hstack((tracks_vis_L, tracks_vis_R))
    out_tracks_LR.write(tracks_vis_LR)

    # -------- Record 3D positions for plotting --------
    for tr in tracker.tracks:
        tid = tr["id"]
        X = float(tr["x"][0, 0])
        Y = float(tr["x"][1, 0])
        Z = float(tr["x"][2, 0])
        track_history.setdefault(tid, []).append((X, Y, Z))

    # Prepare next frame
    grayL_prev = grayL_2.copy()
    grayR_prev = grayR_2.copy()
    frameL_prev = frameL_2.copy()
    frameR_prev = frameR_2.copy()

capL.release()
capR.release()
out_det_LR.release()
out_flow_LR.release()
out_boxes_LR.release()
out_tracks_LR.release()

print("Saved stereo videos (det, flow, boxes, tracks) as Left|Right.")


# --------------------------
# Final 3D Trajectory Plot
# --------------------------
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

for tid, pts in track_history.items():
    if len(pts) < 2:
        continue
    pts = np.array(pts)
    X = pts[:, 0]
    Y = pts[:, 1]
    Z = pts[:, 2]
    ax.plot(X, Y, Z, label=f"ID {tid}")
    ax.scatter(X[0], Y[0], Z[0], marker="o", s=30)
    ax.scatter(X[-1], Y[-1], Z[-1], marker="x", s=30)

ax.set_title("3D Trajectories of Tracked People")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()
