# main.py
import cv2
import numpy as np

from detection import detect_moving_objects
from tracking import TrackManager

# ---------------- Video Setup ----------------
video_path = "./Detection & Tracking/Sequence 1.mp4" # Run from project root
cap = cv2.VideoCapture(video_path)
ret, frame1 = cap.read()
if not ret:
    raise RuntimeError("Could not read first frame, Make sure to run the code from the repository root.")

gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
h, w = frame1.shape[:2]
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out_boxes = cv2.VideoWriter("people_boxes.avi", fourcc, fps, (w, h))

# ---------------- Kalman Filter Constants ----------------
dt = 1.0 / fps
F = np.eye(6, dtype=np.float32)
F[0, 2] = dt
F[1, 3] = dt
F[0, 4] = 0.5 * dt**2
F[1, 5] = 0.5 * dt**2
F[2, 4] = dt
F[3, 5] = dt

Q = np.eye(6, dtype=np.float32) * 0.01
H = np.zeros((2, 6), dtype=np.float32)
H[0, 0] = 1
H[1, 1] = 1
R = np.eye(2, dtype=np.float32) * 0.1
u = np.zeros((6, 1), dtype=np.float32)

# ---------------- Track Manager ----------------
tracker = TrackManager(max_invisible=15, dist_thresh=80)

# ---------------- Main Loop ----------------
frame_count = 0
while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # optical flow
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        0.5, 3, 20, 5, 5, 1.2, 0
    )

    # -------- Detection --------
    detections, movement_mask = detect_moving_objects(flow, frame1)

    # -------- Tracking --------
    tracker.predict_tracks(F, u, Q)
    assigned_tracks, assigned_dets = tracker.associate(detections, H, R)
    tracker.create_new_tracks(detections, assigned_dets)
    tracker.delete_old_tracks()

    # -------- Draw results --------
    output = frame1.copy()
    for tr in tracker.tracks:
        cx = int(tr["x"][0, 0])
        cy = int(tr["x"][1, 0])
        x_b, y_b, w_b, h_b = tr["bbox"]
        color = tr["color"]

        cv2.rectangle(output, (x_b, y_b), (x_b + w_b, y_b + h_b), color, 2)
        cv2.circle(output, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(
            output, f"ID {tr['id']}",
            (x_b, y_b - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color, 2
        )

    out_boxes.write(output)

    gray1 = gray2.copy()
    frame1 = frame2.copy()
    frame_count += 1

cap.release()
out_boxes.release()
print("Finished. Saved people_boxes.avi")
