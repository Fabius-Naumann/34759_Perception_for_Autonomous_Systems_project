import cv2
import numpy as np

# ------------- Kalman filter functions -------------

def kf_update(x, P, Z, H, R):
    y = Z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ y
    I = np.eye(x.shape[0])
    P = (I - K @ H) @ P
    return x, P

def kf_predict(x, P, F, u, Q):
    x = F @ x + u
    P = F @ P @ F.T + Q
    return x, P

def init_kalman_state(cx, cy):
    # State: [x, y, vx, vy, ax, ay]^T
    x = np.zeros((6, 1), dtype=np.float32)
    x[0, 0] = cx
    x[1, 0] = cy
    P = np.eye(6, dtype=np.float32) * 500.0
    return x, P

# ------------- Video setup -------------

cap = cv2.VideoCapture('./Detection & Tracking/Sequence 1.mp4') # Run from project root
ret, frame1 = cap.read()
if not ret:
    raise RuntimeError("Could not read first frame")

gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

h, w = frame1.shape[:2]
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or np.isnan(fps):
    fps = 20.0
dt = 1.0 / fps

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_flow = cv2.VideoWriter('people_flow.avi', fourcc, fps, (w, h))
out_mask = cv2.VideoWriter('people_movement.avi', fourcc, fps, (w, h))
out_boxes = cv2.VideoWriter('people_boxes.avi', fourcc, fps, (w, h))

# ------------- Kalman filter constants -------------

F = np.eye(6, dtype=np.float32)
F[0, 2] = dt
F[1, 3] = dt
F[0, 4] = 0.5 * dt**2
F[1, 5] = 0.5 * dt**2
F[2, 4] = dt
F[3, 5] = dt

Q = np.eye(6, dtype=np.float32) * 0.01

H = np.zeros((2, 6), dtype=np.float32)
H[0, 0] = 1.0
H[1, 1] = 1.0

R = np.eye(2, dtype=np.float32) * 0.1
u = np.zeros((6, 1), dtype=np.float32)

# ------------- Track management -------------

tracks = []  # list of dicts: {id, x, P, bbox, invisible, age, color}
next_track_id = 0
max_invisible_frames = 15
distance_threshold = 80.0  # pixels

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
i = 0

while ret and i < 200:
    ret, frame2 = cap.read()
    if not ret:
        break

    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Optical flow
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        0.5, 3, 20, 5, 5, 1.2, 0
    )

    # Movement mask (LESS FALSE POSITIVES: higher threshold)
    mag = np.linalg.norm(flow, axis=2)
    movement_mask = mag > 5.0  # was 3.0

    flow = flow * movement_mask[..., np.newaxis]

    # Flow visualization
    step = 8
    vis = frame1.copy()
    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            if not movement_mask[y0, x0]:
                continue
            fx, fy = flow[y0, x0]
            end_point = (int(x0 + fx), int(y0 + fy))
            cv2.arrowedLine(vis, (x0, y0), end_point, (0, 0, 255), 1, tipLength=0.3)

    # Movement mask image
    img_movement = np.stack([movement_mask]*3, axis=-1) * frame1

    # ------------- Detection from mask (connected components) -------------

    boxes_frame = frame1.copy()

    mask_uint8 = (movement_mask.astype(np.uint8) * 255)

    # LESS FALSE POSITIVES: stronger morphological cleaning
    kernel_open = np.ones((5, 5), np.uint8)
    kernel_close = np.ones((9, 9), np.uint8)
    mask_clean = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel_open)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_close)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_clean)

    detections = []  # each as dict: {cx, cy, x, y, w, h}
    for label in range(1, num_labels):  # skip background
        x_b, y_b, w_b, h_b, area = stats[label]

        # LESS FALSE POSITIVES: larger area and size constraints
        if area < 800:  # was 150
            continue
        if w_b < 20 or h_b < 40:
            continue

        # Simple aspect ratio check (people-ish blobs)
        aspect = h_b / (w_b + 1e-6)
        if aspect < 0.7 or aspect > 4.0:
            continue

        cx, cy = centroids[label]
        detections.append({
            'cx': float(cx),
            'cy': float(cy),
            'x': int(x_b),
            'y': int(y_b),
            'w': int(w_b),
            'h': int(h_b),
        })

    # ------------- Predict existing tracks -------------

    for tr in tracks:
        tr['x'], tr['P'] = kf_predict(tr['x'], tr['P'], F, u, Q)
        tr['invisible'] += 1
        tr['age'] += 1

    # ------------- Associate detections to tracks -------------

    assigned_tracks = set()
    assigned_dets = set()

    if len(tracks) > 0 and len(detections) > 0:
        track_points = np.array([[tr['x'][0, 0], tr['x'][1, 0]] for tr in tracks], dtype=np.float32)
        det_points = np.array([[d['cx'], d['cy']] for d in detections], dtype=np.float32)

        # distances: shape (num_tracks, num_dets)
        diff = track_points[:, None, :] - det_points[None, :, :]
        distances = np.linalg.norm(diff, axis=2)

        # Greedy nearest neighbor matching
        while True:
            t_idx, d_idx = np.unravel_index(np.argmin(distances), distances.shape)
            min_dist = distances[t_idx, d_idx]
            if not np.isfinite(min_dist) or min_dist > distance_threshold:
                break

            if t_idx in assigned_tracks or d_idx in assigned_dets:
                distances[t_idx, d_idx] = np.inf
                continue

            det = detections[d_idx]
            Z = np.array([[det['cx']], [det['cy']]], dtype=np.float32)
            tracks[t_idx]['x'], tracks[t_idx]['P'] = kf_update(tracks[t_idx]['x'], tracks[t_idx]['P'], Z, H, R)
            tracks[t_idx]['bbox'] = (det['x'], det['y'], det['w'], det['h'])
            tracks[t_idx]['invisible'] = 0

            assigned_tracks.add(t_idx)
            assigned_dets.add(d_idx)

            distances[t_idx, :] = np.inf
            distances[:, d_idx] = np.inf

    # ------------- Create new tracks for unmatched detections -------------

    for d_idx, det in enumerate(detections):
        if d_idx in assigned_dets:
            continue
        cx, cy = det['cx'], det['cy']
        x_kf, P_kf = init_kalman_state(cx, cy)
        track_id = next_track_id
        next_track_id += 1

        color = (
            int((37 * track_id) % 255),
            int((17 * track_id) % 255),
            int((29 * track_id) % 255),
        )

        tracks.append({
            'id': track_id,
            'x': x_kf,
            'P': P_kf,
            'bbox': (det['x'], det['y'], det['w'], det['h']),
            'invisible': 0,
            'age': 1,
            'color': color,
        })

    # ------------- Remove stale tracks -------------

    tracks = [tr for tr in tracks if tr['invisible'] < max_invisible_frames]

    # ------------- Draw tracked boxes and centers -------------

    for tr in tracks:
        cx = int(tr['x'][0, 0])
        cy = int(tr['x'][1, 0])
        x_b, y_b, w_b, h_b = tr['bbox']
        color = tr['color']
        tid = tr['id']

        x_b = max(0, min(x_b, w - 1))
        y_b = max(0, min(y_b, h - 1))
        x2 = max(0, min(x_b + w_b, w - 1))
        y2 = max(0, min(y_b + h_b, h - 1))

        cv2.rectangle(boxes_frame, (x_b, y_b), (x2, y2), color, 2)
        cv2.circle(boxes_frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(
            boxes_frame,
            f"ID {tid}",
            (x_b, y_b - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    # ------------- Write outputs -------------

    out_flow.write(vis)
    out_mask.write(img_movement)
    out_boxes.write(boxes_frame)

    i += 1
    if i % 10 == 0:
        print(f"Processed {i}/{total_frames} frames")

    gray1 = gray2.copy()
    frame1 = frame2.copy()

cap.release()
out_flow.release()
out_mask.release()
out_boxes.release()
print("Saved people_flow.avi, people_movement.avi, people_boxes.avi")
