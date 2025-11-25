import numpy as np
import cv2


def kf_predict(x, P, F, u, Q):
    x = F @ x + u
    P = F @ P @ F.T + Q
    return x, P


def kf_update(x, P, Z, H, R):
    y = Z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ y
    I = np.eye(x.shape[0], dtype=np.float32)
    P = (I - K @ H) @ P
    return x, P


def init_kalman_state(cx, cy, depth=0.0):
    x = np.zeros((6, 1), dtype=np.float32)
    x[0, 0] = cx
    x[1, 0] = cy
    x[2, 0] = depth
    P = np.eye(6, dtype=np.float32) * 500.0
    return x, P

def draw_tracks(frame, tracks):
    """
    Draw tracks with class name and depth (no ID).
    """
    vis = frame.copy()

    for tr in tracks:

        x, y, w, h = tr["bbox"]
        cx = int(tr["x"][0])
        cy = int(tr["x"][1])

        # Color (class-based if preferred)
        color = tr.get("color", (0, 255, 255))

        # Draw box
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)

        # Draw center dot
        cv2.circle(vis, (cx, cy), 3, color, -1)

        # Depth
        depth = float(tr["x"][2])
        depth_txt = f"{depth:.1f} m"

        # Class name
        class_name = tr.get("class_name", "unknown")
        CLASS_LABELS =  ["bicycle", "car", "person"]

        # Final label: CLASS + DEPTH
        label = f"{CLASS_LABELS[int(class_name)]} | {depth_txt}"

        cv2.putText(
            vis,
            label,
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    return vis

class TrackManager:
    def __init__(self, max_invisible=40, dist_thresh=80.0, min_confirm_frames=3):
        self.tracks = []
        self.pending_tracks = []
        self.next_id = 0

        self.max_invisible = max_invisible
        self.dist_thresh = dist_thresh
        self.min_confirm_frames = min_confirm_frames

        self.depth_thresh = 4.0
        self.lambda_depth = 15.0

    def _allowed_distance(self, tr):
        vx = float(tr["x"][3, 0])
        vy = float(tr["x"][4, 0])
        speed = np.hypot(vx, vy)
        return self.dist_thresh + 1.5 * speed

    def predict_tracks(self, F, u, Q):
        # Predict ACTIVE tracks
        for tr in self.tracks:
            tr["x"], tr["P"] = kf_predict(tr["x"], tr["P"], F, u, Q)

            # Invisible counter (no measurement yet this frame)
            tr["invisible"] += 1
            tr["age"] += 1

            # ----- PREDICT BBOX POSITION -----
            cx = float(tr["x"][0, 0])
            cy = float(tr["x"][1, 0])

            x_old, y_old, w_old, h_old = tr["bbox"]
            dx = int(cx - (x_old + w_old / 2))
            dy = int(cy - (y_old + h_old / 2))

            tr["bbox"] = (x_old + dx, y_old + dy, w_old, h_old)

            # Keep last class if somehow missing
            if "class_name" not in tr:
                tr["class_name"] = "unknown"

        # Predict PENDING tracks
        for tr in self.pending_tracks:
            tr["x"], tr["P"] = kf_predict(tr["x"], tr["P"], F, u, Q)
            tr["miss"] += 1

            # Also preserve class here
            if "class_name" not in tr:
                tr["class_name"] = "unknown"

    def associate(self, detections, H, R):
        assigned_tracks = set()
        assigned_dets = set()

        if len(self.tracks) == 0 or len(detections) == 0:
            return assigned_tracks, assigned_dets

        T = len(self.tracks)
        D = len(detections)
        big_cost = 1e6
        cost = np.full((T, D), big_cost, dtype=np.float32)

        for t_idx, tr in enumerate(self.tracks):
            cx_t = float(tr["x"][0, 0])
            cy_t = float(tr["x"][1, 0])
            z_t = float(tr["x"][2, 0])

            allowed_img = self._allowed_distance(tr)

            for d_idx, det in enumerate(detections):
                cx_d = det["cx"]
                cy_d = det["cy"]

                d_img = np.hypot(cx_d - cx_t, cy_d - cy_t)
                if d_img > allowed_img:
                    continue

                dz = 0.0
                if "depth" in det:
                    dz = abs(det["depth"] - z_t)
                    if dz > self.depth_thresh:
                        continue

                cost[t_idx, d_idx] = d_img + self.lambda_depth * dz

        from scipy.optimize import linear_sum_assignment
        row, col = linear_sum_assignment(cost)

        for t_idx, d_idx in zip(row, col):
            if cost[t_idx, d_idx] >= big_cost:
                continue

            det = detections[d_idx]

            Z = np.array(
                [[det["cx"]],
                 [det["cy"]],
                 [det["depth"]]],
                dtype=np.float32
            )

            tr = self.tracks[t_idx]
            tr["x"], tr["P"] = kf_update(tr["x"], tr["P"], Z, H, R)
            tr["bbox"] = (det["x"], det["y"], det["w"], det["h"])
            tr["invisible"] = 0

            if "class_name" in det:
                tr["class_name"] = det["class_name"]

            assigned_tracks.add(t_idx)
            assigned_dets.add(d_idx)

        return assigned_tracks, assigned_dets

    def associate_pending(self, detections, H, R, assigned_dets):
        assigned_pending = set()
        if len(self.pending_tracks) == 0 or len(detections) == 0:
            return assigned_pending

        T = len(self.pending_tracks)
        D = len(detections)
        big_cost = 1e6
        cost = np.full((T, D), big_cost, dtype=np.float32)

        for p_idx, tr in enumerate(self.pending_tracks):
            cx_t = tr["x"][0, 0]
            cy_t = tr["x"][1, 0]
            z_t = tr["x"][2, 0]

            allowed_img = self._allowed_distance(tr)

            for d_idx, det in enumerate(detections):
                if d_idx in assigned_dets:
                    continue

                cx_d = det["cx"]
                cy_d = det["cy"]

                d_img = np.hypot(cx_d - cx_t, cy_d - cy_t)
                if d_img > allowed_img:
                    continue

                dz = 0.0
                if "depth" in det:
                    dz = abs(det["depth"] - z_t)
                    if dz > self.depth_thresh:
                        continue

                cost[p_idx, d_idx] = d_img + self.lambda_depth * dz

        from scipy.optimize import linear_sum_assignment
        row, col = linear_sum_assignment(cost)

        for p_idx, d_idx in zip(row, col):
            if cost[p_idx, d_idx] >= big_cost:
                continue
            if d_idx in assigned_dets:
                continue

            det = detections[d_idx]

            Z = np.array(
                [[det["cx"]],
                 [det["cy"]],
                 [det["depth"]]],
                dtype=np.float32
            )

            tr = self.pending_tracks[p_idx]
            tr["x"], tr["P"] = kf_update(tr["x"], tr["P"], Z, H, R)
            tr["bbox"] = (det["x"], det["y"], det["w"], det["h"])
            tr["hits"] += 1
            tr["miss"] = 0

            if "class_name" in det:
                tr["class_name"] = det["class_name"]

            assigned_pending.add(p_idx)
            assigned_dets.add(d_idx)

        return assigned_pending

    def create_pending_tracks(self, detections, assigned_dets):
        for d_idx, det in enumerate(detections):
            if d_idx in assigned_dets:
                continue

            depth = det["depth"]
            x_kf, P_kf = init_kalman_state(det["cx"], det["cy"], depth)

            new_track = {
                "x": x_kf,
                "P": P_kf,
                "bbox": (det["x"], det["y"], det["w"], det["h"]),
                "hits": 1,
                "miss": 0,
            }

            # Always initialise with some class (or "unknown")
            new_track["class_name"] = det.get("class_name", "unknown")

            self.pending_tracks.append(new_track)

    def promote_tracks(self):
        confirmed = [
            t for t in self.pending_tracks
            if t["hits"] >= self.min_confirm_frames
        ]

        COLOR_POOL = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 255), (255, 128, 0), (0, 128, 255),
            (100, 200, 50), (200, 50, 100), (50, 200, 200),
        ]

        for tr in confirmed:
            tid = self.next_id
            self.next_id += 1

            tr_final = {
                "id": tid,
                "x": tr["x"],
                "P": tr["P"],
                "bbox": tr["bbox"],
                "invisible": 0,
                "age": 1,
                "color": COLOR_POOL[tid % len(COLOR_POOL)],
            }

            tr_final["class_name"] = tr.get("class_name", "unknown")

            self.tracks.append(tr_final)

        self.pending_tracks = [
            t for t in self.pending_tracks
            if t["hits"] < self.min_confirm_frames
        ]

    def remove_stale(self):
        self.tracks = [
            tr for tr in self.tracks
            if tr["invisible"] < self.max_invisible
        ]
        self.pending_tracks = [
            tr for tr in self.pending_tracks
            if tr["miss"] < 8
        ]
