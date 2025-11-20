# tracking.py
import numpy as np

# ---------------- Kalman filter functions -----------------

def kf_predict(x, P, F, u, Q):
    x = F @ x + u
    P = F @ P @ F.T + Q
    return x, P

def kf_update(x, P, Z, H, R):
    y = Z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ y
    I = np.eye(x.shape[0])
    P = (I - K @ H) @ P
    return x, P

def init_kalman_state(cx, cy):
    x = np.zeros((6, 1), dtype=np.float32)
    x[0, 0] = cx
    x[1, 0] = cy
    P = np.eye(6, dtype=np.float32) * 500.0
    return x, P

# ---------------- Track Manager -----------------

class TrackManager:
    def __init__(self, max_invisible=15, dist_thresh=80.0):
        self.tracks = []
        self.next_id = 0
        self.max_invisible = max_invisible
        self.dist_thresh = dist_thresh

    def predict_tracks(self, F, u, Q):
        for tr in self.tracks:
            tr["x"], tr["P"] = kf_predict(tr["x"], tr["P"], F, u, Q)
            tr["invisible"] += 1
            tr["age"] += 1

    def associate(self, detections, H, R):
        assigned_tracks = set()
        assigned_dets = set()

        if len(self.tracks) == 0 or len(detections) == 0:
            return assigned_tracks, assigned_dets

        track_points = np.array([[tr["x"][0, 0], tr["x"][1, 0]] for tr in self.tracks])
        det_points = np.array([[d["cx"], d["cy"]] for d in detections])

        # distance matrix
        diff = track_points[:, None, :] - det_points[None, :, :]
        distances = np.linalg.norm(diff, axis=2)

        # greedy nearest-neighbor
        while True:
            t_idx, d_idx = np.unravel_index(np.argmin(distances), distances.shape)
            d = distances[t_idx, d_idx]

            if not np.isfinite(d) or d > self.dist_thresh:
                break
            if t_idx in assigned_tracks or d_idx in assigned_dets:
                distances[t_idx, d_idx] = np.inf
                continue

            # update the track
            det = detections[d_idx]
            Z = np.array([[det["cx"]], [det["cy"]]], dtype=np.float32)
            self.tracks[t_idx]["x"], self.tracks[t_idx]["P"] = kf_update(
                self.tracks[t_idx]["x"], self.tracks[t_idx]["P"], Z, H, R
            )
            self.tracks[t_idx]["bbox"] = (det["x"], det["y"], det["w"], det["h"])
            self.tracks[t_idx]["invisible"] = 0

            assigned_tracks.add(t_idx)
            assigned_dets.add(d_idx)

            # invalidate row & column
            distances[t_idx, :] = np.inf
            distances[:, d_idx] = np.inf

        return assigned_tracks, assigned_dets

    def create_new_tracks(self, detections, assigned_dets):
        for d_idx, det in enumerate(detections):
            if d_idx in assigned_dets:
                continue

            x_kf, P_kf = init_kalman_state(det["cx"], det["cy"])
            tid = self.next_id
            self.next_id += 1

            # stable color per ID
            color = (
                int((37 * tid) % 255),
                int((17 * tid) % 255),
                int((29 * tid) % 255),
            )

            self.tracks.append({
                "id": tid,
                "x": x_kf,
                "P": P_kf,
                "bbox": (det["x"], det["y"], det["w"], det["h"]),
                "invisible": 0,
                "age": 1,
                "color": color,
            })

    def delete_old_tracks(self):
        self.tracks = [tr for tr in self.tracks if tr["invisible"] < self.max_invisible]
