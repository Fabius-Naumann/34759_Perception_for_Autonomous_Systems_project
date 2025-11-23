# tracking.py
import numpy as np
import cv2

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

# -------------- TRACKING VISUALIZATION -----------------
def draw_tracks(frame, tracks):
    """
    Draw confirmed tracks on the frame.
    Each track must contain:
        - id
        - bbox = (x, y, w, h)
        - x (Kalman state) → cx, cy
        - color (tuple)
    """
    vis = frame.copy()

    for tr in tracks:
        color = tr["color"]
        x, y, w, h = tr["bbox"]
        cx = int(tr["x"][0, 0])
        cy = int(tr["x"][1, 0])

        # Draw bounding box
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)

        # Draw center point
        cv2.circle(vis, (cx, cy), 5, color, -1)

        # Draw ID label
        cv2.putText(
            vis,
            f"ID {tr['id']}",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    return vis


# -------------------------------------------------------------
#                     TRACK MANAGER
#                 (with confirmation stage)
# -------------------------------------------------------------

class TrackManager:
    def __init__(self, max_invisible=40, dist_thresh=80.0,
                 min_confirm_frames=3):

        # real, confirmed tracks
        self.tracks = []

        # temporary candidates waiting to be confirmed
        self.pending_tracks = []

        self.next_id = 0

        self.max_invisible = max_invisible
        self.dist_thresh = dist_thresh
        self.min_confirm_frames = min_confirm_frames

    # ---------------------------------------------------------
    #         PREDICT all confirmed and pending tracks
    # ---------------------------------------------------------
    def predict_tracks(self, F, u, Q):
        for tr in self.tracks:
            tr["x"], tr["P"] = kf_predict(tr["x"], tr["P"], F, u, Q)
            tr["invisible"] += 1
            tr["age"] += 1

        for tr in self.pending_tracks:
            tr["x"], tr["P"] = kf_predict(tr["x"], tr["P"], F, u, Q)
            tr["miss"] += 1

    # ---------------------------------------------------------
    #      ASSOCIATE detections to confirmed tracks
    # ---------------------------------------------------------
    def associate(self, detections, H, R):
        assigned_tracks = set()
        assigned_dets = set()

        if len(self.tracks) == 0 or len(detections) == 0:
            return assigned_tracks, assigned_dets

        track_points = np.array(
            [[tr["x"][0, 0], tr["x"][1, 0]] for tr in self.tracks]
        )
        det_points = np.array([[d["cx"], d["cy"]] for d in detections])

        diff = track_points[:, None, :] - det_points[None, :, :]
        distances = np.linalg.norm(diff, axis=2)

        # Greedy assignment
        while True:
            t_idx, d_idx = np.unravel_index(np.argmin(distances), distances.shape)
            d = distances[t_idx, d_idx]

            if not np.isfinite(d) or d > self.dist_thresh:
                break
            if t_idx in assigned_tracks or d_idx in assigned_dets:
                distances[t_idx, d_idx] = np.inf
                continue

            det = detections[d_idx]
            Z = np.array([[det["cx"]], [det["cy"]]], dtype=np.float32)

            self.tracks[t_idx]["x"], self.tracks[t_idx]["P"] = kf_update(
                self.tracks[t_idx]["x"], self.tracks[t_idx]["P"], Z, H, R
            )
            self.tracks[t_idx]["bbox"] = (det["x"], det["y"], det["w"], det["h"])
            self.tracks[t_idx]["invisible"] = 0

            assigned_tracks.add(t_idx)
            assigned_dets.add(d_idx)

            distances[t_idx, :] = np.inf
            distances[:, d_idx] = np.inf

        return assigned_tracks, assigned_dets

    # ---------------------------------------------------------
    #      ASSOCIATE detections to PENDING (unconfirmed) tracks
    # ---------------------------------------------------------
    def associate_pending(self, detections, H, R, assigned_dets):

        # Handle cases with no pending tracks or no detections
        if len(self.pending_tracks) == 0 or len(detections) == 0:
            return set()

        pending_points = np.array(
            [[tr["x"][0, 0], tr["x"][1, 0]] for tr in self.pending_tracks]
        )
        det_points = np.array([[d["cx"], d["cy"]] for d in detections])

        diff = pending_points[:, None, :] - det_points[None, :, :]
        distances = np.linalg.norm(diff, axis=2)

        assigned_pending = set()

        while True:
            p_idx, d_idx = np.unravel_index(np.argmin(distances), distances.shape)
            d = distances[p_idx, d_idx]

            if (not np.isfinite(d)) or (d > self.dist_thresh):
                break
            if d_idx in assigned_dets:
                distances[p_idx, d_idx] = np.inf
                continue

            det = detections[d_idx]
            Z = np.array([[det["cx"]], [det["cy"]]], dtype=np.float32)

            self.pending_tracks[p_idx]["x"], self.pending_tracks[p_idx]["P"] = kf_update(
                self.pending_tracks[p_idx]["x"], self.pending_tracks[p_idx]["P"], Z, H, R
            )
            self.pending_tracks[p_idx]["bbox"] = (
                det["x"], det["y"], det["w"], det["h"]
            )
            self.pending_tracks[p_idx]["miss"] = 0
            self.pending_tracks[p_idx]["hits"] += 1

            assigned_pending.add(p_idx)
            assigned_dets.add(d_idx)

            distances[p_idx, :] = np.inf
            distances[:, d_idx] = np.inf

        return assigned_pending


    # ---------------------------------------------------------
    #       CREATE pending tracks for unmatched detections
    # ---------------------------------------------------------
    def create_pending_tracks(self, detections, assigned_dets):
        for d_idx, det in enumerate(detections):
            if d_idx in assigned_dets:
                continue

            x_kf, P_kf = init_kalman_state(det["cx"], det["cy"])

            self.pending_tracks.append({
                "x": x_kf,
                "P": P_kf,
                "bbox": (det["x"], det["y"], det["w"], det["h"]),
                "hits": 1,     # must reach min_confirm_frames
                "miss": 0
            })

    # ---------------------------------------------------------
    #      PROMOTE confirmed pending tracks → real tracks
    # ---------------------------------------------------------
    def promote_tracks(self):
        confirmed = [t for t in self.pending_tracks
                     if t["hits"] >= self.min_confirm_frames]

        for tr in confirmed:
            tid = self.next_id
            self.next_id += 1

            color = (
                int((37 * tid) % 255),
                int((17 * tid) % 255),
                int((29 * tid) % 255),
            )

            self.tracks.append({
                "id": tid,
                "x": tr["x"],
                "P": tr["P"],
                "bbox": tr["bbox"],
                "invisible": 0,
                "age": 1,
                "color": color,
            })

        # Remove promoted tracks from pending list
        self.pending_tracks = [
            t for t in self.pending_tracks
            if t["hits"] < self.min_confirm_frames
        ]

    # ---------------------------------------------------------
    #      REMOVE old pending and inactive real tracks
    # ---------------------------------------------------------
    def remove_stale(self):
        self.tracks = [
            tr for tr in self.tracks if tr["invisible"] < self.max_invisible
        ]
        # pending tracks disappear fast
        self.pending_tracks = [
            tr for tr in self.pending_tracks if tr["miss"] < 3
        ]
