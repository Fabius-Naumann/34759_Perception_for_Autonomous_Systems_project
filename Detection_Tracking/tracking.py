# tracking.py
import numpy as np
import cv2
from calibration_stereo import P1, P2


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
    P = (np.eye(x.shape[0]) - K @ H) @ P
    return x, P


def init_kalman_state_3d(X, Y, Z):
    """
    3D constant-velocity state:
        [X, Y, Z, VX, VY, VZ]^T
    """
    x = np.zeros((6, 1), dtype=np.float32)
    x[:3, 0] = [X, Y, Z]
    P = np.eye(6, dtype=np.float32) * 500.0
    return x, P


# -------------- PROJECTION HELPERS -----------------


def project_3d_to_px(P, X, Y, Z):
    X_h = np.array([[X], [Y], [Z], [1.0]], dtype=np.float32)
    uvw = P @ X_h
    w = float(uvw[2, 0])
    if abs(w) < 1e-6:
        return None
    u = float(uvw[0, 0] / w)
    v = float(uvw[1, 0] / w)
    return u, v


def project_left(X, Y, Z):
    return project_3d_to_px(P1, X, Y, Z)


def project_right(X, Y, Z):
    return project_3d_to_px(P2, X, Y, Z)


# -------------- TRACKING VISUALIZATION -----------------


def draw_tracks_left(frame, tracks):
    vis = frame.copy()
    for tr in tracks:
        color = tr["color"]
        x, y, w, h = tr["bbox_L"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)

        X, Y, Z = tr["x"][0, 0], tr["x"][1, 0], tr["x"][2, 0]
        uv = project_left(X, Y, Z)
        if uv:
            cv2.circle(vis, (int(uv[0]), int(uv[1])), 5, color, -1)

        cv2.putText(
            vis, f"ID {tr['id']}", (x, max(0, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )
    return vis


def draw_tracks_right(frame, tracks):
    vis = frame.copy()
    for tr in tracks:
        color = tr["color"]
        x, y, w, h = tr["bbox_R"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)

        X, Y, Z = tr["x"][0, 0], tr["x"][1, 0], tr["x"][2, 0]
        uv = project_right(X, Y, Z)
        if uv:
            cv2.circle(vis, (int(uv[0]), int(uv[1])), 5, color, -1)

        cv2.putText(
            vis, f"ID {tr['id']}", (x, max(0, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )
    return vis


# -------------------------------------------------------------
#                     TRACK MANAGER (3D)
#             2D association in LEFT image
# -------------------------------------------------------------


class TrackManager:
    def __init__(self, max_invisible=40, dist_thresh_px=120.0,
                 min_confirm_frames=3):

        # real, confirmed tracks
        self.tracks = []

        # temporary candidates waiting to be confirmed
        self.pending_tracks = []

        self.next_id = 0

        self.max_invisible = max_invisible
        self.dist_thresh_px = dist_thresh_px
        self.min_confirm_frames = min_confirm_frames

    # ---------------------------------------------------------
    #         PREDICT all confirmed and pending tracks
    # ---------------------------------------------------------
    def predict_tracks(self, F, u, Q):
        # Confirmed tracks
        for tr in self.tracks:
            tr["x"], tr["P"] = kf_predict(tr["x"], tr["P"], F, u, Q)
            tr["invisible"] += 1
            tr["age"] += 1

            X, Y, Z = tr["x"][0, 0], tr["x"][1, 0], tr["x"][2, 0]

            # Move left bbox
            uvL = project_left(X, Y, Z)
            if uvL:
                cx, cy = uvL
                x_old, y_old, w_old, h_old = tr["bbox_L"]
                cx_old = x_old + 0.5 * w_old
                cy_old = y_old + 0.5 * h_old
                dx = int(cx - cx_old)
                dy = int(cy - cy_old)
                tr["bbox_L"] = (x_old + dx, y_old + dy, w_old, h_old)

            # Move right bbox
            uvR = project_right(X, Y, Z)
            if uvR:
                cx, cy = uvR
                x_old, y_old, w_old, h_old = tr["bbox_R"]
                cx_old = x_old + 0.5 * w_old
                cy_old = y_old + 0.5 * h_old
                dx = int(cx - cx_old)
                dy = int(cy - cy_old)
                tr["bbox_R"] = (x_old + dx, y_old + dy, w_old, h_old)

        # Pending tracks
        for tr in self.pending_tracks:
            tr["x"], tr["P"] = kf_predict(tr["x"], tr["P"], F, u, Q)
            tr["miss"] += 1

    # ---------------------------------------------------------
    #      ASSOCIATE detections to confirmed tracks
    #           (2D gating in left image)
    # ---------------------------------------------------------
    def associate(self, detections, H, R):
        assigned_tracks = set()
        assigned_dets = set()

        if len(self.tracks) == 0 or len(detections) == 0:
            return assigned_tracks, assigned_dets

        # Project tracks into left image
        track_uv = []
        track_idx_map = []
        for i, tr in enumerate(self.tracks):
            uv = project_left(tr["x"][0, 0], tr["x"][1, 0], tr["x"][2, 0])
            if uv:
                track_uv.append(uv)
                track_idx_map.append(i)

        if not track_uv:
            return assigned_tracks, assigned_dets

        track_uv = np.array(track_uv, dtype=np.float32)
        det_uv = np.array(
            [[d["detL"]["cx"], d["detL"]["cy"]] for d in detections],
            dtype=np.float32
        )

        diff = track_uv[:, None, :] - det_uv[None, :, :]
        dist = np.linalg.norm(diff, axis=2)

        while True:
            t_loc, d_idx = np.unravel_index(np.argmin(dist), dist.shape)
            d = dist[t_loc, d_idx]
            if not np.isfinite(d) or d > self.dist_thresh_px:
                break

            t_idx = track_idx_map[t_loc]
            if t_idx in assigned_tracks or d_idx in assigned_dets:
                dist[t_loc, d_idx] = np.inf
                continue

            det = detections[d_idx]
            Z_meas = np.array([[det["X"]], [det["Y"]], [det["Z"]]], np.float32)

            self.tracks[t_idx]["x"], self.tracks[t_idx]["P"] = kf_update(
                self.tracks[t_idx]["x"], self.tracks[t_idx]["P"], Z_meas, H, R
            )

            dl = det["detL"]
            dr = det["detR"]
            self.tracks[t_idx]["bbox_L"] = (dl["x"], dl["y"], dl["w"], dl["h"])
            self.tracks[t_idx]["bbox_R"] = (dr["x"], dr["y"], dr["w"], dr["h"])
            self.tracks[t_idx]["invisible"] = 0

            assigned_tracks.add(t_idx)
            assigned_dets.add(d_idx)
            dist[t_loc, :] = np.inf
            dist[:, d_idx] = np.inf

        return assigned_tracks, assigned_dets

    # ---------------------------------------------------------
    #      ASSOCIATE detections to PENDING (unconfirmed) tracks
    #           (also in 2D left image)
    # ---------------------------------------------------------
    def associate_pending(self, detections, H, R, assigned_dets):
        if len(self.pending_tracks) == 0 or len(detections) == 0:
            return set()

        pending_uv = []
        pending_idx_map = []
        for i, tr in enumerate(self.pending_tracks):
            uv = project_left(tr["x"][0, 0], tr["x"][1, 0], tr["x"][2, 0])
            if uv:
                pending_uv.append(uv)
                pending_idx_map.append(i)

        if not pending_uv:
            return set()

        pending_uv = np.array(pending_uv, dtype=np.float32)
        det_uv = np.array(
            [[d["detL"]["cx"], d["detL"]["cy"]] for d in detections],
            dtype=np.float32
        )

        diff = pending_uv[:, None, :] - det_uv[None, :, :]
        dist = np.linalg.norm(diff, axis=2)

        assigned_pending = set()

        while True:
            p_loc, d_idx = np.unravel_index(np.argmin(dist), dist.shape)
            d = dist[p_loc, d_idx]
            if not np.isfinite(d) or d > self.dist_thresh_px:
                break

            if d_idx in assigned_dets:
                dist[p_loc, d_idx] = np.inf
                continue

            p_idx = pending_idx_map[p_loc]
            det = detections[d_idx]

            Z_meas = np.array([[det["X"]], [det["Y"]], [det["Z"]]], np.float32)
            self.pending_tracks[p_idx]["x"], self.pending_tracks[p_idx]["P"] = kf_update(
                self.pending_tracks[p_idx]["x"], self.pending_tracks[p_idx]["P"], Z_meas, H, R
            )

            dl = det["detL"]
            dr = det["detR"]
            self.pending_tracks[p_idx]["bbox_L"] = (dl["x"], dl["y"], dl["w"], dl["h"])
            self.pending_tracks[p_idx]["bbox_R"] = (dr["x"], dr["y"], dr["w"], dr["h"])
            self.pending_tracks[p_idx]["miss"] = 0
            self.pending_tracks[p_idx]["hits"] += 1

            assigned_pending.add(p_idx)
            assigned_dets.add(d_idx)

            dist[p_loc, :] = np.inf
            dist[:, d_idx] = np.inf

        return assigned_pending

    # ---------------------------------------------------------
    #       CREATE pending tracks for unmatched detections
    # ---------------------------------------------------------
    def create_pending_tracks(self, detections, assigned_dets):
        for i, det in enumerate(detections):
            if i in assigned_dets:
                continue

            X, Y, Z = det["X"], det["Y"], det["Z"]
            x_kf, P_kf = init_kalman_state_3d(X, Y, Z)
            dl = det["detL"]
            dr = det["detR"]

            self.pending_tracks.append({
                "x": x_kf,
                "P": P_kf,
                "bbox_L": (dl["x"], dl["y"], dl["w"], dl["h"]),
                "bbox_R": (dr["x"], dr["y"], dr["w"], dr["h"]),
                "hits": 1,
                "miss": 0
            })

    # ---------------------------------------------------------
    #      PROMOTE confirmed pending tracks → real tracks
    # ---------------------------------------------------------
    def promote_tracks(self):
        confirmed = [t for t in self.pending_tracks
                     if t["hits"] >= self.min_confirm_frames]

        COLOR_POOL = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 255), (255, 128, 0), (0, 128, 255),
        ]

        for tr in confirmed:
            tid = self.next_id
            self.next_id += 1

            tr["id"] = tid
            tr["color"] = COLOR_POOL[tid % len(COLOR_POOL)]
            tr["invisible"] = 0
            tr["age"] = 1

            self.tracks.append(tr)

        self.pending_tracks = [
            t for t in self.pending_tracks if t["hits"] < self.min_confirm_frames
        ]

    # ---------------------------------------------------------
    #      REMOVE old pending and inactive real tracks
    # ---------------------------------------------------------
    def remove_stale(self):
        self.tracks = [
            t for t in self.tracks if t["invisible"] < self.max_invisible
        ]
        self.pending_tracks = [
            t for t in self.pending_tracks if t["miss"] < 8
        ]
