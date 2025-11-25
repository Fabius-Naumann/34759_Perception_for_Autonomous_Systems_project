import numpy as np
import cv2

# ---------------- Kalman filter functions -----------------

def kf_predict(x, P, F, u, Q):
    """
    Generic Kalman predict step.

    x: (n,1) state
    P: (n,n) covariance
    F: (n,n) transition
    u: (n,1) control
    Q: (n,n) process noise
    """
    x = F @ x + u
    P = F @ P @ F.T + Q
    return x, P


def kf_update(x, P, Z, H, R):
    """
    Generic Kalman update step.

    Z: (m,1) measurement
    H: (m,n) measurement matrix
    R: (m,m) measurement noise
    """
    y = Z - H @ x                    # innovation
    S = H @ P @ H.T + R              # innovation covariance
    K = P @ H.T @ np.linalg.inv(S)   # Kalman gain

    x = x + K @ y                    # updated state

    # Standard covariance update
    n = P.shape[0]
    I = np.eye(n, dtype=np.float32)
    P = (I - K @ H) @ P

    return x, P


def init_kalman_state(cx, cy, depth=0.0):
    """
    Initialize 3D tracking state.

    State vector (6x1):
      [ cx, cy, z, vx, vy, vz ]^T

    cx, cy: image coordinates (pixels)
    z:      depth (meters)
    v*:     velocities (pixels/s, meters/s)
    """
    x = np.zeros((6, 1), dtype=np.float32)
    x[0, 0] = cx
    x[1, 0] = cy
    x[2, 0] = depth
    P = np.eye(6, dtype=np.float32) * 500.0
    return x, P


# -------------- TRACKING VISUALIZATION -----------------

def draw_tracks(frame, tracks):
    """
    Draw confirmed tracks on the frame.
    Each track must contain:
        - id
        - bbox = (x, y, w, h)
        - x (Kalman state → cx, cy, z)
        - color (tuple)
    """
    vis = frame.copy()

    for tr in tracks:
        color = tr["color"]
        x, y, w, h = tr["bbox"]
        cx = int(tr["x"][0, 0])
        cy = int(tr["x"][1, 0])
        z  = float(tr["x"][2, 0])

        # Draw bounding box
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)

        # Draw center point
        cv2.circle(vis, (cx, cy), 5, color, -1)

        # Draw ID + depth label
        label = f"ID {tr['id']}  z={z:.1f}m"
        cv2.putText(
            vis,
            label,
            (x, max(15, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    return vis


# -------------------------------------------------------------
#                     TRACK MANAGER
#                  3D-aware association
# -------------------------------------------------------------

class TrackManager:
    def __init__(
        self,
        max_invisible=40,
        dist_thresh=80.0,
        min_confirm_frames=3,
    ):
        """
        max_invisible: how many frames a track can be missing before deletion
        dist_thresh:  base gating threshold in pixels
        min_confirm_frames: frames before a pending track becomes confirmed
        """

        # real, confirmed tracks
        self.tracks = []

        # temporary candidates waiting to be confirmed
        self.pending_tracks = []

        self.next_id = 0

        self.max_invisible = max_invisible
        self.dist_thresh = dist_thresh
        self.min_confirm_frames = min_confirm_frames

        # Depth-related tuning (meters)
        self.depth_thresh = 4.0
        self.lambda_depth = 15.0  # weight depth difference vs. image distance

    # ---------------------------------------------------------
    #          Helper: distance gating based on velocity
    # ---------------------------------------------------------
    def _allowed_distance(self, tr):
        """
        Gating distance in image plane, expanded by speed.

        vx, vy indices for 3D state [cx, cy, z, vx, vy, vz]:
            vx → index 3
            vy → index 4
        """
        vx = float(tr["x"][3, 0])
        vy = float(tr["x"][4, 0])
        speed = np.hypot(vx, vy)

        # Base + speed-dependent expansion
        return self.dist_thresh + 1.5 * speed

    # ---------------------------------------------------------
    #         PREDICT all confirmed and pending tracks
    # ---------------------------------------------------------
    def predict_tracks(self, F, u, Q):
        for tr in self.tracks:
            tr["x"], tr["P"] = kf_predict(tr["x"], tr["P"], F, u, Q)
            tr["invisible"] += 1
            tr["age"] += 1

            # Move bbox center with prediction (cx, cy at indices 0,1)
            cx = float(tr["x"][0, 0])
            cy = float(tr["x"][1, 0])
            x_old, y_old, w_old, h_old = tr["bbox"]
            dx = int(round(cx - (x_old + w_old / 2.0)))
            dy = int(round(cy - (y_old + h_old / 2.0)))
            tr["bbox"] = (x_old + dx, y_old + dy, w_old, h_old)

        for tr in self.pending_tracks:
            tr["x"], tr["P"] = kf_predict(tr["x"], tr["P"], F, u, Q)
            tr["miss"] += 1

    # ---------------------------------------------------------
    #      ASSOCIATE detections to confirmed tracks (3D-aware)
    # ---------------------------------------------------------
    def associate(self, detections, H, R):
        """
        Associate detections to confirmed tracks using a cost matrix
        and Hungarian algorithm. Cost uses:

            - image distance (pixels)
            - optional depth difference (meters) when available
        """
        assigned_tracks = set()
        assigned_dets = set()

        if len(self.tracks) == 0 or len(detections) == 0:
            return assigned_tracks, assigned_dets

        T = len(self.tracks)
        D = len(detections)

        # Large value means "no assignment"
        big_cost = 1e6
        cost = np.full((T, D), big_cost, dtype=np.float32)

        # Build cost matrix
        for t_idx, tr in enumerate(self.tracks):
            cx_t = float(tr["x"][0, 0])
            cy_t = float(tr["x"][1, 0])
            z_t  = float(tr["x"][2, 0])

            allowed_img = self._allowed_distance(tr)

            for d_idx, det in enumerate(detections):
                cx_d = float(det["cx"])
                cy_d = float(det["cy"])

                dx = cx_d - cx_t
                dy = cy_d - cy_t
                d_img = float(np.hypot(dx, dy))

                if d_img > allowed_img:
                    continue  # still cost = big_cost

                # Depth difference if available
                if "depth" in det:
                    z_d = float(det["depth"])
                    dz = abs(z_d - z_t)
                else:
                    dz = 0.0

                if "depth" in det and dz > self.depth_thresh:
                    continue

                # Final cost: pixels + weighted depth
                cost_val = d_img + self.lambda_depth * dz
                cost[t_idx, d_idx] = cost_val

        # Hungarian assignment
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost)
        except Exception:
            # Fallback to greedy if scipy is not available
            row_ind, col_ind = [], []
            cost_copy = cost.copy()
            while True:
                t_idx, d_idx = np.unravel_index(
                    np.argmin(cost_copy), cost_copy.shape
                )
                if cost_copy[t_idx, d_idx] >= big_cost:
                    break
                row_ind.append(t_idx)
                col_ind.append(d_idx)
                cost_copy[t_idx, :] = big_cost
                cost_copy[:, d_idx] = big_cost

        # Apply assignments
        for t_idx, d_idx in zip(row_ind, col_ind):
            if cost[t_idx, d_idx] >= big_cost:
                continue  # invalid match

            det = detections[d_idx]

            # Build measurement vector according to H (2D or 3D)
            m = H.shape[0]
            if m == 3 and "depth" in det:
                Z = np.array(
                    [[det["cx"]], [det["cy"]], [det["depth"]]],
                    dtype=np.float32,
                )
            else:
                # 2D fallback
                Z = np.array(
                    [[det["cx"]], [det["cy"]]],
                    dtype=np.float32,
                )

            self.tracks[t_idx]["x"], self.tracks[t_idx]["P"] = kf_update(
                self.tracks[t_idx]["x"], self.tracks[t_idx]["P"], Z, H, R
            )
            self.tracks[t_idx]["bbox"] = (
                det["x"], det["y"], det["w"], det["h"]
            )
            self.tracks[t_idx]["invisible"] = 0

            assigned_tracks.add(t_idx)
            assigned_dets.add(d_idx)

        return assigned_tracks, assigned_dets

    # ---------------------------------------------------------
    #      ASSOCIATE detections to PENDING (unconfirmed) tracks
    # ---------------------------------------------------------
    def associate_pending(self, detections, H, R, assigned_dets):
        """
        Same as associate(), but for pending tracks.
        """
        assigned_pending = set()

        if len(self.pending_tracks) == 0 or len(detections) == 0:
            return assigned_pending

        T = len(self.pending_tracks)
        D = len(detections)

        big_cost = 1e6
        cost = np.full((T, D), big_cost, dtype=np.float32)

        for p_idx, tr in enumerate(self.pending_tracks):
            cx_t = float(tr["x"][0, 0])
            cy_t = float(tr["x"][1, 0])
            z_t  = float(tr["x"][2, 0])

            allowed_img = self._allowed_distance(tr)

            for d_idx, det in enumerate(detections):
                if d_idx in assigned_dets:
                    continue

                cx_d = float(det["cx"])
                cy_d = float(det["cy"])

                dx = cx_d - cx_t
                dy = cy_d - cy_t
                d_img = float(np.hypot(dx, dy))

                if d_img > allowed_img:
                    continue

                if "depth" in det:
                    z_d = float(det["depth"])
                    dz = abs(z_d - z_t)
                else:
                    dz = 0.0

                if "depth" in det and dz > self.depth_thresh:
                    continue

                cost_val = d_img + self.lambda_depth * dz
                cost[p_idx, d_idx] = cost_val

        # Hungarian (or greedy fallback)
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost)
        except Exception:
            row_ind, col_ind = [], []
            cost_copy = cost.copy()
            while True:
                p_idx, d_idx = np.unravel_index(
                    np.argmin(cost_copy), cost_copy.shape
                )
                if cost_copy[p_idx, d_idx] >= big_cost:
                    break
                row_ind.append(p_idx)
                col_ind.append(d_idx)
                cost_copy[p_idx, :] = big_cost
                cost_copy[:, d_idx] = big_cost

        for p_idx, d_idx in zip(row_ind, col_ind):
            if cost[p_idx, d_idx] >= big_cost:
                continue
            if d_idx in assigned_dets:
                continue

            det = detections[d_idx]

            m = H.shape[0]
            if m == 3 and "depth" in det:
                Z = np.array(
                    [[det["cx"]], [det["cy"]], [det["depth"]]],
                    dtype=np.float32,
                )
            else:
                Z = np.array(
                    [[det["cx"]], [det["cy"]]],
                    dtype=np.float32,
                )

            self.pending_tracks[p_idx]["x"], self.pending_tracks[p_idx]["P"] = kf_update(
                self.pending_tracks[p_idx]["x"],
                self.pending_tracks[p_idx]["P"],
                Z,
                H,
                R,
            )
            self.pending_tracks[p_idx]["bbox"] = (
                det["x"], det["y"], det["w"], det["h"]
            )
            self.pending_tracks[p_idx]["miss"] = 0
            self.pending_tracks[p_idx]["hits"] += 1

            assigned_pending.add(p_idx)
            assigned_dets.add(d_idx)

        return assigned_pending

    # ---------------------------------------------------------
    #       CREATE pending tracks for unmatched detections
    # ---------------------------------------------------------
    def create_pending_tracks(self, detections, assigned_dets):
        for d_idx, det in enumerate(detections):
            if d_idx in assigned_dets:
                continue

            depth = float(det.get("depth", 0.0))
            x_kf, P_kf = init_kalman_state(det["cx"], det["cy"], depth)

            self.pending_tracks.append(
                {
                    "x": x_kf,
                    "P": P_kf,
                    "bbox": (det["x"], det["y"], det["w"], det["h"]),
                    "hits": 1,  # must reach min_confirm_frames
                    "miss": 0,
                }
            )

    # ---------------------------------------------------------
    #      PROMOTE confirmed pending tracks → real tracks
    # ---------------------------------------------------------
    def promote_tracks(self):
        confirmed = [
            t for t in self.pending_tracks
            if t["hits"] >= self.min_confirm_frames
        ]

        for tr in confirmed:
            tid = self.next_id
            self.next_id += 1

            COLOR_POOL = [
                (255, 0, 0),
                (0, 255, 0),
                (0, 0, 255),
                (255, 255, 0),
                (255, 0, 255),
                (0, 255, 255),
                (128, 0, 255),
                (255, 128, 0),
                (0, 128, 255),
                (100, 200, 50),
                (200, 50, 100),
                (50, 200, 200),
            ]
            color = COLOR_POOL[tid % len(COLOR_POOL)]

            self.tracks.append(
                {
                    "id": tid,
                    "x": tr["x"],
                    "P": tr["P"],
                    "bbox": tr["bbox"],
                    "invisible": 0,
                    "age": 1,
                    "color": color,
                }
            )

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
            tr for tr in self.pending_tracks if tr["miss"] < 8
        ]
