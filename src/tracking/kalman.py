import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_xyxy(boxA, boxB):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


def xyxy_to_cxcywh(box):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return np.array([cx, cy, w, h], dtype=np.float32)


def cxcywh_to_xyxy(box):
    cx, cy, w, h = box
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return np.array([x1, y1, x2, y2], dtype=np.float32)


class KalmanBoxTracker:
    """
    Simple Kalman filter-based tracker for a single bounding box.
    State: [cx, cy, w, h, vx, vy]
    Measurement: [cx, cy, w, h]
    """

    count = 0

    def __init__(self, bbox_xyxy):
        # Unique track ID
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        # Time since last update, total hits, etc.
        self.time_since_update = 0
        self.hits = 1
        self.age = 0

        # State vector [cx, cy, w, h, vx, vy]
        cx, cy, w, h = xyxy_to_cxcywh(bbox_xyxy)
        self.x = np.array([[cx], [cy], [w], [h], [0.0], [0.0]], dtype=np.float32)

        # State transition (constant velocity on center)
        dt = 1.0
        self.F = np.eye(6, dtype=np.float32)
        self.F[0, 4] = dt   # cx += vx*dt
        self.F[1, 5] = dt   # cy += vy*dt

        # Measurement matrix: we observe [cx, cy, w, h]
        self.H = np.zeros((4, 6), dtype=np.float32)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        # Covariances
        self.P = np.eye(6, dtype=np.float32) * 10.0       # state covariance
        self.Q = np.eye(6, dtype=np.float32) * 0.01       # process noise
        self.R = np.eye(4, dtype=np.float32) * 1.0        # measurement noise

    def predict(self):
        """Predict next state."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.age += 1
        self.time_since_update += 1

        return self.get_state()

    def update(self, bbox_xyxy):
        """Measurement update with a new bounding box."""
        cx, cy, w, h = xyxy_to_cxcywh(bbox_xyxy)
        z = np.array([[cx], [cy], [w], [h]], dtype=np.float32)

        # Innovation
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0], dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

        self.time_since_update = 0
        self.hits += 1

    def get_state(self):
        """Return current state as [x1, y1, x2, y2]."""
        cx, cy, w, h, vx, vy = self.x.flatten()
        return cxcywh_to_xyxy([cx, cy, w, h])


class MultiObjectTracker:
    """
    Simple multi-object tracker using KalmanBoxTracker instances and IoU-based
    data association (Hungarian algorithm).
    """

    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []  # list of KalmanBoxTracker

    def update(self, detections):
        """
        Update the tracker with current frame detections.

        Parameters
        ----------
        detections : list of [x1,y1,x2,y2]

        Returns
        -------
        tracks : list of dicts
            Each dict contains:
                - 'id': track ID
                - 'bbox': [x1,y1,x2,y2]
                - 'age': total age
                - 'hits': number of successful updates
        """
        # Step 1: predict all trackers forward
        predicted_boxes = []
        for t in self.trackers:
            predicted_boxes.append(t.predict())

        # Step 2: data association
        if len(self.trackers) == 0 or len(detections) == 0:
            # No existing tracks or no detections
            matches = np.empty((0, 2), dtype=int)
            unmatched_trks = np.arange(len(self.trackers))
            unmatched_dets = np.arange(len(detections))
        else:
            # Build IoU cost matrix
            cost = np.zeros((len(self.trackers), len(detections)), dtype=np.float32)
            for i, tb in enumerate(predicted_boxes):
                for j, d in enumerate(detections):
                    cost[i, j] = 1.0 - iou_xyxy(tb, d)

            row_ind, col_ind = linear_sum_assignment(cost)
            matches = []
            unmatched_trks = list(range(len(self.trackers)))
            unmatched_dets = list(range(len(detections)))

            for r, c in zip(row_ind, col_ind):
                if cost[r, c] > (1.0 - self.iou_threshold):
                    # Too much cost → no match
                    continue
                matches.append([r, c])
                unmatched_trks.remove(r)
                unmatched_dets.remove(c)

            matches = np.array(matches, dtype=int)

        # Step 3: update matched trackers with assigned detections
        for (trk_idx, det_idx) in matches:
            self.trackers[trk_idx].update(detections[det_idx])

        # Step 4: create new trackers for unmatched detections
        for det_idx in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(detections[det_idx]))

        # Step 5: remove old trackers that haven't been updated for too long
        alive_trackers = []
        for t in self.trackers:
            if t.time_since_update <= self.max_age:
                alive_trackers.append(t)
        self.trackers = alive_trackers

        # Step 6: prepare track outputs
        output_tracks = []
        for t in self.trackers:
            bbox = t.get_state().tolist()
            if t.hits >= self.min_hits:
                output_tracks.append({
                    "id": t.id,
                    "bbox": bbox,
                    "age": t.age,
                    "hits": t.hits,
                })

        return output_tracks
