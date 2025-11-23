"""
optical_flow.py

Optical-flow-based motion detector for moving-camera sequences.

Pipeline:
    - Track sift feature points with Lucas–Kanade optical flow
    - Estimate global (background) motion using RANSAC homography
    - Subtract background motion → residual flow
    - Threshold large residual motion → foreground/moving points
    - Cluster moving points using DBSCAN
    - Generate bounding boxes per cluster

Dependencies:
    - OpenCV (cv2)
    - NumPy
    - scikit-learn (for DBSCAN)
"""

import cv2
import numpy as np

try:
    from sklearn.cluster import DBSCAN
    _HAS_SKLEARN = True
except ImportError:
    DBSCAN = None
    _HAS_SKLEARN = False
    print("[WARNING] scikit-learn not found. OpticalFlowDetector will not cluster points.")


class OpticalFlowDetector:
    def __init__(
        self,
        max_corners: int = 900,
        quality_level: float = 0.01,
        min_distance: float = 3.0,
        block_size: int = 7,
        lk_win_size=(21, 21),
        lk_max_level: int = 3,
        motion_threshold: float = 1.5,
        dbscan_eps: float = 20.0,
        dbscan_min_samples: int = 5,
        min_box_area: int = 100
    ):
        """
        Parameters
        ----------
        max_corners : int
            Maximum number of features to track.
        quality_level : float
            Parameter for cv2.goodFeaturesToTrack.
        min_distance : float
            Minimum distance between features.
        block_size : int
            Block size for Shi–Tomasi detector.
        lk_win_size : (int, int)
            Window size for Lucas–Kanade optical flow.
        lk_max_level : int
            Max pyramid level used in calcOpticalFlowPyrLK.
        motion_threshold : float
            Minimal residual flow magnitude (in pixels) to consider a point moving.
        dbscan_eps : float
            Radius parameter for DBSCAN clustering (in pixels).
        dbscan_min_samples : int
            Minimum samples per DBSCAN cluster.
        min_box_area : int
            Minimum area (in pixels) for bounding box to be kept.
        """
        self.prev_gray = None
        self.prev_points = None  # shape (N, 1, 2)

        self.feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=block_size
        )

        self.lk_params = dict(
            winSize=lk_win_size,
            maxLevel=lk_max_level,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )

        self.motion_threshold = motion_threshold
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.min_box_area = min_box_area

        # Re-detect features if tracking falls below this fraction
        self.redetect_fraction = 0.3

    # ------------------------------------------------------------------
    def _detect_features(self, gray):
        """
        Detect SIFT keypoints and convert them into a numpy (N, 1, 2) array
        suitable for cv2.calcOpticalFlowPyrLK.
        """
        # Create SIFT detector once and reuse it
        if not hasattr(self, "_sift"):
            self._sift = cv2.SIFT_create(
                nfeatures=3000,      # max keypoints
                contrastThreshold=0.02,
                edgeThreshold=10,
                sigma=1.6
            )

        # Detect SIFT keypoints
        keypoints = self._sift.detect(gray, mask=None)

        if not keypoints:
            return None

        # Convert keypoints to numpy array for LK optical flow
        pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)

        # reshape into (N, 1, 2) required by calcOpticalFlowPyrLK
        pts = pts.reshape(-1, 1, 2)

        return pts


    # ------------------------------------------------------------------
    def reset(self):
        """Reset internal state."""
        self.prev_gray = None
        self.prev_points = None

    # ------------------------------------------------------------------
    def detect(self, frame):
        """
        Run optical-flow-based motion detection on a single frame.

        Parameters
        ----------
        frame : np.ndarray (H, W, 3)
            Current RGB/BGR image from the left camera.

        Returns
        -------
        boxes : list of [x1, y1, x2, y2]
            Bounding boxes (int coordinates) of detected moving regions.
        """
        if frame is None:
            raise ValueError("Input frame is None")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ------------------------------------------------------------------
        # If first frame or we lost track, just detect features and return no boxes
        # ------------------------------------------------------------------
        if self.prev_gray is None or self.prev_points is None or len(self.prev_points) < 10:
            self.prev_gray = gray
            self.prev_points = self._detect_features(gray)
            return []

        # ------------------------------------------------------------------
        # Track previous features to current frame with Lucas–Kanade
        # ------------------------------------------------------------------
        p0 = self.prev_points  # (N, 1, 2)
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, p0, None, **self.lk_params
        )

        if p1 is None or st is None:
            # Tracking failed; re-detect and exit
            self.prev_gray = gray
            self.prev_points = self._detect_features(gray)
            return []

        st = st.reshape(-1)
        good_old = p0[st == 1]  # (M, 1, 2)
        good_new = p1[st == 1]  # (M, 1, 2)

        if len(good_old) < 10:
            # Too few points; re-detect
            self.prev_gray = gray
            self.prev_points = self._detect_features(gray)
            return []

        # ------------------------------------------------------------------
        # Estimate global (background) motion via homography with RANSAC
        # ------------------------------------------------------------------
        old_pts = good_old.reshape(-1, 2)
        new_pts = good_new.reshape(-1, 2)

        H, inliers = cv2.findHomography(
            old_pts, new_pts, cv2.RANSAC, 3.0
        )

        if H is None or inliers is None:
            # RANSAC failed – can't separate background/foreground reliably
            # Still update prev frame and points for next time
            self.prev_gray = gray
            self.prev_points = self._detect_features(gray)
            return []

        inliers = inliers.reshape(-1).astype(bool)

        # ------------------------------------------------------------------
        # Predict where old_pts should be under background motion
        # ------------------------------------------------------------------
        old_pts_h = cv2.perspectiveTransform(
            old_pts.reshape(-1, 1, 2), H
        ).reshape(-1, 2)

        # Residual flow = observed - background-predicted
        flow = new_pts - old_pts_h  # (M, 2)

        # Magnitude of residual flow
        mag = np.linalg.norm(flow, axis=1)

        # Foreground (moving) mask
        moving_mask = mag > self.motion_threshold
        moving_pts = new_pts[moving_mask]  # (K, 2)

        # ------------------------------------------------------------------
        # Optional: re-detect features if we are losing too many points
        # ------------------------------------------------------------------
        if len(good_new) < self.redetect_fraction * len(p0):
            # Merge current good_new and fresh detections to maintain coverage
            detected = self._detect_features(gray)
            if detected is not None:
                self.prev_points = np.concatenate([good_new, detected], axis=0)
            else:
                self.prev_points = good_new
        else:
            self.prev_points = good_new

        # Update previous gray frame for next call
        self.prev_gray = gray

        # No moving points detected → no boxes
        if moving_pts.shape[0] == 0:
            return []

        # ------------------------------------------------------------------
        # Cluster moving points into object blobs (DBSCAN)
        # ------------------------------------------------------------------
        if not _HAS_SKLEARN or DBSCAN is None:
            # Fallback: no clustering; just wrap all moving points in one box
            x_min = int(np.min(moving_pts[:, 0]))
            y_min = int(np.min(moving_pts[:, 1]))
            x_max = int(np.max(moving_pts[:, 0]))
            y_max = int(np.max(moving_pts[:, 1]))
            if (x_max - x_min) * (y_max - y_min) >= self.min_box_area:
                return [[x_min, y_min, x_max, y_max]]
            else:
                return []

        clustering = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples
        ).fit(moving_pts)

        labels = clustering.labels_  # -1 is noise
        unique_labels = set(labels)
        boxes = []

        for lab in unique_labels:
            if lab == -1:
                continue  # ignore noise

            cluster_pts = moving_pts[labels == lab]
            if cluster_pts.shape[0] < self.dbscan_min_samples:
                continue

            x_min = int(np.min(cluster_pts[:, 0]))
            y_min = int(np.min(cluster_pts[:, 1]))
            x_max = int(np.max(cluster_pts[:, 0]))
            y_max = int(np.max(cluster_pts[:, 1]))

            if (x_max - x_min) * (y_max - y_min) < self.min_box_area:
                continue

            boxes.append([x_min, y_min, x_max, y_max])

        return boxes


# ----------------------------------------------------------------------
# Example usage (for quick testing)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple test on a video or sequence of images
    import os

    # Example: reading from a video (replace with your SequenceLoader)
    cap = cv2.VideoCapture("example.mp4")
    detector = OpticalFlowDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes = detector.detect(frame)

        # Draw boxes
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Optical Flow Detection", frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
