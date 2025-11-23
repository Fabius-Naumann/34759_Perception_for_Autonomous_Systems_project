"""
stereo_sift_scene_flow.py

Stereo-based 3D motion detector using SIFT + triangulation.

Pipeline per frame:
    1) Detect SIFT keypoints in left & right images
    2) Match left-right with Lowe ratio → stereo correspondences
    3) Triangulate to get 3D landmarks for this frame
    4) Match SIFT descriptors between previous frame and current frame
    5) For matched landmarks, compute 3D displacement
    6) Threshold large 3D motion → moving points
    7) Cluster moving 2D points (in current left image) → 2D bounding boxes

Outputs per frame:
    - list of [x1, y1, x2, y2] bounding boxes in left image coordinates
"""

import numpy as np
import cv2

try:
    from sklearn.cluster import DBSCAN
    _HAS_SKLEARN = True
except ImportError:
    DBSCAN = None
    _HAS_SKLEARN = False
    print("[WARNING] scikit-learn not found. Falling back to single box clustering.")


# ---------------------------------------------------------------------------
# Helper: cluster 2D points into bounding boxes
# ---------------------------------------------------------------------------

def cluster_points_into_boxes(points_2d, eps=20.0, min_samples=5, min_box_area=150):
    """
    Cluster 2D points (Nx2) into boxes using DBSCAN.
    Returns list of [x1, y1, x2, y2].
    """
    if points_2d is None or len(points_2d) == 0:
        return []

    points_2d = np.asarray(points_2d, dtype=np.float32)

    if not _HAS_SKLEARN or DBSCAN is None:
        # Fallback: single bounding box around all moving points
        x_min = int(np.min(points_2d[:, 0]))
        y_min = int(np.min(points_2d[:, 1]))
        x_max = int(np.max(points_2d[:, 0]))
        y_max = int(np.max(points_2d[:, 1]))
        if (x_max - x_min) * (y_max - y_min) >= min_box_area:
            return [[x_min, y_min, x_max, y_max]]
        return []

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_2d)
    labels = clustering.labels_

    boxes = []
    for lab in set(labels):
        if lab == -1:
            continue
        pts = points_2d[labels == lab]
        if pts.shape[0] < min_samples:
            continue
        x_min = int(np.min(pts[:, 0]))
        y_min = int(np.min(pts[:, 1]))
        x_max = int(np.max(pts[:, 0]))
        y_max = int(np.max(pts[:, 1]))
        if (x_max - x_min) * (y_max - y_min) < min_box_area:
            continue
        boxes.append([x_min, y_min, x_max, y_max])

    return boxes


# ---------------------------------------------------------------------------
# Stereo SIFT: matches L-R and triangulates to 3D
# ---------------------------------------------------------------------------


class StereoSIFT:
    """
    Handles:
        - SIFT detection
        - left–right matching
        - triangulation using P_rect_02, P_rect_03
        - returning 3D keypoints + 2D left image coords + descriptors

    This class ONLY extracts stereo pairs for a single frame.
    Temporal matching is done in the SceneFlow detector.
    """

    def __init__(self, K_left, P_left, P_right):
        """
        Parameters
        ----------
        K_left : (3,3) np.ndarray
            Intrinsic matrix of left camera (K_02)
        P_left : (3,4) np.ndarray
            Rectified projection matrix P_rect_02
        P_right : (3,4) np.ndarray
            Rectified projection matrix P_rect_03
        """
        self.K_left = K_left.astype(np.float32)
        self.P_left = P_left.astype(np.float32)
        self.P_right = P_right.astype(np.float32)

        # SIFT + BF matcher
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # ----------------------------------------------------------------------
    def extract_3d_keypoints(self, imgL, imgR, ratio_thresh=0.75):
        """
        Step 1: Detect SIFT keypoints in left and right
        Step 2: Match L→R with Lowe ratio
        Step 3: Triangulate using P_rect_02 and P_rect_03

        Returns
        -------
        pts3D : (N,3) float32
        pts2D_left : (N,2) float32
        descriptors_left : (N,128) float32
        """

        # Convert to grayscale
        if imgL.ndim == 3:
            grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        else:
            grayL = imgL

        if imgR.ndim == 3:
            grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        else:
            grayR = imgR

        # Detect + compute descriptors
        kpL, desL = self.sift.detectAndCompute(grayL, None)
        kpR, desR = self.sift.detectAndCompute(grayR, None)

        if desL is None or desR is None:
            return None, None, None

        if len(kpL) < 10 or len(kpR) < 10:
            return None, None, None

        # KNN matches
        matches = self.matcher.knnMatch(desL, desR, k=2)

        ptsL = []
        ptsR = []
        desc = []

        # Lowe ratio test
        for m, n in matches:
            if m.distance < ratio_thresh * n.distance:
                ptsL.append(kpL[m.queryIdx].pt)
                ptsR.append(kpR[m.trainIdx].pt)
                desc.append(desL[m.queryIdx])

        if len(ptsL) < 10:
            return None, None, None

        ptsL = np.array(ptsL, dtype=np.float32)
        ptsR = np.array(ptsR, dtype=np.float32)
        desc = np.array(desc, dtype=np.float32)

        # Prepare for triangulation
        # Convert to homogeneous pixel coords
        ptsL_h = np.vstack([ptsL.T, np.ones((1, ptsL.shape[0]))])
        ptsR_h = np.vstack([ptsR.T, np.ones((1, ptsR.shape[0]))])

        # TRIANGULATE using KITTI rectified projection matrices
        # This is the correct way:
        #    P_left and P_right already include K, R_rect, and T
        P_h = cv2.triangulatePoints(
            self.P_left,
            self.P_right,
            ptsL_h[:2],
            ptsR_h[:2],
        )

        # Convert homogeneous → Euclidean
        P_h /= P_h[3]
        pts3D = P_h[:3].T  # shape (N,3)


        return pts3D.astype(np.float32), ptsL, desc.astype(np.float32)


class StereoSceneFlowDetector:
    """
    3D motion-based detector using stereo SIFT triangulation.

    Usage:
        sift_stereo = StereoSIFT(K, baseline)
        detector = StereoSceneFlowDetector(sift_stereo)

        boxes = detector.detect(left_t, right_t)  # list of [x1,y1,x2,y2]
    """

    def __init__(
        self,
        stereo_sift: StereoSIFT,
        motion_threshold: float = 0.3,
        dbscan_eps: float = 20.0,
        dbscan_min_samples: int = 5,
        min_box_area: int = 150,
    ):
        """
        Parameters
        ----------
        stereo_sift : StereoSIFT
            Instance of StereoSIFT to handle per-frame 3D feature extraction.
        motion_threshold : float
            Minimal 3D displacement (in meters) to consider a point moving.
        dbscan_eps : float
            DBSCAN eps parameter for clustering 2D moving points (in pixels).
        dbscan_min_samples : int
            DBSCAN min_samples parameter.
        min_box_area : int
            Minimum bounding box area (in pixels^2) to keep.
        """
        self.sift_stereo = stereo_sift
        self.motion_threshold = motion_threshold
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.min_box_area = min_box_area

        # Previous frame's 3D points, 2D points (left), and descriptors
        self.prev_pts3D = None   # (N,3)
        self.prev_pts2D = None   # (N,2)
        self.prev_desc = None    # (N,128)

        # for debugging / visualization
        self.debug_moving_points_2d = None
        self.debug_all_matches_2d = None

        self.temporal_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # -------------------------------------------------------------------
    def reset(self):
        """Reset internal state (e.g., at new sequence)."""
        self.prev_pts3D = None
        self.prev_pts2D = None
        self.prev_desc = None
        self.debug_moving_points_2d = None
        self.debug_all_matches_2d = None

    # -------------------------------------------------------------------
    def detect(self, imgL, imgR):
        """
        Detect moving objects between last frame and current frame using:
        - stereo SIFT
        - temporal matching
        - 3D motion magnitude
        - 3D motion consistency (direction clustering)
        - depth gating
        """

        # ------------------------------------------------------------
        # 1) Extract stereo 3D keypoints for current frame
        # ------------------------------------------------------------
        pts3D_cur, pts2D_cur, desc_cur = self.sift_stereo.extract_3d_keypoints(imgL, imgR)

        if (pts3D_cur is None or pts2D_cur is None or desc_cur is None or
                pts3D_cur.shape[0] < 10):
            self.prev_pts3D = pts3D_cur
            self.prev_pts2D = pts2D_cur
            self.prev_desc = desc_cur
            self.debug_moving_points_2d = None
            self.debug_all_matches_2d = None
            return []

        if self.prev_pts3D is None or self.prev_desc is None:
            self.prev_pts3D = pts3D_cur
            self.prev_pts2D = pts2D_cur
            self.prev_desc = desc_cur
            self.debug_moving_points_2d = None
            self.debug_all_matches_2d = None
            return []

        # ------------------------------------------------------------
        # 2) Temporal matching via SIFT descriptors
        # ------------------------------------------------------------
        matches = self.temporal_matcher.knnMatch(self.prev_desc, desc_cur, k=2)

        idx_prev = []
        idx_cur = []
        matched_pts2D_cur = []

        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                idx_prev.append(m.queryIdx)
                idx_cur.append(m.trainIdx)
                matched_pts2D_cur.append(pts2D_cur[m.trainIdx])

        if len(idx_prev) < 10:
            self.prev_pts3D = pts3D_cur
            self.prev_pts2D = pts2D_cur
            self.prev_desc = desc_cur
            self.debug_moving_points_2d = None
            self.debug_all_matches_2d = None
            return []

        idx_prev = np.array(idx_prev, dtype=np.int32)
        idx_cur = np.array(idx_cur, dtype=np.int32)
        matched_pts2D_cur = np.array(matched_pts2D_cur, dtype=np.float32)

        pts3D_prev_matched = self.prev_pts3D[idx_prev]
        pts3D_cur_matched = pts3D_cur[idx_cur]

        # ------------------------------------------------------------
        # 3) Compute 3D motion vectors + magnitudes
        # ------------------------------------------------------------
        motion_vec = pts3D_cur_matched - pts3D_prev_matched
        disp3D = np.linalg.norm(motion_vec, axis=1)

        # Normalize motion vector for direction clustering
        norms = np.linalg.norm(motion_vec, axis=1, keepdims=True) + 1e-6
        motion_dir = motion_vec / norms

        # ------------------------------------------------------------
        # 4) Depth gating (remove far-away background like trees)
        # ------------------------------------------------------------
        Z_cur = pts3D_cur_matched[:, 2]
        depth_mask = (Z_cur > 1.0) & (Z_cur < 25.0)

        # ------------------------------------------------------------
        # 5) Magnitude gating (remove small fluttering leaf motion)
        # ------------------------------------------------------------
        mag_mask = disp3D > 0.05   # 5cm minimum movement

        # Combine masks
        valid_mask = depth_mask & mag_mask

        if np.sum(valid_mask) < 5:
            # Not enough valid moving points
            self.prev_pts3D = pts3D_cur
            self.prev_pts2D = pts2D_cur
            self.prev_desc = desc_cur
            return []

        motion_dir_valid = motion_dir[valid_mask]
        pts2D_valid = matched_pts2D_cur[valid_mask]

        # ------------------------------------------------------------
        # 6) OPTIONAL: restrict to lower region of image (ignore trees)
        # ------------------------------------------------------------
        h, w = imgL.shape[:2]
        region_mask = pts2D_valid[:, 1] > (h * 0.40)   # keep bottom 60%
        motion_dir_valid = motion_dir_valid[region_mask]
        pts2D_valid = pts2D_valid[region_mask]

        if motion_dir_valid.shape[0] < 5:
            self.prev_pts3D = pts3D_cur
            self.prev_pts2D = pts2D_cur
            self.prev_desc = desc_cur
            return []

        # ------------------------------------------------------------
        # 7) Cluster by motion direction consistency (key trick!)
        # ------------------------------------------------------------
        clustering = DBSCAN(eps=0.3, min_samples=5).fit(motion_dir_valid)
        labels = clustering.labels_

        # choose the largest non-noise cluster
        best_cluster = None
        best_size = 0

        for lab in set(labels):
            if lab == -1:
                continue
            size = np.sum(labels == lab)
            if size > best_size:
                best_size = size
                best_cluster = lab

        if best_cluster is None:
            self.prev_pts3D = pts3D_cur
            self.prev_pts2D = pts2D_cur
            self.prev_desc = desc_cur
            return []

        moving_pts2D = pts2D_valid[labels == best_cluster]

        # Save debug visualizations
        self.debug_all_matches_2d = matched_pts2D_cur
        self.debug_moving_points_2d = moving_pts2D

        # ------------------------------------------------------------
        # 8) Convert 2D moving points to initial bounding boxes
        # ------------------------------------------------------------
        raw_boxes = cluster_points_into_boxes(
            moving_pts2D,
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            min_box_area=self.min_box_area,
        )

        refined_boxes = []
        h, w = imgL.shape[:2]

        # Get motion vector cluster for shape adjustment
        cluster_motion = motion_vec[valid_mask][region_mask][labels == best_cluster]
        mean_motion = np.mean(cluster_motion, axis=0)  # (dx, dy, dz)

        for (x1, y1, x2, y2) in raw_boxes:

            box_w = x2 - x1
            box_h = y2 - y1

            # --- FIX 1: base padding ---
            pad_w = int(0.3 * box_w)
            pad_h = int(0.4 * box_h)

            # --- FIX 2: motion-dependent shaping ---
            if abs(mean_motion[0]) > abs(mean_motion[2]):
                # horizontal motion → widen box more
                pad_w = int(pad_w * 1.6)
            else:
                # depth motion → taller box
                pad_h = int(pad_h * 1.6)

            # --- FIX 3: height from stereo depth ---
            Z_center = np.median(pts3D_cur_matched[:, 2])
            expected_h = int((self.sift_stereo.K_left[1,1] * 1.7) / Z_center)
            if expected_h > box_h:
                missing_h = expected_h - box_h
                pad_h = max(pad_h, missing_h // 2)

            # --- Apply padding ---
            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(w - 1, x2 + pad_w)
            y2 = min(h - 1, y2 + pad_h)

            refined_boxes.append([x1, y1, x2, y2])

        # ------------------------------------------------------------
        # 9) Update state and return refined boxes
        # ------------------------------------------------------------
        self.prev_pts3D = pts3D_cur
        self.prev_pts2D = pts2D_cur
        self.prev_desc = desc_cur

        return refined_boxes




# ---------------------------------------------------------------------------
# Example usage (you can delete or adapt this in your project)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy K and baseline — replace with real calibration
    fx = 700.0
    fy = 700.0
    cx = 600.0
    cy = 180.0
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float32)
    baseline = 0.54  # meters, typical KITTI value

    # Example: use video or your SequenceLoader providing left/right frames
    capL = cv2.VideoCapture("left_video.mp4")
    capR = cv2.VideoCapture("right_video.mp4")

    stereo_sift = StereoSIFT(K, baseline)
    detector = StereoSceneFlowDetector(stereo_sift)

    while True:
        retL, frameL = capL.read()
        retR, frameR = capR.read()
        if not retL or not retR:
            break

        boxes = detector.detect(frameL, frameR)

        vis = frameL.copy()
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Stereo 3D Motion Detection", vis)
        if cv2.waitKey(1) == 27:
            break

    capL.release()
    capR.release()
    cv2.destroyAllWindows()
