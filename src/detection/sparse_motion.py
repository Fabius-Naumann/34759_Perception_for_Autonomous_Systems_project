"""
Sparse-motion detector that mirrors the bounding-box logic developed in
`notebooks/fromwp.ipynb`.

The detector relies on dense Shi–Tomasi features tracked with LK-optical
flow between consecutive frames, selects the strongest movers, clusters
them with simple morphology, and returns bounding boxes for the moving
regions near the road surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np


def non_max_suppression(boxes: Sequence[Sequence[int]], iou_threshold: float = 0.4) -> List[List[int]]:
    """
    Basic NMS over axis-aligned boxes.

    Parameters
    ----------
    boxes : Sequence[[x1, y1, x2, y2]]
    iou_threshold : float
        IoU threshold for suppressing overlapping boxes.
    """
    if not boxes:
        return []

    boxes_np = np.asarray(boxes, dtype=np.float32)
    x1 = boxes_np[:, 0]
    y1 = boxes_np[:, 1]
    x2 = boxes_np[:, 2]
    y2 = boxes_np[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(y2)
    keep = []

    while len(order) > 0:
        last = order[-1]
        keep.append(last)
        order = order[:-1]
        if len(order) == 0:
            break

        xx1 = np.maximum(x1[last], x1[order])
        yy1 = np.maximum(y1[last], y1[order])
        xx2 = np.minimum(x2[last], x2[order])
        yy2 = np.minimum(y2[last], y2[order])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        overlap = w * h
        iou = overlap / (areas[last] + areas[order] - overlap + 1e-6)

        order = order[np.where(iou <= iou_threshold)[0]]

    return [list(map(int, boxes_np[i])) for i in keep]


@dataclass
class SparseMotionParams:
    max_corners: int = 500
    quality_level: float = 0.2
    min_distance: float = 5.0
    lk_win_size: Tuple[int, int] = (21, 21)
    lk_max_level: int = 3
    circle_radius: int = 10
    road_ratio: float = 0.45
    top_k: int = 80
    min_area: int = 800
    nms_thresh: float = 0.35
    morph_kernel: int = 21


class SparseMotionDetector:
    """
    Stateful detector that consumes a video stream (left camera frames)
    and produces sparse motion bounding boxes per frame.
    """

    def __init__(self, params: SparseMotionParams | None = None):
        self.params = params or SparseMotionParams()
        self.prev_gray: np.ndarray | None = None
        self.kernel_sparse = np.ones(
            (self.params.morph_kernel, self.params.morph_kernel), dtype=np.uint8
        )
        self.feature_params = dict(
            maxCorners=self.params.max_corners,
            qualityLevel=self.params.quality_level,
            minDistance=self.params.min_distance,
            blockSize=7,
        )
        self.lk_params = dict(
            winSize=self.params.lk_win_size,
            maxLevel=self.params.lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        # Debug hooks for notebooks / visualization
        self.debug_points: np.ndarray | None = None
        self.debug_mask: np.ndarray | None = None

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.prev_gray = None
        self.debug_points = None
        self.debug_mask = None

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> List[List[int]]:
        """
        Process a BGR frame and return sparse-motion bounding boxes.
        """
        if frame is None:
            raise ValueError("Frame is None")

        gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray_curr
            return []

        road_start = int(gray_curr.shape[0] * self.params.road_ratio)
        boxes = self._compute_sparse_motion_boxes(self.prev_gray, gray_curr, road_start)
        self.prev_gray = gray_curr
        return boxes

    # ------------------------------------------------------------------
    def _compute_sparse_motion_boxes(
        self,
        gray_prev: np.ndarray,
        gray_curr: np.ndarray,
        road_start: int,
    ) -> List[List[int]]:
        feat_prev = cv2.goodFeaturesToTrack(gray_prev, mask=None, **self.feature_params)
        if feat_prev is None:
            self.debug_points = None
            self.debug_mask = None
            return []

        feat_curr, status, _ = cv2.calcOpticalFlowPyrLK(
            gray_prev,
            gray_curr,
            feat_prev,
            None,
            **self.lk_params,
        )

        if feat_curr is None or status is None:
            self.debug_points = None
            self.debug_mask = None
            return []

        status = status.reshape(-1).astype(bool)
        if not np.any(status):
            self.debug_points = None
            self.debug_mask = None
            return []

        feat_prev_valid = feat_prev[status]
        feat_curr_valid = feat_curr[status]
        displacements = np.linalg.norm(
            feat_curr_valid - feat_prev_valid, axis=2
        ).flatten()

        top_k = min(self.params.top_k, len(displacements))
        if top_k == 0:
            self.debug_points = None
            self.debug_mask = None
            return []

        sparse_mask = np.zeros_like(gray_curr, dtype=np.uint8)
        top_idx = np.argpartition(displacements, -top_k)[-top_k:]
        for idx in top_idx:
            x, y = feat_curr_valid[idx][0]
            if y < road_start:
                continue
            cv2.circle(
                sparse_mask,
                (int(x), int(y)),
                self.params.circle_radius,
                255,
                thickness=-1,
            )

        if not np.any(sparse_mask):
            self.debug_points = None
            self.debug_mask = sparse_mask
            return []

        sparse_mask = cv2.morphologyEx(
            sparse_mask, cv2.MORPH_CLOSE, self.kernel_sparse, iterations=1
        )
        sparse_mask = cv2.dilate(sparse_mask, self.kernel_sparse, iterations=1)
        contours_info = cv2.findContours(
            sparse_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

        boxes: List[List[int]] = []
        for cnt in contours:
            if cv2.contourArea(cnt) < self.params.min_area:
                continue
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            x1 = max(x, 0)
            y1 = max(y, road_start)
            x2 = min(x + w_box, gray_curr.shape[1] - 1)
            y2 = min(y + h_box, gray_curr.shape[0] - 1)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])

        self.debug_points = feat_curr_valid.reshape(-1, 2)
        self.debug_mask = sparse_mask
        return non_max_suppression(boxes, iou_threshold=self.params.nms_thresh)


__all__ = [
    "SparseMotionDetector",
    "SparseMotionParams",
    "non_max_suppression",
]
