"""
Utility helpers for drawing bounding boxes and trajectories on images.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import cv2
import numpy as np


def draw_bounding_boxes(
    image: np.ndarray,
    boxes: Iterable[Sequence[int]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Return a copy of `image` with rectangles rendered for each bounding box.

    Parameters
    ----------
    image : np.ndarray
        BGR frame.
    boxes : iterable of [x1, y1, x2, y2]
    color : tuple
        Box color in BGR.
    thickness : int
    """
    vis = image.copy()
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return vis


def draw_track_paths(
    image: np.ndarray,
    tracks: dict[int, list[Tuple[int, int]]],
    color_map=None,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw polyline trajectories for track histories.

    Parameters
    ----------
    image : np.ndarray
    tracks : dict track_id -> [(x, y), ...]
    color_map : callable or None
        If provided, called with `track_id` and should return a BGR tuple.
    thickness : int
    """
    vis = image.copy()
    for track_id, pts in tracks.items():
        if len(pts) < 2:
            continue
        color = (
            color_map(track_id)
            if color_map
            else (
                int((37 * track_id) % 255),
                int((17 * track_id + 80) % 255),
                int((29 * track_id + 160) % 255),
            )
        )
        cv2.polylines(vis, [np.array(pts, dtype=np.int32)], False, color, thickness)
    return vis


__all__ = ["draw_bounding_boxes", "draw_track_paths"]
