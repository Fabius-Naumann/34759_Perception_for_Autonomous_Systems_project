"""
Chessboard detection utilities for camera calibration.

Provides functions for detecting multiple chessboards in calibration images,
with support for parallel processing and caching.
"""

import hashlib
import multiprocessing as mp
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:

    def tqdm(it, **_kwargs):
        return it


def find_all_chessboards(
    gray: np.ndarray,
    pattern_sizes: list[tuple[int, int]],
    max_boards: int = 10,
    subpixel_window_scale: float = 0.25,
    subpixel_iterations: int = 30,
    subpixel_epsilon: float = 0.001,
) -> tuple[list[np.ndarray], list[np.ndarray], list[tuple[int, int]]]:
    """
    Detect multiple chessboards of different sizes in a single image.

    This function iteratively detects chessboards, masks out found boards,
    and searches for additional boards until no more are found or max_boards
    is reached.

    Args:
        gray: Grayscale input image
        pattern_sizes: List of (nb_vertical, nb_horizontal) inner corner counts
        max_boards: Maximum number of boards to detect per pattern size
        subpixel_window_scale: Window size for cornerSubPix as fraction of square size
        subpixel_iterations: Max iterations for cornerSubPix
        subpixel_epsilon: Epsilon for cornerSubPix termination

    Returns:
        Tuple of (corners_list, objpoints_list, pattern_sizes_list) where:
        - corners_list: List of detected corner arrays (Nx1x2 float32)
        - objpoints_list: List of corresponding 3D object points (Nx3 float32)
        - pattern_sizes_list: List of pattern sizes for each detection
    """
    all_corners = []
    all_objp = []
    all_pattern_sizes = []
    work = gray.copy()

    flags_normal = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    flags_SB = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY
    avg_color = np.average(gray)

    for nb_vert, nb_horiz in pattern_sizes:
        for _ in range(max_boards):
            # Try SB variant first (subpixel-accurate)
            ret, corners = cv2.findChessboardCornersSB(work, (nb_vert, nb_horiz), flags=flags_SB)
            use_sb = True
            if not ret:
                # Fall back to regular detector
                ret, corners = cv2.findChessboardCorners(work, (nb_vert, nb_horiz), flags=flags_normal)
                use_sb = False

            if not ret:
                break

            # Prepare object points: (x, y, 0) grid
            objp = np.zeros((nb_horiz * nb_vert, 3), np.float32)
            objp[:, :2] = np.mgrid[0:nb_vert, 0:nb_horiz].T.reshape(-1, 2)
            all_objp.append(objp)
            all_pattern_sizes.append((nb_vert, nb_horiz))

            # Subpixel refinement (only if not using SB)
            if use_sb:
                corners_ref = corners.astype(np.float32, copy=False)
            else:
                corners = corners.astype(np.float32, copy=False)
                # Determine adaptive window from detected board bbox
                x, y, w, h = cv2.boundingRect(corners)
                sq_w = max(1.0, w / max(1, (nb_vert - 1)))
                sq_h = max(1.0, h / max(1, (nb_horiz - 1)))
                auto_win = int(max(3, min(15, subpixel_window_scale * min(sq_w, sq_h))))
                win = (auto_win, auto_win)
                term = (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    subpixel_iterations,
                    subpixel_epsilon,
                )
                corners = cv2.cornerSubPix(gray, corners, win, (-1, -1), term)
                corners_ref = corners

            all_corners.append(corners_ref)

            # Mask out detected board region to find next board
            x, y, w, h = cv2.boundingRect(corners_ref)
            pad = 10
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(work.shape[1], x + w + pad), min(work.shape[0], y + h + pad)
            cv2.rectangle(work, (x0, y0), (x1, y1), color=avg_color, thickness=-1)

    return all_corners, all_objp, all_pattern_sizes


def _detect_single_image(args: tuple[Any, ...]) -> dict[str, Any]:
    """
    Worker function for parallel chessboard detection.

    Args:
        args: Tuple of (image_path, img_idx, pattern_sizes, max_boards, subpixel_params)

    Returns:
        Dictionary with detection results
    """
    image_path, img_idx, pattern_sizes, max_boards, subpixel_params = args

    img = cv2.imread(str(image_path))
    if img is None:
        return {"img_idx": img_idx, "success": False, "corners": [], "objpoints": [], "sizes": []}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, objps, sizes = find_all_chessboards(gray, pattern_sizes, max_boards, **subpixel_params)

    return {
        "img_idx": img_idx,
        "success": True,
        "corners": corners,
        "objpoints": objps,
        "sizes": sizes,
        "image_size": (gray.shape[1], gray.shape[0]),
    }


def detect_chessboards_parallel(
    image_paths: list[str | Path],
    pattern_sizes: list[tuple[int, int]],
    max_boards: int = 10,
    num_workers: int = -1,
    progress: bool = True,
    subpixel_window_scale: float = 0.25,
    subpixel_iterations: int = 30,
    subpixel_epsilon: float = 0.001,
    cache_dir: Path | None = None,
) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]], list[list[int]], tuple[int, int] | None]:
    """
    Detect chessboards in multiple images using parallel processing with optional caching.

    Args:
        image_paths: List of paths to calibration images
        pattern_sizes: List of (nb_vertical, nb_horizontal) patterns to search
        max_boards: Maximum boards per image per pattern size
        num_workers: Number of parallel workers (-1 = all cores, 1 = serial)
        progress: Show progress bar
        subpixel_window_scale: Window scale for cornerSubPix
        subpixel_iterations: Max iterations for subpixel refinement
        subpixel_epsilon: Epsilon for subpixel termination
        cache_dir: Optional directory for caching detection results

    Returns:
        Tuple of (all_corners, all_objpoints, all_image_indices, image_size):
        - all_corners: Nested list - corners[board_idx] = corner array
        - all_objpoints: Nested list - objpoints[board_idx] = object point array
        - all_image_indices: Nested list - image_indices[board_idx] = source image index
        - image_size: (width, height) from first valid image, or None
    """
    # Check cache
    if cache_dir is not None:
        from .cache import load_from_cache
        cache_key = _compute_detection_cache_key(image_paths, pattern_sizes, max_boards)
        cached = load_from_cache(cache_dir, cache_key, "detection", progress=progress)
        if cached is not None:
            return cached["corners"], cached["objpoints"], cached["indices"], cached["image_size"]

    subpixel_params = {
        "subpixel_window_scale": subpixel_window_scale,
        "subpixel_iterations": subpixel_iterations,
        "subpixel_epsilon": subpixel_epsilon,
    }

    # Prepare arguments for workers
    args_list = [(path, idx, pattern_sizes, max_boards, subpixel_params) for idx, path in enumerate(image_paths)]

    if num_workers == -1:
        num_workers = mp.cpu_count()

    all_corners = []
    all_objpoints = []
    all_image_indices = []
    image_size = None

    if num_workers == 1:
        # Serial processing
        iterator = tqdm(args_list, desc="Detecting boards") if progress else args_list
        results = [_detect_single_image(args) for args in iterator]
    else:
        # Parallel processing
        with mp.Pool(num_workers) as pool:
            iterator = pool.imap(_detect_single_image, args_list)
            if progress:
                iterator = tqdm(iterator, total=len(args_list), desc="Detecting boards")
            results = list(iterator)

    # Aggregate results
    for result in results:
        if result["success"]:
            if image_size is None:
                image_size = result["image_size"]

            img_idx = result["img_idx"]
            for corners, objp in zip(result["corners"], result["objpoints"], strict=False):
                all_corners.append(corners)
                all_objpoints.append(objp)
                all_image_indices.append(img_idx)

    # Save to cache
    if cache_dir is not None:
        from .cache import save_to_cache
        cache_data = {
            "corners": all_corners,
            "objpoints": all_objpoints,
            "indices": all_image_indices,
            "image_size": image_size,
        }
        save_to_cache(cache_dir, cache_key, "detection", cache_data, progress=progress)

    return all_corners, all_objpoints, all_image_indices, image_size


def _compute_detection_cache_key(
    image_paths: list[str | Path], pattern_sizes: list[tuple[int, int]], max_boards: int
) -> str:
    """Compute cache key for detection results based on image paths and parameters."""
    h = hashlib.md5()
    for path in sorted(str(p) for p in image_paths):
        h.update(str(path).encode())
        # Include file modification time if available
        try:
            mtime = Path(path).stat().st_mtime
            h.update(str(mtime).encode())
        except Exception:
            pass
    h.update(str(sorted(pattern_sizes)).encode())
    h.update(str(max_boards).encode())
    return h.hexdigest()


def filter_detections_by_image_count(
    corners: list[np.ndarray],
    objpoints: list[np.ndarray],
    image_indices: list[int],
    min_boards_per_image: int = 1,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """
    Filter out boards from images with too few detections.

    Args:
        corners: List of corner arrays
        objpoints: List of object point arrays
        image_indices: List of source image indices
        min_boards_per_image: Minimum boards required per image

    Returns:
        Filtered (corners, objpoints, image_indices)
    """
    from collections import Counter

    img_counts = Counter(image_indices)
    valid_images = {img_idx for img_idx, count in img_counts.items() if count >= min_boards_per_image}

    filtered_corners = []
    filtered_objpoints = []
    filtered_indices = []

    for c, o, idx in zip(corners, objpoints, image_indices, strict=False):
        if idx in valid_images:
            filtered_corners.append(c)
            filtered_objpoints.append(o)
            filtered_indices.append(idx)

    return filtered_corners, filtered_objpoints, filtered_indices
