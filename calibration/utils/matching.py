"""
Stereo chessboard matching utilities.

Provides algorithms for matching chessboard detections between left and right
camera images in a stereo rig.
"""

import hashlib
from bisect import bisect_left, bisect_right
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def _centroid_and_bbox(corners: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Compute centroid and bounding box dimensions of corner points.

    Args:
        corners: Corner array (Nx1x2 or Nx2)

    Returns:
        Tuple of (centroid, (width, height))
    """
    pts = corners.reshape(-1, 2)
    c = pts.mean(axis=0)  # (x, y)
    _, _, w, h = cv2.boundingRect(corners.astype(np.float32))
    return c, (w, h)


def match_chessboards(
    dets_left: list[tuple[np.ndarray, tuple[int, int]]],
    dets_right: list[tuple[np.ndarray, tuple[int, int]]],
    dy_thresh: float = 12.0,
    scale_ratio_thresh: float = 0.25,
    method: str = "greedy",
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Match chessboard detections between left and right images.

    Uses vertical (epipolar) constraint and scale consistency to match boards
    that correspond to the same physical chessboard seen by both cameras.

    Args:
        dets_left: List of (corners, (nb_vert, nb_horiz)) for left image
        dets_right: List of (corners, (nb_vert, nb_horiz)) for right image
        dy_thresh: Vertical gating threshold in pixels
        scale_ratio_thresh: Relative height difference threshold (0-1)
        method: 'greedy' (fast) or 'hungarian' (optimal assignment)

    Returns:
        Tuple of (matches, unmatched_left, unmatched_right):
        - matches: List of (left_idx, right_idx) pairs
        - unmatched_left: List of unmatched left detection indices
        - unmatched_right: List of unmatched right detection indices

    Raises:
        ValueError: If method is not 'greedy' or 'hungarian'
    """
    if method not in {"greedy", "hungarian"}:
        raise ValueError("method must be 'greedy' or 'hungarian'")

    # Group by pattern size
    left_groups = defaultdict(list)
    right_groups = defaultdict(list)
    for i, (_, sz) in enumerate(dets_left):
        left_groups[tuple(sz)].append(i)
    for j, (_, sz) in enumerate(dets_right):
        right_groups[tuple(sz)].append(j)

    all_sizes = set(left_groups.keys()) | set(right_groups.keys())

    matches = []
    unmatched_left = []
    unmatched_right = []

    for sz in all_sizes:
        L = left_groups.get(sz, [])
        R = right_groups.get(sz, [])
        if not L:
            unmatched_right.extend(R)
            continue
        if not R:
            unmatched_left.extend(L)
            continue

        # Precompute features
        L_feats = {}
        for i in L:
            c, (w, h) = _centroid_and_bbox(dets_left[i][0])
            L_feats[i] = (c, (w, h))
        R_feats = {}
        for j in R:
            c, (w, h) = _centroid_and_bbox(dets_right[j][0])
            R_feats[j] = (c, (w, h))

        # Sort both sides by vertical coordinate
        L_sorted = sorted(L, key=lambda k: L_feats[k][0][1])
        R_sorted = sorted(R, key=lambda k: R_feats[k][0][1])
        R_y_vals = [R_feats[j][0][1] for j in R_sorted]

        if method == "greedy":
            used_R = set()
            for i in L_sorted:
                cL, (wL, hL) = L_feats[i]
                yL = float(cL[1])
                # Binary search for vertical band
                lo = bisect_left(R_y_vals, yL - dy_thresh)
                hi = bisect_right(R_y_vals, yL + dy_thresh)
                if lo == hi:
                    unmatched_left.append(i)
                    continue

                best_j = None
                best_cost = float("inf")
                for pos in range(lo, hi):
                    j = R_sorted[pos]
                    if j in used_R:
                        continue
                    cR, (wR, hR) = R_feats[j]
                    dy = abs(yL - float(cR[1]))
                    dx = abs(float(cL[0]) - float(cR[0]))

                    # Scale similarity check
                    h_max = max(1.0, float(max(hL, hR)))
                    scale_rel = abs(float(hL) - float(hR)) / h_max
                    if scale_rel > scale_ratio_thresh:
                        continue

                    # Cost: prioritize vertical alignment
                    cost = dy + 0.3 * dx + 0.25 * abs(float(hL) - float(hR))
                    if cost < best_cost:
                        best_cost = cost
                        best_j = j

                if best_j is not None:
                    matches.append((i, best_j))
                    used_R.add(best_j)
                else:
                    unmatched_left.append(i)
            unmatched_right.extend([j for j in R if j not in used_R])

        else:  # hungarian
            from scipy.optimize import linear_sum_assignment

            nL, nR = len(L_sorted), len(R_sorted)
            # Build cost matrix
            C = np.full((nL, nR), fill_value=1e6, dtype=np.float64)
            for a, i in enumerate(L_sorted):
                cL, (wL, hL) = L_feats[i]
                yL = float(cL[1])
                xL = float(cL[0])
                for b, j in enumerate(R_sorted):
                    cR, (wR, hR) = R_feats[j]
                    yR = float(cR[1])
                    xR = float(cR[0])
                    dy = abs(yL - yR)
                    if dy > dy_thresh:
                        continue
                    dx = abs(xL - xR)
                    h_max = max(1.0, float(max(hL, hR)))
                    scale_rel = abs(float(hL) - float(hR)) / h_max
                    if scale_rel > scale_ratio_thresh:
                        continue
                    cost = dy + 0.3 * dx + 0.25 * abs(float(hL) - float(hR))
                    C[a, b] = cost
            row_ind, col_ind = linear_sum_assignment(C)
            used_R = set()
            for a, b in zip(row_ind, col_ind, strict=False):
                if C[a, b] >= 1e6:
                    continue
                i = L_sorted[a]
                j = R_sorted[b]
                matches.append((i, j))
                used_R.add(j)
            unmatched_left.extend([i for i in L if i not in {m[0] for m in matches}])
            unmatched_right.extend([j for j in R if j not in used_R])

    return matches, unmatched_left, unmatched_right


def build_stereo_dataset(
    images_left: list[str],
    images_right: list[str],
    pattern_sizes: list[tuple[int, int]],
    square_size: float,
    detection_func: callable,
    matching_params: dict | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """
    Build stereo calibration dataset from image pairs.

    Detects chessboards in both cameras and matches them to build
    corresponding object/image point sets for stereo calibration.

    Args:
        images_left: List of left image paths
        images_right: List of right image paths
        pattern_sizes: List of (nb_vert, nb_horiz) patterns
        square_size: Physical size of chessboard squares (meters)
        detection_func: Function to detect chessboards (e.g., find_all_chessboards)
        matching_params: Parameters for match_chessboards (optional)

    Returns:
        Tuple of (objpoints, imgpoints_left, imgpoints_right)
    """
    if matching_params is None:
        matching_params = {}

    objpoints = []
    imgpoints_left = []
    imgpoints_right = []

    pair_count = min(len(images_left), len(images_right))

    for k in range(pair_count):
        imgL = cv2.imread(images_left[k])
        imgR = cv2.imread(images_right[k])
        if imgL is None or imgR is None:
            continue

        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        # Detect boards
        corners_L, _objps_L, sizes_L = detection_func(grayL, pattern_sizes)
        corners_R, _objps_R, sizes_R = detection_func(grayR, pattern_sizes)

        dets_L = list(zip(corners_L, sizes_L, strict=False))
        dets_R = list(zip(corners_R, sizes_R, strict=False))

        # Match boards
        matches, _, _ = match_chessboards(dets_L, dets_R, **matching_params)

        # Build stereo pairs
        for iL, iR in matches:
            cornersL, sizeL = dets_L[iL]
            cornersR, sizeR = dets_R[iR]

            # Validate
            if tuple(sizeL) != tuple(sizeR):
                continue
            nb_vert, nb_horiz = sizeL
            if cornersL.shape[0] != nb_vert * nb_horiz:
                continue
            if cornersR.shape[0] != nb_vert * nb_horiz:
                continue

            # Create scaled object points
            objp = np.zeros((nb_horiz * nb_vert, 3), np.float32)
            objp[:, :2] = np.mgrid[0:nb_vert, 0:nb_horiz].T.reshape(-1, 2)
            objp *= float(square_size)

            objpoints.append(objp)
            imgpoints_left.append(cornersL.astype(np.float32))
            imgpoints_right.append(cornersR.astype(np.float32))

    return objpoints, imgpoints_left, imgpoints_right


def group_detections_by_image(
    corners: list[np.ndarray],
    objpoints: list[np.ndarray],
    indices: list[int],
) -> dict[int, list[tuple[np.ndarray, tuple[int, int]]]]:
    """
    Group flat detection lists by image index.

    Args:
        corners: List of corner arrays
        objpoints: List of object point arrays
        indices: List of image indices

    Returns:
        Dictionary mapping image index to list of (corners, pattern_size) tuples
    """
    grouped = defaultdict(list)
    for c, obj, idx in zip(corners, objpoints, indices, strict=False):
        # Infer pattern size from objpoints
        n_corners = obj.shape[0]
        # Find divisors to get pattern size (simple heuristic matching implementation)
        for nb_vert in range(2, 20):
            if n_corners % nb_vert == 0:
                nb_horiz = n_corners // nb_vert
                grouped[idx].append((c, (nb_vert, nb_horiz)))
                break
    return grouped


def build_stereo_pairs_from_detections(
    corners_left: list[np.ndarray],
    objpoints_left: list[np.ndarray],
    indices_left: list[int],
    corners_right: list[np.ndarray],
    objpoints_right: list[np.ndarray],
    indices_right: list[int],
    dy_thresh: float = 12.0,
    scale_ratio_thresh: float = 0.25,
    method: str = "hungarian",
    cache_dir: Path | None = None,
    progress: bool = True,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], dict]:
    """
    Build stereo pairs from already-detected chessboards by matching boards across image pairs.

    This function uses the pre-detected chessboard corners and matches them between
    left and right camera images to create stereo calibration pairs.

    Args:
        corners_left: List of corner arrays from left camera detection
        objpoints_left: List of object point arrays from left camera detection
        indices_left: List of image indices for left detections
        corners_right: List of corner arrays from right camera detection
        objpoints_right: List of object point arrays from right camera detection
        indices_right: List of image indices for right detections
        dy_thresh: Vertical gating threshold in pixels for matching
        scale_ratio_thresh: Relative height difference threshold for matching
        method: Matching method ('hungarian' or 'greedy')
        cache_dir: Optional directory for caching matching results
        progress: Show progress messages

    Returns:
        Tuple of (stereo_objpoints, stereo_imgpoints_left, stereo_imgpoints_right, metadata):
        - stereo_objpoints: Object points for each matched stereo pair
        - stereo_imgpoints_left: Image points for left camera
        - stereo_imgpoints_right: Image points for right camera
        - metadata: Dictionary with matching statistics and per-pair info
    """
    # Check cache
    if cache_dir is not None:
        from .cache import load_from_cache

        cache_key = _compute_matching_cache_key(
            corners_left,
            indices_left,
            corners_right,
            indices_right,
            dy_thresh,
            scale_ratio_thresh,
            method,
        )
        cached = load_from_cache(cache_dir, cache_key, "stereo_matching", progress=progress)
        if cached is not None:
            if progress:
                print(f"Total matched boards: {len(cached['objpoints'])}")
            return (cached["objpoints"], cached["imgpoints_left"], cached["imgpoints_right"], cached["metadata"])

    # Group detections by image index
    left_by_image = group_detections_by_image(corners_left, objpoints_left, indices_left)
    right_by_image = group_detections_by_image(corners_right, objpoints_right, indices_right)
    objpoints_by_image_left: dict[int, list[np.ndarray]] = {}
    for idx, objp in zip(indices_left, objpoints_left, strict=False):
        objpoints_by_image_left.setdefault(idx, []).append(objp)

    # Match boards across stereo pairs
    stereo_objpoints = []
    stereo_imgpoints_left = []
    stereo_imgpoints_right = []

    # Metadata for visualization and analysis
    metadata = {
        "total_pairs_processed": 0,
        "total_boards_matched": 0,
        "matches_per_pair": [],
        "unmatched_left_per_pair": [],
        "unmatched_right_per_pair": [],
        "pair_details": {},  # Store details for visualization: {img_idx: {matches, unmatched_L, unmatched_R}}
    }

    # Get common image indices
    common_indices = sorted(set(left_by_image.keys()) & set(right_by_image.keys()))

    if progress:
        print(f"Matching chessboards across {len(common_indices)} stereo pairs...\n")

    for img_idx in common_indices:
        dets_left = left_by_image[img_idx]
        dets_right = right_by_image[img_idx]

        # Match boards in this stereo pair
        matches, unmatched_L, unmatched_R = match_chessboards(
            dets_left,
            dets_right,
            dy_thresh=dy_thresh,
            scale_ratio_thresh=scale_ratio_thresh,
            method=method,
        )

        metadata["matches_per_pair"].append(len(matches))
        metadata["unmatched_left_per_pair"].append(len(unmatched_L))
        metadata["unmatched_right_per_pair"].append(len(unmatched_R))
        metadata["pair_details"][img_idx] = {
            "matches": matches,
            "unmatched_L": unmatched_L,
            "unmatched_R": unmatched_R,
        }

        # Build stereo pairs from matches
        for iL, iR in matches:
            cornersL, sizeL = dets_left[iL]
            cornersR, sizeR = dets_right[iR]

            # Sizes must match
            if tuple(sizeL) != tuple(sizeR):
                continue

            nb_vert, nb_horiz = sizeL
            expected_corners = nb_vert * nb_horiz

            # Validate corner counts
            if cornersL.shape[0] != expected_corners or cornersR.shape[0] != expected_corners:
                continue

            # pick the objpoints that corresponds to detection iL in this image.
            # rely on group_detections_by_image preserving the same order as objpoints_by_image_left[img_idx]
            try:
                objp_list = objpoints_by_image_left[img_idx]
                objp = objp_list[iL].copy()  # <-- make a copy to avoid shared references
            except (KeyError, IndexError):
                # fallback: keep original behaviour but still copy to avoid mutation
                objp = objpoints_left[indices_left.index(img_idx)].copy()

            stereo_objpoints.append(objp)
            stereo_imgpoints_left.append(cornersL.astype(np.float32))
            stereo_imgpoints_right.append(cornersR.astype(np.float32))
            metadata["total_boards_matched"] += 1

        metadata["total_pairs_processed"] += 1

        if progress:
            print(f"Pair {img_idx}: {len(matches)} matched boards (total so far: {metadata['total_boards_matched']})")

    if progress:
        print(f"\nTotal matched boards across all pairs: {metadata['total_boards_matched']}")

    # Save to cache
    if cache_dir is not None:
        from .cache import save_to_cache

        cache_data = {
            "objpoints": stereo_objpoints,
            "imgpoints_left": stereo_imgpoints_left,
            "imgpoints_right": stereo_imgpoints_right,
            "metadata": metadata,
        }
        save_to_cache(cache_dir, cache_key, "stereo_matching", cache_data, progress=progress)

    return stereo_objpoints, stereo_imgpoints_left, stereo_imgpoints_right, metadata


def _compute_matching_cache_key(
    corners_left: list[np.ndarray],
    indices_left: list[int],
    corners_right: list[np.ndarray],
    indices_right: list[int],
    dy_thresh: float,
    scale_ratio_thresh: float,
    method: str,
) -> str:
    """Compute cache key for stereo matching results."""
    h = hashlib.md5()

    # Hash detection results (just counts and indices for efficiency)
    h.update(str(len(corners_left)).encode())
    h.update(str(len(corners_right)).encode())
    h.update(str(sorted(indices_left)).encode())
    h.update(str(sorted(indices_right)).encode())

    # Hash a sample of corner data to detect changes
    for corners in corners_left[::10]:  # Sample every 10th
        h.update(corners.tobytes())
    for corners in corners_right[::10]:
        h.update(corners.tobytes())

    # Hash matching parameters
    h.update(str(dy_thresh).encode())
    h.update(str(scale_ratio_thresh).encode())
    h.update(method.encode())

    return h.hexdigest()
