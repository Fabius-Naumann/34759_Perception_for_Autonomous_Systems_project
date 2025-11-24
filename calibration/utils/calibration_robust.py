"""
Robust intrinsic and stereo calibration algorithms.

Implements RANSAC-based robust calibration with multiple filtering stages
and optional result caching.
"""

import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from statistics import median

import cv2
import numpy as np

from .data_structures import RobustCalibrationResult, StereoRig

try:
    from tqdm.auto import tqdm
except ImportError:

    def tqdm(it, **_kwargs):
        return it


# -------------------------
# Helper functions
# -------------------------


def _standardize_points(
    obj_list: list[np.ndarray], img_list: list[np.ndarray]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Standardize object and image point arrays to consistent dtypes."""
    o_list = [np.asarray(o, dtype=np.float32).reshape(-1, 3) for o in obj_list]
    i_list = [np.asarray(i, dtype=np.float32).reshape(-1, 1, 2) for i in img_list]
    return o_list, i_list


def _errors_with_model(
    K: np.ndarray,
    dist: np.ndarray,
    obj_list: list[np.ndarray],
    img_list: list[np.ndarray],
    progress: bool = False,
    desc: str = "Errors",
) -> np.ndarray:
    """Compute per-board RMS reprojection errors using given camera model."""
    errs = []
    iterable = list(zip(obj_list, img_list, strict=False))
    if progress and len(obj_list) > 12:
        iterable = tqdm(iterable, total=len(obj_list), desc=desc)
    for o, i in iterable:
        ok, rvec, tvec = cv2.solvePnP(o, i.reshape(-1, 2), K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            rvec = np.zeros((3, 1))
            tvec = np.zeros((3, 1))
        proj, _ = cv2.projectPoints(o, rvec, tvec, K, dist)
        e = np.linalg.norm(proj.reshape(-1, 2) - i.reshape(-1, 2), axis=1)
        errs.append(float(np.sqrt(np.mean(e * e))))
    return np.array(errs, dtype=np.float64)


def _adaptive_threshold(errors: np.ndarray, scale: float = 1.5) -> float:
    """Compute adaptive error threshold using median + scaled IQR."""
    if len(errors) == 0:
        return 0.0
    med = float(median(errors))
    q1 = np.percentile(errors, 25)
    q3 = np.percentile(errors, 75)
    iqr = q3 - q1
    base = iqr if iqr > 0 else (med * 0.5 + 1e-6)
    return med + scale * base


def _pose_vectors(
    K: np.ndarray, dist: np.ndarray, obj_list: list[np.ndarray], img_list: list[np.ndarray]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Compute pose (rvec, tvec) for each board detection."""
    poses = []
    for o, i in zip(obj_list, img_list, strict=False):
        ok, rvec, tvec = cv2.solvePnP(o, i.reshape(-1, 2), K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            rvec = np.zeros((3, 1))
            tvec = np.zeros((3, 1))
        poses.append((rvec.reshape(3), tvec.reshape(3)))
    return poses


def _diversity_subsample(
    poses: list[tuple[np.ndarray, np.ndarray]],
    max_samples: int,
    min_angle_deg: float,
    min_translation: float,
    order: str = "sequential",
) -> list[int]:
    """Select diverse subset of poses based on rotation and translation thresholds."""
    if max_samples <= 0 or len(poses) <= max_samples:
        return list(range(len(poses)))
    min_angle = math.radians(min_angle_deg)
    selected = []
    indices = list(range(len(poses)))
    if order != "sequential":
        indices = list(np.random.default_rng(42).permutation(len(poses)))
    for idx in indices:
        r, t = poses[idx]
        keep = True
        for si in selected:
            r2, t2 = poses[si]
            angle_diff = np.linalg.norm(r - r2)
            trans_diff = np.linalg.norm(t - t2)
            if angle_diff < min_angle and trans_diff < min_translation:
                keep = False
                break
        if keep:
            selected.append(idx)
            if len(selected) >= max_samples:
                break
    return selected


def _deduplicate(
    obj_list: list[np.ndarray],
    img_list: list[np.ndarray],
    hash_precision: int = 2,
    include_img_index: list[int] | None = None,
) -> list[int]:
    """
    Deduplicate detections using geometric hash of corner positions.

    Samples first/mid/last corners at specified precision to build hash key.
    """
    keys = {}
    keep_indices = []
    for idx, (o, i) in enumerate(zip(obj_list, img_list, strict=False)):
        corners = i.reshape(-1, 2)
        first = tuple(np.round(corners[0], hash_precision).astype(float).tolist())
        mid = tuple(np.round(corners[len(corners) // 2], hash_precision).astype(float).tolist())
        last = tuple(np.round(corners[-1], hash_precision).astype(float).tolist())
        img_tag = include_img_index[idx] if include_img_index is not None else None
        key = (o.shape[0], first, mid, last, img_tag)
        if key in keys:
            continue
        keys[key] = idx
        keep_indices.append(idx)
    return keep_indices


def _extrinsic_deduplicate(
    K: np.ndarray,
    dist: np.ndarray,
    obj_list: list[np.ndarray],
    img_list: list[np.ndarray],
    errors: np.ndarray,
    angle_thresh_deg: float = 0.5,
    translation_thresh: float = 0.02,
    keep_per_cluster: int = 1,
) -> list[int]:
    """
    Cluster detections by extrinsic pose similarity and keep best from each cluster.

    Args:
        K, dist: Camera model
        obj_list, img_list: Detections
        errors: Per-detection errors (for ranking)
        angle_thresh_deg: Rotation angle threshold for clustering
        translation_thresh: Translation threshold as fraction of median distance
        keep_per_cluster: Number of detections to keep per cluster

    Returns:
        List of indices to keep
    """
    poses = _pose_vectors(K, dist, obj_list, img_list)
    Rs = []
    Ts = []
    for rvec, tvec in poses:
        R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
        Rs.append(R)
        Ts.append(tvec.reshape(3))

    dists = np.array([np.linalg.norm(t) for t in Ts])
    median_dist = float(np.median(dists)) if len(dists) else 1.0
    translation_thresh_rel = max(translation_thresh * median_dist, 1e-9)

    angle_thresh = math.radians(angle_thresh_deg)

    clusters = []
    for idx in np.argsort(errors):
        R1 = Rs[idx]
        t1 = Ts[idx]
        if np.allclose(t1, 0) and np.allclose(R1, np.eye(3), atol=1e-6):
            continue
        placed = False
        for cluster in clusters:
            rep_idx = cluster[0]
            R2 = Rs[rep_idx]
            t2 = Ts[rep_idx]
            R_rel = R2.T.dot(R1)
            val = (np.trace(R_rel) - 1.0) / 2.0
            val = float(np.clip(val, -1.0, 1.0))
            rel_angle = math.acos(val)
            rel_trans = np.linalg.norm(t1 - t2)
            if rel_angle < angle_thresh and rel_trans < translation_thresh_rel:
                if len(cluster) < max(1, int(keep_per_cluster)):
                    cluster.append(idx)
                placed = True
                break
        if not placed:
            clusters.append([idx])

    keep = []
    for cluster in clusters:
        keep.extend(cluster)
    return keep


# -------------------------
# Main calibration functions
# -------------------------


def calibrate_intrinsics_robust(
    objpoints_list: list[np.ndarray],
    imgpoints_list: list[np.ndarray],
    image_size: tuple[int, int],
    image_indices_list: list[int] | None = None,
    # Stage 0: Per-image provisional filtering
    enable_stage0: bool = True,
    stage0_min_per_image: int = 3,
    # Stage 1: Geometric deduplication
    deduplicate: bool = True,
    hash_precision: int = 2,
    # Stage 2: Extrinsic pose clustering
    extrinsic_deduplicate: bool = True,
    extrinsic_angle_thresh_deg: float = 0.5,
    extrinsic_translation_thresh: float = 0.02,
    extrinsic_keep_per_cluster: int = 1,
    # Stage 3: Diversity subsampling
    diversity_subsample: bool = True,
    max_diverse_samples: int = 150,
    min_pose_angle_deg: float = 2.0,
    min_pose_translation: float = 0.01,
    # Stage 4: Pre-rejection filtering
    pre_rejection: bool = True,
    # Stage 5: RANSAC calibration
    ransac_iterations: int = 150,
    subset_size: int = 6,
    # Stage 6: Post-rejection tightening
    post_rejection: bool = True,
    rms_threshold: float | None = None,
    # Other parameters
    naive_cal_max_samples: int = 50,
    verbose: bool = True,
    progress: bool = True,
    cache_dir: Path | None = None,
) -> RobustCalibrationResult:
    """
    Robust intrinsic camera calibration with RANSAC and multi-stage filtering.

    Implements a multi-stage pipeline:
    1. Optional per-image provisional filtering
    2. Geometric and extrinsic deduplication
    3. Diversity subsampling for pose coverage
    4. Pre-rejection of high-error outliers
    5. RANSAC over calibration subsets
    6. Post-rejection tightening
    7. Final calibration on inliers

    Args:
        objpoints_list: List of 3D object point arrays
        imgpoints_list: List of 2D image point arrays
        image_size: Image dimensions (width, height)
        image_indices_list: Optional list mapping each board to source image

        Stage 0 - Per-image filtering:
        enable_stage0: Enable per-image provisional filtering
        stage0_min_per_image: Minimum boards per image for stage0

        Stage 1 - Geometric deduplication:
        deduplicate: Enable geometric deduplication
        hash_precision: Precision for geometric hashing

        Stage 2 - Extrinsic clustering:
        extrinsic_deduplicate: Enable extrinsic pose clustering
        extrinsic_angle_thresh_deg: Angle threshold for extrinsic clustering (degrees)
        extrinsic_translation_thresh: Translation threshold (fraction of median)
        extrinsic_keep_per_cluster: Boards to keep per pose cluster

        Stage 3 - Diversity subsampling:
        diversity_subsample: Enable pose diversity sampling
        max_diverse_samples: Target number after diversity sampling
        min_pose_angle_deg: Minimum pose angle difference (degrees)
        min_pose_translation: Minimum pose translation difference

        Stage 4 - Pre-rejection:
        pre_rejection: Enable pre-rejection filtering

        Stage 5 - RANSAC:
        ransac_iterations: Number of RANSAC iterations
        subset_size: Subset size for RANSAC hypotheses

        Stage 6 - Post-rejection:
        post_rejection: Enable post-rejection tightening
        rms_threshold: Fixed RMS threshold (None = adaptive)

        Other:
        naive_cal_max_samples: Max boards for initial naive calibration
        verbose: Print progress messages
        progress: Show progress bars
        cache_dir: Directory for caching intermediate results

    Returns:
        RobustCalibrationResult with optimized camera matrix and statistics

    Raises:
        ValueError: If fewer than 4 detections provided
        RuntimeError: If RANSAC fails to produce a candidate model
    """
    t_start = time.time()
    if len(objpoints_list) < 4:
        raise ValueError("Need at least 4 board detections for robust calibration")

    # Check cache if enabled
    if cache_dir is not None:
        from .cache import load_from_cache

        cache_key = _compute_cache_key(objpoints_list, imgpoints_list, image_size)
        cached_result = load_from_cache(cache_dir, cache_key, "intrinsic", progress=False)
        if cached_result is not None:
            if verbose:
                print(f"Loaded cached calibration result (key={cache_key[:8]}...)")
            return cached_result

    # Standardize representations
    obj_all, img_all = _standardize_points(objpoints_list, imgpoints_list)
    img_indices = image_indices_list if image_indices_list is not None else [None] * len(obj_all)
    original_count = len(obj_all)

    # Stage0: per-image provisional filter
    removed_stage0 = 0
    if enable_stage0:
        if verbose:
            print("[Stage 0] Per-image provisional filtering")
        image_groups = defaultdict(list)
        for idx, img_id in enumerate(img_indices):
            image_groups[img_id].append(idx)
        keep_mask = np.ones(len(obj_all), dtype=bool)
        for _img_id, indices in (
            tqdm(image_groups.items(), desc="Stage0") if (progress and len(image_groups) > 4) else image_groups.items()
        ):
            if len(indices) < stage0_min_per_image:
                continue
            obj_subset = [obj_all[i] for i in indices]
            img_subset = [img_all[i] for i in indices]
            _, K_tmp, dist_tmp, _, _ = cv2.calibrateCamera(obj_subset, img_subset, image_size, None, None)
            errs = _errors_with_model(K_tmp, dist_tmp, obj_subset, img_subset, progress=False)
            thr = _adaptive_threshold(errs, scale=1.5)
            for local_i, board_err in zip(indices, errs, strict=False):
                if board_err > thr:
                    keep_mask[local_i] = False
        removed_stage0 = int((~keep_mask).sum())
        if keep_mask.sum() < 4:
            keep_mask[:] = True
        obj_all = [o for o, m in zip(obj_all, keep_mask, strict=False) if m]
        img_all = [i for i, m in zip(img_all, keep_mask, strict=False) if m]
        img_indices = [ii for ii, m in zip(img_indices, keep_mask, strict=False) if m]

    # Geometric deduplication
    removed_duplicates_geom = 0
    if deduplicate:
        dedup_indices = _deduplicate(obj_all, img_all, hash_precision=hash_precision, include_img_index=img_indices)
        if len(dedup_indices) < 4:
            dedup_indices = list(range(len(obj_all)))
        removed_duplicates_geom = len(obj_all) - len(dedup_indices)
        obj_all = [obj_all[i] for i in dedup_indices]
        img_all = [img_all[i] for i in dedup_indices]
        img_indices = [img_indices[i] for i in dedup_indices]

    # Naive calibration for initial model
    naive_cal_max_samples = min(naive_cal_max_samples, max(1, len(obj_all)))
    if verbose:
        print(
            f"[Stage 1] Naive calibration on up to {naive_cal_max_samples} boards "
            f"(removed stage0={removed_stage0}/{original_count}, geom duplicates={removed_duplicates_geom})"
        )

    rng_tmp = np.random.default_rng(2025)
    indexes = list(range(len(obj_all)))
    rng_tmp.shuffle(indexes)
    n_sub = min(naive_cal_max_samples, len(indexes))
    objp_subset = [obj_all[i] for i in indexes[:n_sub]]
    imgp_subset = [img_all[i] for i in indexes[:n_sub]]
    _, K0, dist0, _, _ = cv2.calibrateCamera(objp_subset, imgp_subset, image_size, None, None)

    naive_errors = _errors_with_model(K0, dist0, obj_all, img_all, progress=progress, desc="Naive RMS")
    rms_naive = float(np.sqrt(np.mean(naive_errors**2)))

    # Extrinsic deduplication
    removed_duplicates_extr = 0
    if extrinsic_deduplicate and len(obj_all) > 8:
        keep_extr = _extrinsic_deduplicate(
            K0,
            dist0,
            obj_all,
            img_all,
            naive_errors,
            angle_thresh_deg=extrinsic_angle_thresh_deg,
            translation_thresh=extrinsic_translation_thresh,
            keep_per_cluster=extrinsic_keep_per_cluster,
        )
        if len(keep_extr) >= 4 and len(keep_extr) < len(obj_all):
            removed_duplicates_extr = len(obj_all) - len(keep_extr)
            obj_all = [obj_all[i] for i in keep_extr]
            img_all = [img_all[i] for i in keep_extr]
            img_indices = [img_indices[i] for i in keep_extr]
            naive_errors = naive_errors[keep_extr]
            if verbose:
                print(f"[Stage 1a] Extrinsic clustering: kept {len(obj_all)} (removed {removed_duplicates_extr})")

    # Diversity subsample
    diversity_selected = list(range(len(obj_all)))
    if diversity_subsample and len(obj_all) > max_diverse_samples:
        poses = _pose_vectors(K0, dist0, obj_all, img_all)
        diversity_selected = _diversity_subsample(
            poses,
            max_samples=max_diverse_samples,
            min_angle_deg=min_pose_angle_deg,
            min_translation=min_pose_translation,
            order="sequential",
        )
        if len(diversity_selected) < 4:
            diversity_selected = list(range(len(obj_all)))
        obj_all = [obj_all[i] for i in diversity_selected]
        img_all = [img_all[i] for i in diversity_selected]
        img_indices = [img_indices[i] for i in diversity_selected]
        naive_errors = naive_errors[diversity_selected]
        if verbose:
            print(f"[Stage 1b] Diversity subsample: kept {len(obj_all)} (target={max_diverse_samples})")

    # Pre-rejection
    pre_mask = np.ones(len(obj_all), dtype=bool)
    removed_pre = 0
    if pre_rejection and len(obj_all) > 8:
        if verbose:
            print("[Stage 2] Pre-rejection filtering")
        thr_pre = rms_threshold * 1.5 if rms_threshold is not None else _adaptive_threshold(naive_errors, scale=1.5)
        pre_mask = naive_errors <= thr_pre
        if pre_mask.sum() < 4:
            pre_mask[:] = True
        removed_pre = int((~pre_mask).sum())
        if verbose:
            print(f"Pre-rejection: thr={thr_pre:.3f} removed={removed_pre}")
    pre_obj = [o for o, m in zip(obj_all, pre_mask, strict=False) if m]
    pre_img = [i for i, m in zip(img_all, pre_mask, strict=False) if m]

    # RANSAC
    subset_size = max(4, min(subset_size, len(pre_obj)))
    rng = np.random.default_rng(2025)
    best = None
    if verbose:
        print("[Stage 3] RANSAC search")
        print(f"RANSAC: iters={ransac_iterations} subset_size={subset_size} candidates={len(pre_obj)}")
    r_iter = range(ransac_iterations)
    if progress and ransac_iterations > 10:
        r_iter = tqdm(r_iter, desc="RANSAC", total=ransac_iterations)
    for _ in r_iter:
        if len(pre_obj) <= subset_size:
            subset_idx = list(range(len(pre_obj)))
        else:
            subset_idx = list(rng.choice(len(pre_obj), size=subset_size, replace=False))
        obj_subset = [pre_obj[j] for j in subset_idx]
        img_subset = [pre_img[j] for j in subset_idx]
        _, K_c, dist_c, _, _ = cv2.calibrateCamera(obj_subset, img_subset, image_size, None, None)
        errs_all = _errors_with_model(K_c, dist_c, pre_obj, pre_img, progress=False)
        thr_ransac = rms_threshold * 1.2 if rms_threshold is not None else _adaptive_threshold(errs_all, scale=1.2)
        inlier_mask = errs_all <= thr_ransac
        inlier_count = int(inlier_mask.sum())
        score = float(np.sqrt(np.mean(errs_all[inlier_mask] ** 2))) if inlier_count else math.inf
        if best is None:
            best = (K_c, dist_c, inlier_mask, score, inlier_count)
        else:
            _, _, _, best_score, best_count = best
            if (inlier_count > best_count) or (inlier_count == best_count and score < best_score):
                best = (K_c, dist_c, inlier_mask, score, inlier_count)
    if best is None:
        raise RuntimeError("RANSAC failed to produce a candidate model")
    K_best, dist_best, inlier_mask_best, score_best, count_best = best
    if verbose:
        print(f"RANSAC best: inliers={count_best}/{len(pre_obj)} RMS={score_best:.3f}")
    inlier_obj = [pre_obj[i] for i, m in enumerate(inlier_mask_best) if m]
    inlier_img = [pre_img[i] for i, m in enumerate(inlier_mask_best) if m]

    # Post-rejection
    removed_post = 0
    if post_rejection and len(inlier_obj) > 8:
        if verbose:
            print("[Stage 4] Post-rejection tightening")
        errs_inliers = _errors_with_model(K_best, dist_best, inlier_obj, inlier_img, progress=progress, desc="Post RMS")
        thr_post = rms_threshold if rms_threshold is not None else _adaptive_threshold(errs_inliers, scale=1.0)
        post_mask = errs_inliers <= thr_post
        if post_mask.sum() >= 4:
            removed_post = int((~post_mask).sum())
            inlier_obj = [o for o, m in zip(inlier_obj, post_mask, strict=False) if m]
            inlier_img = [i for i, m in zip(inlier_img, post_mask, strict=False) if m]
            if verbose:
                print(f"Post-rejection: thr={thr_post:.3f} removed={removed_post}")

    # Final calibration
    if verbose:
        print("[Stage 5] Final calibration")
    _, K_f, dist_f, _, _ = cv2.calibrateCamera(inlier_obj, inlier_img, image_size, None, None)
    final_errors = _errors_with_model(K_f, dist_f, inlier_obj, inlier_img, progress=progress, desc="Final RMS")
    rms_final = float(np.sqrt(np.mean(final_errors**2)))

    t_total = time.time() - t_start

    result = RobustCalibrationResult(
        K=K_f,
        dist=dist_f,
        rms_final=rms_final,
        rms_naive=rms_naive,
        inliers_count=len(inlier_obj),
        total_detections=original_count,
        removed_stage0=removed_stage0,
        removed_duplicates_geom=removed_duplicates_geom,
        removed_duplicates_extr=removed_duplicates_extr,
        diversity_selected=len(diversity_selected),
        removed_pre=removed_pre,
        removed_post=removed_post,
        final_boards=len(inlier_obj),
        runtime_sec=t_total,
        final_imgpoints=inlier_img,
    )

    # Cache result if enabled
    if cache_dir is not None:
        from .cache import save_to_cache

        save_to_cache(cache_dir, cache_key, "intrinsic", result, progress=False)

    return result


def subsample_stereo_pairs(
    objpoints: list[np.ndarray],
    imgpoints_left: list[np.ndarray],
    imgpoints_right: list[np.ndarray],
    K_left: np.ndarray,
    dist_left: np.ndarray,
    max_pairs: int = 100,
    min_angle_deg: float = 10.0,
    min_translation: float = 0.05,
    verbose: bool = True,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[int]]:
    """
    Subsample stereo pairs for diversity in left camera pose.

    Uses the left camera poses to determine diversity, keeping pairs that
    provide good geometric coverage for stereo calibration.

    Args:
        objpoints, imgpoints_left, imgpoints_right: Stereo detection data
        K_left, dist_left: Left camera calibration
        max_pairs: Maximum number of pairs to keep
        min_angle_deg: Minimum rotation angle difference (degrees)
        min_translation: Minimum translation difference
        verbose: Print progress information

    Returns:
        Filtered (objpoints, imgpoints_left, imgpoints_right, kept_indices)
    """
    if len(objpoints) <= max_pairs:
        if verbose:
            print(f"Stereo pairs ({len(objpoints)}) <= max_pairs ({max_pairs}), no subsampling needed")
        return objpoints, imgpoints_left, imgpoints_right, list(range(len(objpoints)))

    # Compute poses for left camera
    poses = _pose_vectors(K_left, dist_left, objpoints, imgpoints_left)

    # Select diverse subset
    selected_indices = _diversity_subsample(poses, max_pairs, min_angle_deg, min_translation, order="sequential")

    obj_sub = [objpoints[i] for i in selected_indices]
    img_L_sub = [imgpoints_left[i] for i in selected_indices]
    img_R_sub = [imgpoints_right[i] for i in selected_indices]

    if verbose:
        print(f"Stereo diversity subsampling: {len(objpoints)} → {len(selected_indices)} pairs")
        print(f"  Criteria: min_angle={min_angle_deg}°, min_translation={min_translation}")

    return obj_sub, img_L_sub, img_R_sub, selected_indices


def calibrate_stereo(
    rig: StereoRig,
    objpoints: list[np.ndarray],
    imgpoints_left: list[np.ndarray],
    imgpoints_right: list[np.ndarray],
    image_size: tuple[int, int],
    fix_intrinsic: bool = True,
    max_iterations: int = 100,
    epsilon: float = 1e-5,
    rectify_alpha: float = 0.2,
    zero_disparity: bool = True,
    verbose: bool = True,
) -> StereoRig:
    """
    Perform stereo calibration and rectification.

    Args:
        rig: StereoRig with pre-calibrated cameras (if fix_intrinsic=True)
        objpoints: List of 3D object point arrays
        imgpoints_left: List of left image point arrays
        imgpoints_right: List of right image point arrays
        image_size: Image dimensions (width, height)
        fix_intrinsic: Use fixed intrinsics from rig
        max_iterations: Maximum iterations for stereo calibration
        epsilon: Convergence epsilon
        rectify_alpha: Rectification alpha (0=crop, 1=keep all)
        zero_disparity: Use CALIB_ZERO_DISPARITY flag
        verbose: Print progress messages

    Returns:
        Updated StereoRig with stereo parameters and rectification maps
    """
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, max_iterations, epsilon)
    flags = cv2.CALIB_FIX_INTRINSIC if fix_intrinsic else 0

    if verbose:
        print(f"Stereo calibration with {len(objpoints)} board pairs...")

    rms, K1, d1, K2, d2, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints_left,
        imgpoints_right,
        rig.left.K,
        rig.left.dist,
        rig.right.K,
        rig.right.dist,
        image_size,
        criteria=criteria,
        flags=flags,
    )

    if not fix_intrinsic:
        rig.left.K = K1
        rig.left.dist = d1
        rig.right.K = K2
        rig.right.dist = d2

    rig.R = R
    rig.T = T
    rig.E = E
    rig.F = F
    rig.stereo_rms = rms
    rig.baseline = float(np.linalg.norm(T))

    if verbose:
        print(f"Stereo RMS: {rms:.4f} px")
        print(f"Baseline: {rig.baseline:.4f} m")

    # Rectification
    rect_flags = cv2.CALIB_ZERO_DISPARITY if zero_disparity else 0
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        rig.left.K, rig.left.dist, rig.right.K, rig.right.dist, image_size, R, T, flags=rect_flags, alpha=rectify_alpha
    )

    rig.left.R = R1
    rig.left.P = P1
    rig.left.roi = roi1
    rig.right.R = R2
    rig.right.P = P2
    rig.right.roi = roi2
    rig.Q = Q

    # Compute rectification maps
    rig.left.map1, rig.left.map2 = cv2.initUndistortRectifyMap(
        rig.left.K, rig.left.dist, R1, P1, image_size, cv2.CV_16SC2
    )
    rig.right.map1, rig.right.map2 = cv2.initUndistortRectifyMap(
        rig.right.K, rig.right.dist, R2, P2, image_size, cv2.CV_16SC2
    )

    rig.left.image_size = image_size
    rig.right.image_size = image_size

    if verbose:
        print(f"Rectification complete. ROIs: L={roi1}, R={roi2}")

    return rig


# -------------------------
# Caching helpers
# -------------------------


def _compute_cache_key(objpoints: list[np.ndarray], imgpoints: list[np.ndarray], image_size: tuple[int, int]) -> str:
    """Compute MD5 hash for caching calibration results."""
    data = {
        "objpoints_shapes": [o.shape for o in objpoints],
        "imgpoints_shapes": [i.shape for i in imgpoints],
        "image_size": image_size,
        "objpoints_hash": hashlib.md5(b"".join(o.tobytes() for o in objpoints)).hexdigest(),
        "imgpoints_hash": hashlib.md5(b"".join(i.tobytes() for i in imgpoints)).hexdigest(),
    }
    key_str = json.dumps(data, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()


# Cache helpers now use centralized cache.py utilities
