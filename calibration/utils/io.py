"""
I/O utilities for saving and loading calibration results.

Supports YAML, JSON, and NPZ formats with human-readable output.
"""

import contextlib
import json
from pathlib import Path

import numpy as np
import yaml

from .data_structures import CameraIndex, StereoRig


def _numpy_to_python(obj):
    """Convert numpy types to Python native types for JSON/YAML serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_numpy_to_python(item) for item in obj]
    return obj


def save_calibration(rig: StereoRig, output_path: Path | str, fmt: str = "yaml", include_metadata: bool = True) -> None:
    """
    Save stereo calibration results to file.

    Args:
        rig: StereoRig with calibration data
        output_path: Path to output file
        fmt: 'yaml', 'json', or 'npz'
        include_metadata: Include auxiliary metadata (image counts, etc.)

    Raises:
        ValueError: If format is not supported
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt in ("yaml", "json"):
        data = {"stereo": {}, "left": {}, "right": {}}

        # Stereo parameters
        if rig.is_stereo_calibrated():
            data["stereo"] = {
                "R": _numpy_to_python(rig.R),
                "T": _numpy_to_python(rig.T),
                "E": _numpy_to_python(rig.E),
                "F": _numpy_to_python(rig.F),
                "Q": _numpy_to_python(rig.Q),
                "stereo_rms": float(rig.stereo_rms) if rig.stereo_rms is not None else None,
                "baseline": float(rig.baseline) if rig.baseline is not None else None,
            }

        # Per-camera parameters
        for cam_idx in [CameraIndex.LEFT, CameraIndex.RIGHT]:
            cam = rig.get_camera(cam_idx)
            cam_name = "left" if cam_idx == CameraIndex.LEFT else "right"

            cam_data = {}
            if cam.is_calibrated():
                cam_data["K"] = _numpy_to_python(cam.K)
                cam_data["dist"] = _numpy_to_python(cam.dist)

            if cam.is_rectified():
                cam_data["R"] = _numpy_to_python(cam.R)
                cam_data["P"] = _numpy_to_python(cam.P)
                cam_data["roi"] = list(cam.roi) if cam.roi is not None else None

            if include_metadata:
                cam_data["num_detections"] = cam.num_detections()
                cam_data["image_size"] = list(cam.image_size) if cam.image_size is not None else None

            data[cam_name] = cam_data

        # Write file
        if fmt == "yaml":
            with Path(output_path).open("w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        else:  # json
            with Path(output_path).open("w") as f:
                json.dump(data, f, indent=2)

    elif fmt == "npz":
        arrays = {}
        if rig.R is not None:
            arrays["stereo_R"] = rig.R
        if rig.T is not None:
            arrays["stereo_T"] = rig.T
        if rig.E is not None:
            arrays["stereo_E"] = rig.E
        if rig.F is not None:
            arrays["stereo_F"] = rig.F
        if rig.Q is not None:
            arrays["stereo_Q"] = rig.Q
        if rig.stereo_rms is not None:
            arrays["stereo_rms"] = np.array([rig.stereo_rms])
        if rig.baseline is not None:
            arrays["baseline"] = np.array([rig.baseline])

        for cam_idx in [CameraIndex.LEFT, CameraIndex.RIGHT]:
            cam = rig.get_camera(cam_idx)
            prefix = "left" if cam_idx == CameraIndex.LEFT else "right"
            if cam.K is not None:
                arrays[f"{prefix}_K"] = cam.K
            if cam.dist is not None:
                arrays[f"{prefix}_dist"] = cam.dist
            if cam.R is not None:
                arrays[f"{prefix}_R"] = cam.R
            if cam.P is not None:
                arrays[f"{prefix}_P"] = cam.P
            if cam.roi is not None:
                arrays[f"{prefix}_roi"] = np.array(cam.roi)
            if cam.image_size is not None:
                arrays[f"{prefix}_image_size"] = np.array(cam.image_size)

        np.savez_compressed(output_path, **arrays)
    else:
        raise ValueError(f"Unsupported format: {fmt}. Use 'yaml', 'json', or 'npz'.")

    print(f"Calibration saved to {output_path}")


def load_calibration(input_path: Path | str, fmt: str | None = None) -> StereoRig:
    """
    Load stereo calibration results from file.

    Args:
        input_path: Path to calibration file
        fmt: 'yaml', 'json', or 'npz' (auto-detected from extension if None)

    Returns:
        StereoRig with loaded calibration data
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {input_path}")

    if fmt is None:
        ext = input_path.suffix.lower()
        if ext in (".yaml", ".yml"):
            fmt = "yaml"
        elif ext == ".json":
            fmt = "json"
        elif ext == ".npz":
            fmt = "npz"
        else:
            raise ValueError(f"Cannot determine format from extension: {ext}")

    rig = StereoRig()

    if fmt in ("yaml", "json"):
        if fmt == "yaml":
            with Path(input_path).open() as f:
                data = yaml.safe_load(f)
        else:
            with Path(input_path).open() as f:
                data = json.load(f)

        if data.get("stereo"):
            stereo = data["stereo"]
            if "R" in stereo and stereo["R"] is not None:
                rig.R = np.array(stereo["R"], dtype=np.float64)
            if "T" in stereo and stereo["T"] is not None:
                rig.T = np.array(stereo["T"], dtype=np.float64)
            if "E" in stereo and stereo["E"] is not None:
                rig.E = np.array(stereo["E"], dtype=np.float64)
            if "F" in stereo and stereo["F"] is not None:
                rig.F = np.array(stereo["F"], dtype=np.float64)
            if "Q" in stereo and stereo["Q"] is not None:
                rig.Q = np.array(stereo["Q"], dtype=np.float64)
            if "stereo_rms" in stereo:
                rig.stereo_rms = stereo["stereo_rms"]
            if "baseline" in stereo:
                rig.baseline = stereo["baseline"]

        for cam_idx in [CameraIndex.LEFT, CameraIndex.RIGHT]:
            cam_name = "left" if cam_idx == CameraIndex.LEFT else "right"
            cam = rig.get_camera(cam_idx)

            if data.get(cam_name):
                cam_data = data[cam_name]
                if "K" in cam_data and cam_data["K"] is not None:
                    cam.K = np.array(cam_data["K"], dtype=np.float64)
                if "dist" in cam_data and cam_data["dist"] is not None:
                    cam.dist = np.array(cam_data["dist"], dtype=np.float64)
                if "R" in cam_data and cam_data["R"] is not None:
                    cam.R = np.array(cam_data["R"], dtype=np.float64)
                if "P" in cam_data and cam_data["P"] is not None:
                    cam.P = np.array(cam_data["P"], dtype=np.float64)
                if "roi" in cam_data and cam_data["roi"] is not None:
                    cam.roi = tuple(cam_data["roi"])
                if "image_size" in cam_data and cam_data["image_size"] is not None:
                    cam.image_size = tuple(cam_data["image_size"])

    elif fmt == "npz":
        arrays = np.load(input_path)
        if "stereo_R" in arrays:
            rig.R = arrays["stereo_R"]
        if "stereo_T" in arrays:
            rig.T = arrays["stereo_T"]
        if "stereo_E" in arrays:
            rig.E = arrays["stereo_E"]
        if "stereo_F" in arrays:
            rig.F = arrays["stereo_F"]
        if "stereo_Q" in arrays:
            rig.Q = arrays["stereo_Q"]
        if "stereo_rms" in arrays:
            rig.stereo_rms = float(arrays["stereo_rms"][0])
        if "baseline" in arrays:
            rig.baseline = float(arrays["baseline"][0])

        for cam_idx in [CameraIndex.LEFT, CameraIndex.RIGHT]:
            cam = rig.get_camera(cam_idx)
            prefix = "left" if cam_idx == CameraIndex.LEFT else "right"
            if f"{prefix}_K" in arrays:
                cam.K = arrays[f"{prefix}_K"]
            if f"{prefix}_dist" in arrays:
                cam.dist = arrays[f"{prefix}_dist"]
            if f"{prefix}_R" in arrays:
                cam.R = arrays[f"{prefix}_R"]
            if f"{prefix}_P" in arrays:
                cam.P = arrays[f"{prefix}_P"]
            if f"{prefix}_roi" in arrays:
                cam.roi = tuple(arrays[f"{prefix}_roi"])
            if f"{prefix}_image_size" in arrays:
                cam.image_size = tuple(arrays[f"{prefix}_image_size"])
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    print(f"Calibration loaded from {input_path}")
    return rig


def load_kitti_calibration(input_path: Path | str) -> StereoRig:
    """
    Load calibration from KITTI-style calib_cam_to_cam.txt file.

    Assumes Camera 02 is Left and Camera 03 is Right (standard KITTI setup).

    Args:
        input_path: Path to calib_cam_to_cam.txt

    Returns:
        StereoRig populated with K, dist, R, T, etc.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {input_path}")

    data = {}
    with input_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            with contextlib.suppress(ValueError):
                # Parse space-separated floats
                data[key] = np.array([float(x) for x in value.strip().split()])

    rig = StereoRig()

    # Helper to reshape arrays
    def get_matrix(key, shape):
        if key in data:
            return data[key].reshape(shape)
        return None

    # Load Camera 02 (Left)
    if "S_02" in data:
        rig.left.image_size = tuple(data["S_02"].astype(int))
    rig.left.K = get_matrix("K_02", (3, 3))
    rig.left.dist = get_matrix("D_02", (5,))  # KITTI usually has 5 distortion coeffs
    rig.left.R = get_matrix("R_rect_02", (3, 3))  # Rectification rotation
    rig.left.P = get_matrix("P_rect_02", (3, 4))  # Projection matrix

    # Load Camera 03 (Right)
    if "S_03" in data:
        rig.right.image_size = tuple(data["S_03"].astype(int))
    rig.right.K = get_matrix("K_03", (3, 3))
    rig.right.dist = get_matrix("D_03", (5,))
    rig.right.R = get_matrix("R_rect_03", (3, 3))
    rig.right.P = get_matrix("P_rect_03", (3, 4))

    # Calculate Stereo Extrinsics (T and Baseline)
    # In KITTI, T_xx is translation from Cam 00 to Cam xx.
    # We want T from Left (02) to Right (03).
    # T_02 = position of 02 wrt 00
    # T_03 = position of 03 wrt 00
    # Approximate baseline: B = ||T_03 - T_02||
    # More precisely, if we assume cameras are rectified/aligned:
    # Baseline is usually T_02(x) - T_03(x) or similar depending on coordinate system.
    # KITTI T vectors are 3x1.

    t_02 = get_matrix("T_02", (3, 1))
    t_03 = get_matrix("T_03", (3, 1))

    if t_02 is not None and t_03 is not None:
        # Relative translation T_rel = T_03 - T_02 (approximate if R is small)
        # Actually, P_rect_03(0,3) / P_rect_03(0,0) gives baseline * fx
        # P_rect_02(0,3) is usually 0.
        # P_rect_03(0,3) = -fx * baseline

        # if rig.right.P is not None:
        #     fx = rig.right.P[0, 0]
        #     tx = rig.right.P[0, 3]
        #     # baseline in meters = -tx / fx
        #     rig.baseline = abs(tx / fx)

        #     # Populate rig.T with [ -baseline, 0, 0 ] assuming rectification
        #     rig.T = np.array([[-rig.baseline], [0], [0]])
        #     rig.R = np.eye(3)  # Assuming rectified
        # else:
        # Fallback to raw translations
        rig.T = t_03 - t_02
        rig.baseline = np.linalg.norm(rig.T)

    print(f"KITTI calibration loaded from {input_path}")
    return rig
