"""
Calibration utilities package for stereo camera calibration.

This package provides modular tools for:
- Chessboard detection and matching
- Robust intrinsic and stereo calibration
- Visualization and result persistence
"""

from .calibration_robust import (
    calibrate_intrinsics_robust,
    calibrate_stereo,
    subsample_stereo_pairs,
)
from .data_structures import (
    CalibrationConfig,
    CameraCalibration,
    CameraIndex,
    StereoRig,
)
from .detection import detect_chessboards_parallel, find_all_chessboards
from .io import load_calibration, save_calibration
from .matching import build_stereo_pairs_from_detections, match_chessboards
from .visualization import (
    plot_imagepoints_heatmap,
    plot_matched_boards,
    plot_rectification_preview,
    plot_stereo_pair_coverage,
    plot_undistortion_comparison,
)

__all__ = [
    # Enums and data structures
    "CameraIndex",
    "CameraCalibration",
    "StereoRig",
    "CalibrationConfig",
    # Detection
    "find_all_chessboards",
    "detect_chessboards_parallel",
    # Matching
    "match_chessboards",
    "build_stereo_pairs_from_detections",
    # Calibration
    "calibrate_intrinsics_robust",
    "calibrate_stereo",
    "subsample_stereo_pairs",
    # Visualization
    "plot_undistortion_comparison",
    "plot_imagepoints_heatmap",
    "plot_matched_boards",
    "plot_rectification_preview",
    "plot_stereo_pair_coverage",
    # I/O
    "save_calibration",
    "load_calibration",
]
