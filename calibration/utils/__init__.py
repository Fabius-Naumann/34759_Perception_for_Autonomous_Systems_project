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
    calibrate_stereo_robust,
    compute_stereo_errors,
    subsample_stereo_pairs,
)
from .data_structures import (
    CalibrationConfig,
    CameraCalibration,
    CameraIndex,
    StereoRig,
)
from .detection import detect_chessboards_parallel, find_all_chessboards
from .io import load_calibration, load_kitti_calibration, save_calibration
from .matching import (
    build_stereo_pairs_from_detections,
    group_detections_by_image,
    match_chessboards,
)
from .visualization import (
    plot_all_imgpoints_overlay,
    plot_imagepoints_heatmap,
    plot_matched_boards,
    plot_rectification_preview,
    plot_stereo_pair_coverage,
    plot_undistortion_comparison,
    visualize_stereo_matches,
)

__all__ = [
    # Enums and data structures
    "CalibrationConfig",
    "CameraCalibration",
    "CameraIndex",
    "RobustCalibrationResult",
    "StereoRig",
    # Detection
    "detect_chessboards_parallel",
    "find_all_chessboards",
    # Matching
    "build_stereo_pairs_from_detections",
    "group_detections_by_image",
    "match_chessboards",
    # Calibration
    "calibrate_intrinsics_robust",
    "calibrate_stereo",
    "calibrate_stereo_robust",
    "compute_stereo_errors",
    "subsample_stereo_pairs",
    # Visualization
    "plot_all_imgpoints_overlay",
    "plot_imagepoints_heatmap",
    "plot_matched_boards",
    "plot_rectification_preview",
    "plot_stereo_pair_coverage",
    "plot_undistortion_comparison",
    "visualize_stereo_matches",
    # I/O
    "load_calibration",
    "load_kitti_calibration",
    "save_calibration",
]
