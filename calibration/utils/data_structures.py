"""
Data structures for stereo camera calibration.

Provides type-safe containers for calibration data and results.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np


class CameraIndex(IntEnum):
    """Enum for camera indices in stereo rig."""

    LEFT = 0
    RIGHT = 1


@dataclass
class CameraCalibration:
    """
    Container for single camera calibration data.

    Attributes:
        objpoints: List of 3D object point arrays (one per detected board)
        imgpoints: List of 2D image point arrays (one per detected board)
        image_indices: List mapping each board detection to source image index
        K: Intrinsic camera matrix (3x3)
        dist: Distortion coefficients
        R: Rectification rotation matrix (3x3, for stereo)
        P: Projection matrix (3x4, for stereo)
        map1: Rectification map x-coordinates
        map2: Rectification map y-coordinates
        roi: Valid region of interest after rectification (x, y, w, h)
        image_size: Image dimensions (width, height)
    """

    objpoints: list[np.ndarray] = field(default_factory=list)
    imgpoints: list[np.ndarray] = field(default_factory=list)
    image_indices: list[int] = field(default_factory=list)
    K: np.ndarray | None = None
    dist: np.ndarray | None = None
    R: np.ndarray | None = None
    P: np.ndarray | None = None
    map1: np.ndarray | None = None
    map2: np.ndarray | None = None
    roi: tuple[int, int, int, int] | None = None
    image_size: tuple[int, int] | None = None

    def is_calibrated(self) -> bool:
        """Check if intrinsic calibration is complete."""
        return self.K is not None and self.dist is not None

    def is_rectified(self) -> bool:
        """Check if rectification maps are computed."""
        return self.map1 is not None and self.map2 is not None

    def num_detections(self) -> int:
        """Return number of board detections."""
        return len(self.objpoints)


@dataclass
class StereoRig:
    """
    Container for stereo camera rig calibration.

    Attributes:
        left: Left camera calibration
        right: Right camera calibration
        R: Rotation matrix from left to right camera (3x3)
        T: Translation vector from left to right camera (3x1)
        E: Essential matrix (3x3)
        F: Fundamental matrix (3x3)
        Q: Disparity-to-depth mapping matrix (4x4)
        stereo_rms: RMS reprojection error from stereo calibration
        baseline: Distance between camera centers (meters)
    """

    left: CameraCalibration = field(default_factory=CameraCalibration)
    right: CameraCalibration = field(default_factory=CameraCalibration)
    R: np.ndarray | None = None
    T: np.ndarray | None = None
    E: np.ndarray | None = None
    F: np.ndarray | None = None
    Q: np.ndarray | None = None
    stereo_rms: float | None = None
    baseline: float | None = None

    def is_stereo_calibrated(self) -> bool:
        """Check if stereo calibration is complete."""
        return self.R is not None and self.T is not None

    def is_rectified(self) -> bool:
        """Check if both cameras have rectification maps."""
        return self.left.is_rectified() and self.right.is_rectified()

    def get_camera(self, idx: CameraIndex) -> CameraCalibration:
        """Get camera by index."""
        return self.left if idx == CameraIndex.LEFT else self.right


@dataclass
class RobustCalibrationResult:
    """
    Results from robust intrinsic calibration.

    Attributes:
        K: Optimized camera matrix
        dist: Optimized distortion coefficients
        rms_final: Final RMS reprojection error
        rms_naive: Initial naive calibration RMS
        inliers_count: Number of inlier boards used in final calibration
        total_detections: Original number of board detections
        removed_stage0: Boards removed in stage 0 filtering
        removed_duplicates_geom: Geometric duplicates removed
        removed_duplicates_extr: Extrinsic pose duplicates removed
        diversity_selected: Boards kept after diversity sampling
        removed_pre: Boards removed in pre-rejection
        removed_post: Boards removed in post-rejection
        final_boards: Final number of boards used
        runtime_sec: Total calibration time
        final_imgpoints: Image points of final inlier boards
    """

    K: np.ndarray
    dist: np.ndarray
    rms_final: float
    rms_naive: float
    inliers_count: int
    total_detections: int
    removed_stage0: int = 0
    removed_duplicates_geom: int = 0
    removed_duplicates_extr: int = 0
    diversity_selected: int = 0
    removed_pre: int = 0
    removed_post: int = 0
    final_boards: int = 0
    runtime_sec: float = 0.0
    final_imgpoints: list[np.ndarray] = field(default_factory=list)

    def summary(self) -> str:
        """Generate human-readable summary."""
        improvement = self.rms_naive - self.rms_final
        return (
            f"Robust Calibration Summary:\n"
            f"  Total detections         : {self.total_detections}\n"
            f"  Removed stage0           : {self.removed_stage0}\n"
            f"  Removed duplicates (geom): {self.removed_duplicates_geom}\n"
            f"  Removed duplicates (extr): {self.removed_duplicates_extr}\n"
            f"  Diversity selected       : {self.diversity_selected}\n"
            f"  Removed pre-rejection    : {self.removed_pre}\n"
            f"  Removed post-rejection   : {self.removed_post}\n"
            f"  Final boards             : {self.final_boards}\n"
            f"  RMS naive                : {self.rms_naive:.4f} px\n"
            f"  RMS final                : {self.rms_final:.4f} px\n"
            f"  Improvement              : {improvement:.4f} px\n"
            f"  Runtime                  : {self.runtime_sec:.2f} s\n"
        )


@dataclass
class CalibrationConfig:
    """
    Configuration container loaded from YAML.

    Wraps configuration dictionary with type hints for common access patterns.
    """

    config: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: Path) -> "CalibrationConfig":
        """Load configuration from YAML file."""
        import yaml

        with Path(path).open() as f:
            config = yaml.safe_load(f)
        return cls(config=config)

    def get(self, key: str, default: Any = None) -> Any:
        """Get nested config value using dot notation (e.g., 'detection.parallel_enabled')."""
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

    @property
    def pattern_sizes(self) -> list[tuple[int, int]]:
        """Get chessboard pattern sizes as list of tuples."""
        sizes = self.get("detection.pattern_sizes", [])
        return [tuple(s) for s in sizes]

    @property
    def square_size(self) -> float:
        """Get chessboard square size in meters."""
        return float(self.get("detection.square_size_meters", 0.1))

    @property
    def visualization_mode(self) -> str:
        """Get visualization mode ('show', 'save', or 'both')."""
        return self.get("visualization.mode", "save")
