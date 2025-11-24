"""
Basic tests for calibration utilities.

Run with: pytest test_calibration.py
"""

import numpy as np
import pytest

from calibration.utils.data_structures import (
    CameraCalibration,
    CameraIndex,
    StereoRig,
)


def test_camera_index_enum():
    """Test CameraIndex enum values."""
    assert CameraIndex.LEFT == 0
    assert CameraIndex.RIGHT == 1


def test_camera_calibration_init():
    """Test CameraCalibration initialization."""
    cam = CameraCalibration()
    assert cam.num_detections() == 0
    assert not cam.is_calibrated()
    assert not cam.is_rectified()


def test_camera_calibration_with_data():
    """Test CameraCalibration with mock data."""
    K = np.eye(3)
    dist = np.zeros(5)

    cam = CameraCalibration(K=K, dist=dist)
    assert cam.is_calibrated()
    assert not cam.is_rectified()
    assert np.array_equal(cam.K, K)


def test_stereo_rig_init():
    """Test StereoRig initialization."""
    rig = StereoRig()
    assert not rig.is_stereo_calibrated()
    assert not rig.is_rectified()


def test_stereo_rig_camera_access():
    """Test camera access by index."""
    rig = StereoRig()
    assert rig.get_camera(CameraIndex.LEFT) is rig.left
    assert rig.get_camera(CameraIndex.RIGHT) is rig.right


def test_stereo_rig_with_calibration():
    """Test StereoRig with mock calibration data."""
    rig = StereoRig()

    # Set up mock calibration
    rig.left.K = np.eye(3)
    rig.left.dist = np.zeros(5)
    rig.right.K = np.eye(3)
    rig.right.dist = np.zeros(5)

    rig.R = np.eye(3)
    rig.T = np.array([0.1, 0, 0])

    assert rig.left.is_calibrated()
    assert rig.right.is_calibrated()
    assert rig.is_stereo_calibrated()


def test_detection_empty_list():
    """Test that empty detection lists work correctly."""
    from calibration.utils.detection import filter_detections_by_image_count

    corners, objpoints, indices = filter_detections_by_image_count([], [], [], min_boards_per_image=1)
    assert len(corners) == 0
    assert len(objpoints) == 0
    assert len(indices) == 0


def test_matching_empty_detections():
    """Test matching with empty detection lists."""
    from calibration.utils.matching import match_chessboards

    matches, unmatched_L, unmatched_R = match_chessboards([], [])
    assert len(matches) == 0
    assert len(unmatched_L) == 0
    assert len(unmatched_R) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
