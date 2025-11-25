# calibration_stereo.py
import numpy as np

CALIB_PATH = "./Detection_Tracking/calib_cam_to_cam.txt"


def read_matrix(line):
    """Reads a KITTI-style matrix line: key: v1 v2 ..."""
    return np.array([float(v) for v in line.split()[1:]], dtype=np.float32)


def load_kitti_calibration(path=CALIB_PATH):
    calib = {}
    with open(path, "r") as f:
        for line in f:
            key = line.split(":")[0]

            if key.startswith(("P_rect_", "K_", "R_rect_", "D_")):
                calib[key] = read_matrix(line)

            if key.startswith("S_rect_"):
                calib[key] = np.array([float(v) for v in line.split()[1:]], dtype=np.float32)

    return calib


calib = load_kitti_calibration()

# --- Projection matrices ---
P1 = calib["P_rect_02"].reshape(3, 4)   # rectified left
P2 = calib["P_rect_03"].reshape(3, 4)   # rectified right

# --- Intrinsics ---
K1 = calib["K_02"].reshape(3, 3)
K2 = calib["K_03"].reshape(3, 3)

# --- Image sizes ---
S1 = calib["S_rect_02"].astype(int)
S2 = calib["S_rect_03"].astype(int)
IMAGE_SIZE_L = (int(S1[0]), int(S1[1]))
IMAGE_SIZE_R = (int(S2[0]), int(S2[1]))

# --- Baseline ---
# P = [fx 0 cx -fx*B], so baseline = -P[0,3] / fx
baseline = -(P2[0, 3] - P1[0, 3]) / P1[0, 0]
BASELINE = float(baseline)


if __name__ == "__main__":
    print("P1 (left):\n", P1)
    print("P2 (right):\n", P2)
    print("Baseline:", BASELINE)
    print("Left size:", IMAGE_SIZE_L)
    print("Right size:", IMAGE_SIZE_R)
