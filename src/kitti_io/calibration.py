import numpy as np


class KittiCalibration:
    """
    Robust parser for KITTI-style calibration files, ignoring non-numeric fields
    like dates, strings, etc., and correctly extracting K_XX, R_XX, T_XX,
    P_rect_XX matrices.
    """

    def __init__(self, calib_path):
        self.data = {}
        
        with open(calib_path, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if ":" not in line:
                    continue

                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()

                # Try to parse numerical fields only
                vals = value.split()

                numeric_values = []
                for v in vals:
                    try:
                        numeric_values.append(float(v))
                    except ValueError:
                        # Non-numeric token (e.g., "09-Jan-2012") → ignore whole line
                        numeric_values = None
                        break

                if numeric_values is not None:
                    self.data[key] = np.array(numeric_values, dtype=np.float64)

        # Validate expected keys exist
        required = ["K_02", "K_03", "P_rect_02", "P_rect_03", "T_02", "T_03"]
        for k in required:
            if k not in self.data:
                raise ValueError(f"Calibration file missing required field: {k}")

        # Extract matrices
        self.K2 = self._get_matrix("K_02", (3, 3))
        self.K3 = self._get_matrix("K_03", (3, 3))

        self.P2 = self._get_matrix("P_rect_02", (3, 4))
        self.P3 = self._get_matrix("P_rect_03", (3, 4))

        self.T2 = self._get_vector("T_02")
        self.T3 = self._get_vector("T_03")

        # Compute baseline = difference in x-translation
        self.baseline = abs(self.T3[0] - self.T2[0])

    # -------------------------------------------------------------
    def _get_matrix(self, key, shape):
        arr = self.data[key]
        if arr.size != np.prod(shape):
            raise ValueError(f"Field {key} has wrong number of values.")
        return arr.reshape(shape)

    def _get_vector(self, key):
        return self.data[key]

    # -------------------------------------------------------------
    def print_info(self):
        print("K2 (left intrinsics):\n", self.K2)
        print("K3 (right intrinsics):\n", self.K3)
        print("P2 (P_rect_02):\n", self.P2)
        print("P3 (P_rect_03):\n", self.P3)
        print("T2:", self.T2)
        print("T3:", self.T3)
        print("Baseline:", self.baseline)
