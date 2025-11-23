import os
import glob
import cv2
import numpy as np
import pandas as pd
import datetime


class SequenceLoader:
    """
    Loads a KITTI-style stereo image sequence with timestamps:
    
    seq_xxx/
    ├── image_02/
    │   ├── data/*.png      # left images
    │   └── timestamps.txt
    └── image_03/
        ├── data/*.png      # right images
        └── timestamps.txt
    """

    def __init__(self, seq_path, return_gray=False):
        """
        Parameters:
            seq_path (str): path to e.g. data/seq_01
            return_gray (bool): return grayscale frames instead of RGB
        """
        self.seq_path = seq_path
        self.return_gray = return_gray

        # Left and right image directories
        self.left_img_dir  = os.path.join(seq_path, "image_02", "data")
        self.right_img_dir = os.path.join(seq_path, "image_03", "data")

        # Timestamps
        self.left_ts_path  = os.path.join(seq_path, "image_02", "timestamps.txt")
        self.right_ts_path = os.path.join(seq_path, "image_03", "timestamps.txt")

        # Load all image file paths
        
        self.left_images  = sorted(glob.glob(os.path.join(self.left_img_dir, "*.png")))
        self.right_images = sorted(glob.glob(os.path.join(self.right_img_dir, "*.png")))

        if len(self.left_images) == 0:
            raise RuntimeError(f"No left camera images found in {self.left_img_dir}")
        if len(self.right_images) == 0:
            raise RuntimeError(f"No right camera images found in {self.right_img_dir}")

        # Load timestamps
        self.left_timestamps  = self._load_timestamps(self.left_ts_path)
        self.right_timestamps = self._load_timestamps(self.right_ts_path)

        # Sanity checks
        if len(self.left_images) != len(self.right_images):
            raise ValueError("Left/right camera image counts do not match!")

        if len(self.left_timestamps) != len(self.left_images):
            raise ValueError("Mismatch between left images and timestamps!")

        if len(self.right_timestamps) != len(self.right_images):
            raise ValueError("Mismatch between right images and timestamps!")

        self.num_frames = len(self.left_images)

    # ------------------------------------------------------------


    def _load_timestamps(self, file_path):
        """
        Load KITTI-style timestamps with nanosecond precision, like:
        2011-09-28 12:50:02.009517056
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        timestamps = []

        with open(file_path, "r") as f:
            for line in f:
                ts = line.strip()
                if not ts:
                    continue

                # Split into date, time
                date_str, time_str = ts.split()

                # Split seconds into "seconds" and fractional
                if "." in time_str:
                    sec, frac = time_str.split(".")
                    # Truncate to microseconds (6 digits)
                    frac = frac[:6]
                    time_str = f"{sec}.{frac}"
                else:
                    # no fractional seconds
                    pass

                # Recombine into a valid datetime string
                ts_clean = f"{date_str} {time_str}"

                # Parse into datetime object
                dt = datetime.datetime.strptime(ts_clean, "%Y-%m-%d %H:%M:%S.%f")

                # Convert to UNIX timestamp
                timestamps.append(dt.timestamp())

        return timestamps


    # ------------------------------------------------------------
    def __len__(self):
        return self.num_frames

    # ------------------------------------------------------------
    def get_frame(self, idx):
        """Return a stereo frame pair and metadata by index."""
        if idx < 0 or idx >= self.num_frames:
            raise IndexError("Frame index out of range")

        # Load images
        left  = cv2.imread(self.left_images[idx], cv2.IMREAD_COLOR)
        right = cv2.imread(self.right_images[idx], cv2.IMREAD_COLOR)

        if left is None or right is None:
            raise RuntimeError(f"Failed to read images at frame {idx}")

        if self.return_gray:
            left  = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        return {
            "frame_id": idx,
            "left": left,
            "right": right,
            "timestamp_left": self.left_timestamps[idx],
            "timestamp_right": self.right_timestamps[idx],
            "left_path": self.left_images[idx],
            "right_path": self.right_images[idx]
        }

    # ------------------------------------------------------------
    def __iter__(self):
        """Enables: for frame in loader:"""
        for i in range(self.num_frames):
            yield self.get_frame(i)


# ===================================================================
# Optional helper to load labels from labels.txt
# ===================================================================
def load_labels(labels_file):
    """
    Load labels.txt into a DataFrame.
    Format (17 columns):
    frame, track_id, type, truncated, occluded, alpha,
    bbox_left, bbox_top, bbox_right, bbox_bottom,
    dim_h, dim_w, dim_l,
    loc_x, loc_y, loc_z,
    rotation_y
    """

    if not os.path.exists(labels_file):
        print(f"[WARNING] No labels found: {labels_file}")
        return None

    # Column names as specified in the dataset description:
    columns = [
        "frame", "track_id", "type", "truncated", "occluded", "alpha",
        "bbox_left", "bbox_top", "bbox_right", "bbox_bottom",
        "dim_h", "dim_w", "dim_l",
        "loc_x", "loc_y", "loc_z",
        "rotation_y"
    ]

    df = pd.read_csv(labels_file, sep=" ", header=None, names=columns)
    return df

