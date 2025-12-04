import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


def load_calibration(
    yaml_path: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    tuple[int, int],
    tuple[int, int, int, int] | None,
    tuple[int, int, int, int] | None,
]:
    """Load stereo calibration from YAML and return intrinsics, distortion, rectification matrices, image size, and ROIs."""
    with Path(yaml_path).open("r") as f:
        data = yaml.safe_load(f)

    left = data.get("left", {})
    right = data.get("right", {})

    # Intrinsics
    K1 = np.array(left["K"], dtype=np.float64)
    K2 = np.array(right["K"], dtype=np.float64)
    D1 = np.array(left.get("dist", [0, 0, 0, 0, 0]), dtype=np.float64).ravel()
    D2 = np.array(right.get("dist", [0, 0, 0, 0, 0]), dtype=np.float64).ravel()

    # Rectification matrices (if already computed)
    R1 = np.array(left["R"], dtype=np.float64) if "R" in left else None
    R2 = np.array(right["R"], dtype=np.float64) if "R" in right else None
    P1 = np.array(left["P"], dtype=np.float64) if "P" in left else None
    P2 = np.array(right["P"], dtype=np.float64) if "P" in right else None

    # ROI (Region of Interest) for valid pixels after rectification
    roi1 = tuple(map(int, left["roi"])) if left.get("roi") else None
    roi2 = tuple(map(int, right["roi"])) if right.get("roi") else None

    # Image size
    image_size_left = tuple(map(int, left.get("image_size", [0, 0])))
    image_size_right = tuple(map(int, right.get("image_size", [0, 0])))
    if image_size_left != image_size_right:
        raise ValueError("Left and right image sizes differ; cannot rectify.")
    if image_size_left == (0, 0):
        raise ValueError("Image size missing in calibration YAML.")

    return K1, D1, K2, D2, R1, R2, P1, P2, image_size_left, roi1, roi2


def compute_rectification_maps(
    K1: np.ndarray,
    D1: np.ndarray,
    K2: np.ndarray,
    D2: np.ndarray,
    R1: np.ndarray,
    R2: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Compute undistort-rectify maps from precomputed rectification matrices."""
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)

    return (map1x, map1y), (map2x, map2y)


def list_images(folder: str) -> list[str]:
    """List all image files in a folder."""
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    files = []
    folder_path = Path(folder)
    for e in exts:
        files.extend(str(p) for p in folder_path.glob(e))
    files.sort()
    return files


def rectify_folders(
    calib_yaml: str,
    left_dir: str,
    right_dir: str,
    out_left: str,
    out_right: str,
    crop_to_roi: bool = True,
) -> None:
    """Rectify stereo image pairs using precomputed calibration."""
    K1, D1, K2, D2, R1, R2, P1, P2, image_size, roi1, roi2 = load_calibration(calib_yaml)

    if R1 is None or R2 is None or P1 is None or P2 is None:
        raise ValueError("Calibration YAML missing rectification matrices (left.R, left.P, right.R, right.P).")

    (map1x, map1y), (map2x, map2y) = compute_rectification_maps(K1, D1, K2, D2, R1, R2, P1, P2, image_size)

    Path(out_left).mkdir(parents=True, exist_ok=True)
    Path(out_right).mkdir(parents=True, exist_ok=True)

    left_imgs = list_images(left_dir)
    right_imgs = list_images(right_dir)

    if len(left_imgs) == 0 or len(right_imgs) == 0:
        raise RuntimeError("No images found in one of the input folders.")

    if len(left_imgs) != len(right_imgs):
        print(f"Warning: left ({len(left_imgs)}) and right ({len(right_imgs)}) counts differ. Proceeding by index.")

    n = min(len(left_imgs), len(right_imgs))
    for i in range(n):
        l_path = left_imgs[i]
        r_path = right_imgs[i]

        imgL = cv2.imread(l_path, cv2.IMREAD_COLOR)
        imgR = cv2.imread(r_path, cv2.IMREAD_COLOR)

        if imgL is None or imgR is None:
            print(f"Skipping pair {i}: failed to read images.")
            continue

        rectL = cv2.remap(imgL, map1x, map1y, interpolation=cv2.INTER_LINEAR)
        rectR = cv2.remap(imgR, map2x, map2y, interpolation=cv2.INTER_LINEAR)

        # Crop to ROI if requested and available
        if crop_to_roi:
            if roi1 is not None and roi1 != (0, 0, 0, 0):
                x, y, w, h = roi1
                rectL = rectL[y : y + h, x : x + w]
            if roi2 is not None and roi2 != (0, 0, 0, 0):
                x, y, w, h = roi2
                rectR = rectR[y : y + h, x : x + w]

        baseL = Path(l_path).stem
        baseR = Path(r_path).stem
        outL_path = Path(out_left) / f"{baseL}_rect.png"
        outR_path = Path(out_right) / f"{baseR}_rect.png"

        cv2.imwrite(str(outL_path), rectL)
        cv2.imwrite(str(outR_path), rectR)

    print(f"Saved {n} rectified pairs to '{out_left}' and '{out_right}'.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Rectify stereo image folders using calibration YAML")
    p.add_argument(
        "--calib",
        required=True,
        help="Path to stereo calibration YAML (with 'left', 'right' sections)",
    )
    p.add_argument("--left", required=True, help="Folder of left images")
    p.add_argument("--right", required=True, help="Folder of right images")
    p.add_argument("--out-left", required=True, help="Output folder for rectified left")
    p.add_argument("--out-right", required=True, help="Output folder for rectified right")
    p.add_argument(
        "--no-crop",
        action="store_true",
        help="Do not crop to ROI (keep full rectified images)",
    )
    return p.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    try:
        rectify_folders(
            calib_yaml=args.calib,
            left_dir=args.left,
            right_dir=args.right,
            out_left=args.out_left,
            out_right=args.out_right,
            crop_to_roi=not args.no_crop,
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
