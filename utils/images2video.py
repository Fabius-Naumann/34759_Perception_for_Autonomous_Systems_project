import os
import cv2
import numpy as np
from datetime import datetime

# ------------------------------------------------------------
# Helper: Parse KITTI timestamps (nanoseconds → microseconds)
# ------------------------------------------------------------
def parse_kitti_timestamp(t_str):
    """
    Converts KITTI timestamp format:
    '2011-09-28 12:50:02.009517056'
    
    Into a Python datetime (microseconds only).
    """
    date, time = t_str.split(" ")
    
    # Split seconds + nanoseconds
    if "." in time:
        sec, fraction = time.split(".")
        micro = fraction[:6]  # trim nanoseconds → microseconds
    else:
        sec = time
        micro = "000000"
    
    # Build string suitable for datetime
    fixed = f"{date} {sec}.{micro}"
    return datetime.strptime(fixed, "%Y-%m-%d %H:%M:%S.%f")


# ------------------------------------------------------------
# Main function
# ------------------------------------------------------------
def images_and_timestamps_to_video(root_dir, output_path, default_fps=30.0):
    """
    Reads PNG images and KITTI timestamps, builds a video.
    
    root_dir:
        Expects structure:
            image_02/
                data/*.png
                timestamps.txt
    """
    data_dir = os.path.join(root_dir, "data")
    ts_path = os.path.join(root_dir, "timestamps.txt")

    # ------------------- Load image files -------------------
    frames = sorted([
        f for f in os.listdir(data_dir)
        if f.lower().endswith(".png")
    ])
    if not frames:
        raise RuntimeError(f"No PNG images found inside {data_dir}")

    frame_paths = [os.path.join(data_dir, f) for f in frames]

    # ------------------- Load KITTI timestamps -------------------
    timestamps = None
    if os.path.exists(ts_path):
        with open(ts_path, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        try:
            dt_list = [parse_kitti_timestamp(l) for l in lines]

            if len(dt_list) == len(frame_paths):
                t0 = dt_list[0]
                timestamps = np.array([(dt - t0).total_seconds()
                                       for dt in dt_list])
            else:
                print("Warning: timestamp count mismatch. Using default FPS.")
        except Exception as e:
            print("Timestamp parsing error:", e)
            print("Falling back to default FPS.")

    # ------------------- Determine FPS -------------------
    if timestamps is not None and len(timestamps) > 1:
        duration = timestamps[-1] - timestamps[0]
        fps = (len(timestamps) - 1) / duration if duration > 0 else default_fps
        print(f"Computed FPS from KITTI timestamps: {fps:.3f}")
    else:
        fps = default_fps
        print(f"Using default FPS: {fps:.3f}")

    # ------------------- Prepare video writer -------------------
    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        raise RuntimeError(f"Could not read first frame {frame_paths[0]}")

    h, w = first_frame.shape[:2]

    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".avi":
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # ------------------- Write frames -------------------
    for i, img_path in enumerate(frame_paths):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: could not read {img_path}, skipping.")
            continue

        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))

        out.write(img)

        if (i + 1) % 50 == 0:
            print(f"Wrote {i + 1} frames...")

    out.release()
    print(f"\nVideo successfully written to: {output_path}")


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    DATASET_DIR = "C:\\Users\\roger\\OneDrive\\Documentos\\06_Uni 6\\Perception\\Final project\\34759_final_project_rect\\"
    images_and_timestamps_to_video(
        root_dir=os.path.join(DATASET_DIR, "seq_02\\image_03"),
        output_path=os.path.join(DATASET_DIR, "seq_02\\seq2_image_03_video.mp4"),
        default_fps=30.0
    )
