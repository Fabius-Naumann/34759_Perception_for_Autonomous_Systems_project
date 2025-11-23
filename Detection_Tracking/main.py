import os
import cv2
import numpy as np
from detection import detect_moving_objects
from tracking import TrackManager

# ---------------- Video Setup ----------------
video_path = "./Detection_Tracking/image_02_video.mp4"  # Run from project root
cap = cv2.VideoCapture(video_path)
ret, frame1 = cap.read()
if not ret:
    raise RuntimeError(f"Cannot read first frame from {video_path}")

gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
h, w = frame1.shape[:2]

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0 or np.isnan(fps):
    # Fallback FPS if OpenCV fails to read it
    fps = 30.0
    print("Warning: FPS from video was invalid. Using fallback fps = 30.0")

# Make sure output directory exists
out_dir = "./Detection_Tracking/out"
os.makedirs(out_dir, exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"XVID")

out_boxes_path = os.path.join(out_dir, "people_boxes.avi")
out_flow_path  = os.path.join(out_dir, "people_flow.avi")
out_det_path   = os.path.join(out_dir, "people_detection.avi")

out_boxes = cv2.VideoWriter(out_boxes_path, fourcc, fps, (w, h))
out_flow  = cv2.VideoWriter(out_flow_path,  fourcc, fps, (w, h))
out_det   = cv2.VideoWriter(out_det_path,   fourcc, fps, (w, h))

if not out_boxes.isOpened():
    raise RuntimeError(f"Failed to open VideoWriter for {out_boxes_path}")
if not out_flow.isOpened():
    raise RuntimeError(f"Failed to open VideoWriter for {out_flow_path}")
if not out_det.isOpened():
    raise RuntimeError(f"Failed to open VideoWriter for {out_det_path}")


# ---------------- Main Loop ----------------
frame_count = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 10 == 0:
        print(f"Processed {frame_count} / {total_frames} frames")

    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Optical flow (Farneback)
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        0.5, 3, 20, 5, 5, 1.2, 0
    )

    # -------- Detection --------
    detections, movement_mask = detect_moving_objects(flow, frame1, frame2)

    # -------- Detection visualization video (red semitransparent mask) --------
    det_vis = frame2.copy()
    overlay = det_vis.copy()

    mask_bool = movement_mask.astype(bool)
    overlay[mask_bool] = (0, 0, 255)  # BGR red

    det_vis = cv2.addWeighted(det_vis, 0.7, overlay, 0.3, 0.0)
    out_det.write(det_vis)

    # -------- Draw optical flow visualization --------
    flow_vis = frame2.copy()
    step = 8
    for y in range(0, h, step):
        for x in range(0, w, step):
            fx, fy = flow[y, x]
            if np.hypot(fx, fy) < 2:
                continue
            end_point = (int(x + fx), int(y + fy))
            cv2.arrowedLine(flow_vis, (x, y), end_point, (0, 0, 255), 1, tipLength=0.3)

    out_flow.write(flow_vis)

    # ---------------- Draw detection bounding boxes ----------------
    frame_boxes = frame2.copy()

    for det in detections:
        x, y, w, h = det["x"], det["y"], det["w"], det["h"]

        # Blue bounding box
        cv2.rectangle(
            frame_boxes,
            (x, y),
            (x + w, y + h),
            (255, 0, 0), 2
        )

        # Center point (yellow dot)
        cx, cy = int(det["cx"]), int(det["cy"])
        cv2.circle(frame_boxes, (cx, cy), 4, (0, 255, 255), -1)

    # Save the box-visualization video
    out_boxes.write(frame_boxes)




    # Prepare next frame
    gray1 = gray2.copy()
    frame1 = frame2.copy()

cap.release()
out_boxes.release()
out_flow.release()
out_det.release()

print("Saved people_boxes.avi, people_flow.avi and people_detection.avi")
