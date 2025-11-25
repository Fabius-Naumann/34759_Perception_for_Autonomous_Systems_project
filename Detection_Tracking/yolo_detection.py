import cv2
import numpy as np
from ultralytics import YOLO

def yolo_init(model_path="yolov8n.pt"):
    """
    Loads YOLO model only once.
    """
    if not hasattr(yolo_init, "model"):
        print("[INFO] Loading YOLO model:", model_path)
        yolo_init.model = YOLO(model_path)
    return yolo_init.model

def detect_moving_objects(frame):
    """
    YOLO-based detector.
    Returns:
      detections: [{cx, cy, x, y, w, h}, ...]
      movement_mask: dummy 0 mask (kept for compatibility)
    """

    model = yolo_init()   # Load once

    # Run YOLO
    results = model(frame, verbose=False)[0]

    detections = []
    h, w = frame.shape[:2]

    for box in results.boxes:
        cls_id = int(box.cls[0])

        # Only detect people  (cls 0 in COCO)
        if cls_id != 0:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        w_box = x2 - x1
        h_box = y2 - y1
        cx = x1 + w_box / 2
        cy = y1 + h_box / 2

        detections.append({
            "cx": float(cx),
            "cy": float(cy),
            "x": int(x1),
            "y": int(y1),
            "w": int(w_box),
            "h": int(h_box),
        })

    # We no longer use motion mask; return a blank mask to keep compatibility
    movement_mask = np.zeros((h, w), dtype=np.uint8)

    return detections, movement_mask



# -------------------------------------------------------------------------
# NEW: Visualization helper — Red semitransparent detection mask
# -------------------------------------------------------------------------
def draw_movement_mask(frame, movement_mask):
    """
    Draws a red semitransparent mask over detected movement.
    Returns a visualization frame.
    """
    vis = frame.copy()
    overlay = vis.copy()

    mask_bool = movement_mask.astype(bool)
    overlay[mask_bool] = (0, 0, 255)  # Red

    vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0.0)
    return vis


# -------------------------------------------------------------------------
# NEW: Optical flow visualization
# -------------------------------------------------------------------------
def draw_optical_flow(frame, flow, step=8, min_mag=2.0):
    """
    Draws optical flow arrows.
    Returns a visualization frame.
    """
    h, w = frame.shape[:2]
    vis = frame.copy()

    for y in range(0, h, step):
        for x in range(0, w, step):
            fx, fy = flow[y, x]
            if np.hypot(fx, fy) < min_mag:
                continue

            end_point = (int(x + fx), int(y + fy))
            cv2.arrowedLine(vis, (x, y), end_point, (0, 0, 255), 1, tipLength=0.3)

    return vis


# -------------------------------------------------------------------------
# NEW: Bounding boxes + center dots
# -------------------------------------------------------------------------
def draw_detections(frame, detections):
    """
    Draws bounding boxes + center points.
    Returns a visualization frame.
    """
    vis = frame.copy()

    for det in detections:
        x, y, w, h = det["x"], det["y"], det["w"], det["h"]

        # Box
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Center point
        cx, cy = int(det["cx"]), int(det["cy"])
        cv2.circle(vis, (cx, cy), 4, (0, 255, 255), -1)

    return vis