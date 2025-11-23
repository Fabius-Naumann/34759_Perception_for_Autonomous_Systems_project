import cv2
import numpy as np

def detect_moving_objects(flow, frame_prev, frame_curr, min_area=800):
    """
    Returns
      detections: [{cx, cy, x, y, w, h}, ...]
      movement_mask: binary mask of moving pixels (after shadow removal + cleaning)
    """

    h, w = frame_prev.shape[:2]

    # Magnitude of flow
    mag = np.linalg.norm(flow, axis=2)
    movement_mask = (mag > 3.0).astype(np.uint8)

    # Morphological cleaning
    kernel_open = np.ones((5, 5), np.uint8)
    kernel_close = np.ones((9, 9), np.uint8)
    mask_clean = cv2.morphologyEx(movement_mask * 255, cv2.MORPH_OPEN, kernel_open)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_close)

    # Convert back to 0/1 mask for downstream usage
    movement_mask_clean = (mask_clean > 0).astype(np.uint8)

    # Connected components on cleaned mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_clean)

    detections = []
    for label in range(1, num_labels):
        x_b, y_b, w_b, h_b, area = stats[label]

        if area < min_area:
            continue
        if w_b < 20 or h_b < 40:
            continue

        aspect = h_b / (w_b + 1e-6)
        if aspect < 0.7 or aspect > 4.0:
            continue

        cx, cy = centroids[label]

        detections.append({
            "cx": float(cx),
            "cy": float(cy),
            "x": int(x_b),
            "y": int(y_b),
            "w": int(w_b),
            "h": int(h_b)
        })

    return detections, movement_mask_clean
