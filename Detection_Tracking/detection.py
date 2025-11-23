import cv2
import numpy as np

def detect_moving_objects(flow, frame_prev, frame_curr, min_area=600):
    """
    Returns
      detections: [{cx, cy, x, y, w, h}, ...]
      movement_mask: binary mask (0/1) of moving pixels after all filtering
    """

    h, w = frame_prev.shape[:2]

    # -------------------------------
    # 1) Base motion from flow magnitude (moderate fixed threshold)
    # -------------------------------
    mag = np.linalg.norm(flow, axis=2)

    # Tuned for this sequence (slow walking, static camera)
    base_th = 0.8
    movement_mask = (mag > base_th).astype(np.uint8)

    # -------------------------------
    # 2) HSV shadow suppression
    #    Remove pixels that mostly just get darker
    # -------------------------------
    hsv_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2HSV)
    hsv_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2HSV)

    H1, S1, V1 = cv2.split(hsv_prev)
    H2, S2, V2 = cv2.split(hsv_curr)

    V1_f = V1.astype(np.float32) + 1e-3
    V2_f = V2.astype(np.float32) + 1e-3
    v_ratio = V2_f / V1_f  # < 1 if darker

    dH = cv2.absdiff(H1, H2)
    dS = cv2.absdiff(S1, S2)

    # Slightly stricter than the last version
    shadow_pixels = (
        (v_ratio < 0.85) &   # darker
        (dH < 12)        &   # similar hue
        (dS < 50)            # similar saturation
    )

    shadow_mask = shadow_pixels.astype(np.uint8)
    non_shadow = (1 - shadow_mask).astype(np.uint8)
    movement_mask = movement_mask * non_shadow

    # -------------------------------
    # 3) Background subtraction fusion (MOG2)
    # -------------------------------
    if not hasattr(detect_moving_objects, "bg_subtractor"):
        detect_moving_objects.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=20, detectShadows=False
        )

    fgmask = detect_moving_objects.bg_subtractor.apply(frame_curr)
    fg_binary = (fgmask > 0).astype(np.uint8)

    # Require BOTH flow and bg-sub change
    movement_mask = movement_mask * fg_binary
    movement_mask = movement_mask.astype(np.uint8)

    # -------------------------------
    # 4) Morphology (much stronger to avoid holes in people)
    # -------------------------------

    # Step 1: Remove pixel noise (light opening)
    mask_clean = cv2.morphologyEx(movement_mask * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Step 2: Close holes inside human shapes
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    # Step 3: Dilate slightly to restore full body outline
    mask_clean = cv2.dilate(mask_clean, np.ones((7, 7), np.uint8), iterations=1)

    # Convert to binary mask
    movement_mask_clean = (mask_clean > 0).astype(np.uint8)


    # -------------------------------
    # 5) Connected components
    #    + reject low-flow blobs (e.g. large ground patches)
    # -------------------------------
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_clean)

    detections = []
    for label in range(1, num_labels):
        x_b, y_b, w_b, h_b, area = stats[label]

        if area < min_area:
            continue

        # Average flow magnitude inside this blob
        comp_mask = (labels == label)
        comp_mag_mean = float(mag[comp_mask].mean()) if np.any(comp_mask) else 0.0

        # Reject big blobs whose motion is very weak overall (typical for floor)
        if comp_mag_mean < base_th * 1.2:
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