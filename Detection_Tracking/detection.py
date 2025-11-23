import cv2
import numpy as np

def detect_moving_objects(flow, frame_prev, frame_curr, min_area=1500):
    """
    Returns
      detections: [{cx, cy, x, y, w, h}, ...]
      movement_mask: binary mask (0/1) of moving pixels after all filtering
    """

    h, w = frame_prev.shape[:2]

    # -------------------------------
    # 1) Base motion from flow magnitude (adaptive threshold)
    # -------------------------------
    mag = np.linalg.norm(flow, axis=2)

    valid_mag = mag[mag > 0]
    if valid_mag.size > 0:
        # 95th percentile scaled down – tight but adaptive
        base_th = np.percentile(valid_mag, 95) * 0.050 # PARAM: Bigger, tighter mask
        base_th = max(base_th, 0.5)  # avoid too small
    else:
        base_th = 1.0

    movement_mask = (mag > base_th).astype(np.uint8)

    # # -------------------------------
    # # 2) HSV shadow suppression
    # #    - Remove pixels that only get darker with similar hue/saturation
    # # -------------------------------
    # hsv_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2HSV)
    # hsv_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2HSV)

    # H1, S1, V1 = cv2.split(hsv_prev)
    # H2, S2, V2 = cv2.split(hsv_curr)

    # V1_f = V1.astype(np.float32) + 1e-3
    # V2_f = V2.astype(np.float32) + 1e-3
    # v_ratio = V2_f / V1_f  # < 1 if darker

    # dH = cv2.absdiff(H1, H2)
    # dS = cv2.absdiff(S1, S2)

    # shadow_pixels = (
    #     (v_ratio < 0.7) &    # became darker #PARAM: Bigger, tighter mask
    #     (dH < 20)      &    # hue similar #PARAM: Smaller, tighter mask
    #     (dS < 60)           # saturation similar #PARAM: Smaller, tighter mask
    # )

    # shadow_mask = shadow_pixels.astype(np.uint8)
    # non_shadow = (1 - shadow_mask).astype(np.uint8)
    # movement_mask = movement_mask * non_shadow

    # # -------------------------------
    # # 3) Flow-direction structure filtering (remove fuzzy blobs)
    # #    Real objects have structured internal flow
    # # -------------------------------
    # flow_x = flow[..., 0]
    # flow_y = flow[..., 1]

    # angle = np.arctan2(flow_y, flow_x)

    # angle_blur = cv2.GaussianBlur(angle, (9, 9), 0)
    # angle_sq_blur = cv2.GaussianBlur(angle ** 2, (9, 9), 0)
    # angle_var = angle_sq_blur - angle_blur ** 2

    # flow_structure_mask = (angle_var > 0.0005).astype(np.uint8) # PARAM: Bigger, tighter mask
    # movement_mask = movement_mask * flow_structure_mask

    # -------------------------------
    # 4) Chromaticity-based filtering
    #    Shadows mostly change intensity, not color ratios
    # -------------------------------
    rgb_prev = frame_prev.astype(np.float32) + 1.0
    rgb_curr = frame_curr.astype(np.float32) + 1.0

    sum_prev = np.sum(rgb_prev, axis=2, keepdims=True)
    sum_curr = np.sum(rgb_curr, axis=2, keepdims=True)

    c_prev = rgb_prev / sum_prev
    c_curr = rgb_curr / sum_curr

    # chrom_diff = np.linalg.norm(c_curr - c_prev, axis=2)
    # chrom_mask = (chrom_diff > 0.001).astype(np.uint8) #PARAM: Bigger, tighter mask
    # movement_mask = movement_mask * chrom_mask

    # -------------------------------
    # 5) Background subtraction fusion (MOG2)
    #    Helps tighten silhouettes and remove static regions
    # -------------------------------
    if not hasattr(detect_moving_objects, "bg_subtractor"):
        detect_moving_objects.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )

    fgmask = detect_moving_objects.bg_subtractor.apply(frame_curr)
    fg_binary = (fgmask > 0).astype(np.uint8)

    movement_mask = movement_mask * fg_binary
    movement_mask = movement_mask.astype(np.uint8)

    # -------------------------------
    # 6) Morphological closing (tight, hole-free humans)
    # -------------------------------
    # Morphological cleaning
    kernel_open = np.ones((20, 20), np.uint8)
    kernel_close = np.ones((7, 7), np.uint8) # PARAM: Bigger, tighter mask
    mask_clean = cv2.morphologyEx(movement_mask * 255, cv2.MORPH_OPEN, kernel_open)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_close)

    movement_mask_clean = (mask_clean > 0).astype(np.uint8)

    # -------------------------------
    # 7) Connected components on cleaned mask
    # -------------------------------
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
