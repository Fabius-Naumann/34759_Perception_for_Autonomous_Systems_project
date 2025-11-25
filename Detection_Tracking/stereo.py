import cv2
import numpy as np

# ---- INPUT VIDEOS ----
SEQ = "seq3"
LEFT_VIDEO  = f"./Detection_Tracking/inputs/{SEQ}_image_02_video.mp4"
RIGHT_VIDEO = f"./Detection_Tracking/inputs/{SEQ}_image_03_video.mp4"

OUT_VIDEO   = f"./Detection_Tracking/out/{SEQ}/depth_video.avi"

# ---- Open the videos ----
capL = cv2.VideoCapture(LEFT_VIDEO)
capR = cv2.VideoCapture(RIGHT_VIDEO)

if not capL.isOpened() or not capR.isOpened():
    raise RuntimeError("Could not open stereo videos.")

fps = capL.get(cv2.CAP_PROP_FPS)
w   = int(capL.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(capL.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Resolution:", w, "x", h, "FPS:", fps)

# ---- Stereo SGBM ----
min_disp = 10
num_disp = 16 * 12
block = 7

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=11,
    P1=8 * 3 * 11 * 11,
    P2=32 * 3 * 11 * 11,
    uniquenessRatio=12,
    speckleWindowSize=200,
    speckleRange=3,
    disp12MaxDiff=0
)

# ---- KITTI CALIBRATION ----
# From calib_cam_to_cam.txt (P_rect_02 & P_rect_03)
f = 707.0493        # pixels  (P_rect_02[0,0])
B = 0.4727          # meters  (derived from P_rect_03)

# ---- Output video ----
writer = cv2.VideoWriter(
    OUT_VIDEO,
    cv2.VideoWriter_fourcc(*"XVID"),
    fps,
    (w, h)
)

frame_id = 0

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        break

    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    # ---- Compute disparity ----
    disp = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

    # ---- Compute depth ----
    disp_safe = np.where(disp > min_disp, disp, np.nan)
    depth = (f * B) / disp_safe   # meters

    # ---- Visualize depth ----
    depth_vis = depth.copy()
    depth_vis[np.isnan(depth_vis)] = 0

    # Clip very far points for visualization
    depth_vis = np.clip(depth_vis, 0, 50)   # 0–50 m

    depth_norm = (depth_vis / 50.0 * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_PLASMA)

    writer.write(depth_color)

    print("Processed frame", frame_id)
    frame_id += 1

capL.release()
capR.release()
writer.release()

print("DONE. Saved depth video to:", OUT_VIDEO)
