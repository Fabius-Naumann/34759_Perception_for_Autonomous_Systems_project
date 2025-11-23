"""
yolo_v8.py

Wrapper around Ultralytics YOLOv8 for object detection on KITTI-like data.

Provides:
    - YoloV8Detector class
    - detect(frame) -> list of dicts with bbox, score, class_id, class_name
"""

import os
from typing import List, Dict, Optional

import numpy as np

try:
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError(
        "ultralytics is not installed. Install it with `pip install ultralytics`."
    ) from e


# COCO class names (YOLOv8 default training set)
# We map COCO classes to your target types (Car, Pedestrian, Cyclist-like)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]


class YoloV8Detector:
    """
    YOLOv8 detector wrapper.

    Usage:
        detector = YoloV8Detector(model_path="yolov8n.pt")
        detections = detector.detect(frame)

    `detect` returns a list of dicts:
        {
            "bbox": [x1, y1, x2, y2],
            "score": float,
            "class_id": int,
            "class_name": str,
            "kitti_type": str or None   # "Car", "Pedestrian", "Cyclist"
        }
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        device: str = "cuda",  # "cpu" or "cuda"
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        restrict_to_kitti_types: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        model_path : str
            Path to YOLOv8 model weights, e.g. "yolov8n.pt", "yolov8s.pt".
            If not present, Ultralytics will try to download it.
        device : str
            "cpu" or "cuda" for GPU.
        conf_threshold : float
            Minimum confidence score to keep a detection.
        iou_threshold : float
            NMS IoU threshold (handled by YOLO internally).
        restrict_to_kitti_types : bool
            If True, only keep detections that map to KITTI types:
            - Car:    COCO "car", "truck", "bus"
            - Pedestrian: COCO "person"
            - Cyclist:    COCO "bicycle", "motorcycle"
        """
        if not os.path.exists(model_path) and not model_path.startswith("yolov8"):
            print(
                f"[YoloV8Detector] Warning: model_path '{model_path}' "
                "does not exist locally. Ultralytics will attempt to download it."
            )

        self.model = YOLO(model_path)
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.restrict_to_kitti_types = restrict_to_kitti_types

        # Map COCO → KITTI-ish labels
        self.kitti_mapping = {
            "person": "Pedestrian",
            "bicycle": "Cyclist",
            "motorcycle": "Cyclist",
            "car": "Car",
            "truck": "Car",
            "bus": "Car",
        }

    # ------------------------------------------------------------------
    def _map_to_kitti_type(self, class_name: str) -> Optional[str]:
        """Map COCO class name to KITTI-like type or None."""
        return self.kitti_mapping.get(class_name, None)

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Run YOLOv8 detection on a single frame.

        Parameters
        ----------
        frame : np.ndarray (H, W, 3) in BGR (OpenCV) format.

        Returns
        -------
        detections : list of dict
            Each dict has keys:
                - "bbox": [x1, y1, x2, y2]
                - "score": float
                - "class_id": int
                - "class_name": str
                - "kitti_type": str or None
        """
        # Ultralytics expects RGB by default; but it also accepts BGR.
        # To be explicit, we can convert:
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_input = frame  # keep BGR; YOLO handles it fine internally

        # Run inference
        results = self.model(
            frame_input,
            device=self.device,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        detections = []

        # YOLOv8 returns a list of Results objects; we passed one image → take first
        result = results[0]

        if result.boxes is None:
            return detections

        # xyxy bounding boxes, confidences, class ids
        boxes = result.boxes.xyxy.cpu().numpy()   # shape (N, 4)
        scores = result.boxes.conf.cpu().numpy()  # shape (N,)
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # shape (N,)

        for box, score, cls_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box
            class_name = COCO_CLASSES[cls_id] if 0 <= cls_id < len(COCO_CLASSES) else str(cls_id)
            kitti_type = self._map_to_kitti_type(class_name)

            if self.restrict_to_kitti_types and kitti_type is None:
                continue

            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "score": float(score),
                "class_id": int(cls_id),
                "class_name": class_name,
                "kitti_type": kitti_type,
            })

        return detections
