import os
import cv2

def to_xyxy(bbox):
    """
    Convert bbox to [x1,y1,x2,y2] format.
    Accepts either [x1,y1,x2,y2] or [cx,cy,w,h].
    """
    b = list(map(float, bbox))
    # treat as [x1,y1,x2,y2] if it looks like that
    if len(b) == 4 and b[2] >= b[0] and b[3] >= b[1]:
        return [b[0], b[1], b[2], b[3]]
    # otherwise treat as [cx,cy,w,h]
    cx, cy, w, h = b[0], b[1], b[2], b[3]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return [x1, y1, x2, y2]

def iou(boxA, boxB):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.   
    boxA and boxB should be in [x1,y1,x2,y2] format.
    """ 
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH
    areaA = max(0.0, (boxA[2] - boxA[0])) * max(0.0, (boxA[3] - boxA[1]))
    areaB = max(0.0, (boxB[2] - boxB[0])) * max(0.0, (boxB[3] - boxB[1]))
    union = areaA + areaB - interArea
    return 0.0 if union <= 0 else (interArea / union)

# Function to compute mAP
def compute_map(detections, annotations, iou_threshold=0.3):
    """
    Match detections to annotations per-frame and per-class, compute:
      - mAP (simple average of per-frame-per-class precision values)
      - overall precision, recall, F1 (aggregated across all frames/classes)

    detections: dict frame_id -> list of dicts with keys including 'bbox' and 'class' (or 'type')
    annotations: dict frame_id -> list of dicts with keys including 'bbox' and 'type'

    The function will normalize bbox descriptions: accepts either [x1,y1,x2,y2] or [cx,cy,w,h].
    Matching is only allowed between items of the same class (case-insensitive string compare).
    """

    # collect all frame ids
    frames = sorted(set(list(annotations.keys()) + list(detections.keys())))

    total_tp = total_fp = total_fn = 0
    per_frame_class_precisions = []

    for fid in frames:
        gt_objs = annotations.get(fid, [])
        det_objs = detections.get(fid, [])

        # build per-class lists
        gt_by_class = {}
        for g in gt_objs:
            cls = (g.get('type') or g.get('class') or '').lower()
            if g.get('bbox') is None:
                continue
            gt_by_class.setdefault(cls, []).append(to_xyxy(g['bbox']))

        det_by_class = {}
        for d in det_objs:
            cls = (d.get('class') or d.get('type') or '').lower()
            if d.get('bbox') is None:
                continue
            det_by_class.setdefault(cls, []).append(to_xyxy(d['bbox']))

        classes = set(list(gt_by_class.keys()) + list(det_by_class.keys()))
        for cls in classes:
            gts = gt_by_class.get(cls, [])
            dets = det_by_class.get(cls, [])

            matched_gt = set()
            tp = 0
            fp = 0

            for det_bbox in dets:
                best_iou = 0.0
                best_idx = -1
                for i, gt_bbox in enumerate(gts):
                    if i in matched_gt:
                        continue
                    cur_iou = iou(det_bbox, gt_bbox)
                    if cur_iou > best_iou:
                        best_iou = cur_iou
                        best_idx = i
                if best_iou >= iou_threshold and best_idx >= 0:
                    tp += 1
                    matched_gt.add(best_idx)
                else:
                    fp += 1

            fn = len(gts) - len(matched_gt)
            total_tp += tp
            total_fp += fp
            total_fn += fn

            # precision for this frame/class (append for mAP computation)
            if tp + fp > 0:
                per_frame_class_precisions.append(tp / (tp + fp))
            else:
                # if there are GTs but no detections, precision = 0
                if len(gts) > 0:
                    per_frame_class_precisions.append(0.0)
                # if no GTs and no detections skip

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mAP = (sum(per_frame_class_precisions) / len(per_frame_class_precisions)) if per_frame_class_precisions else 0.0

    return {
        'mAP': mAP,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn
    }

def read_detections_file(filepath, class_map=None):
    """
    Reads detection results from a text or CSV file.
    Supports header lines and either comma-separated or whitespace-separated values.

    Flexible expected columns:
      frame, track_id, class, x, y, w, h [, depth]
    where x,y,w,h are top-left coordinates + width/height (not center).
    Returns dict: frame_id -> list of {'id','class','bbox'} where bbox is [x1,y1,x2,y2].
    """
    detections = {}
    with open(filepath, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            low = line.lower()
            if 'frame' in low and 'track' in low and 'class' in low:
                continue

            if ',' in line:
                parts = [p.strip() for p in line.split(',') if p.strip() != '']
            else:
                parts = [p.strip() for p in line.split() if p.strip() != '']

            if len(parts) < 7:
                continue

            try:
                frame_id = int(float(parts[0]))
            except Exception:
                continue
            try:
                obj_id = int(float(parts[1]))
            except Exception:
                obj_id = -1

            class_token = parts[2]
            try:
                class_id = int(float(class_token))
            except Exception:
                class_id = None

            # Interpret the next four numbers as top-left x,y and width,height
            try:
                x = float(parts[3])
                y = float(parts[4])
                w_box = float(parts[5])
                h_box = float(parts[6])
                # convert to corner format [x1,y1,x2,y2]
                x1 = x
                y1 = y
                x2 = x + w_box
                y2 = y + h_box
                bbox = [x1, y1, x2, y2]
            except Exception:
                # fallback: try corner format if parsing above fails
                try:
                    x1 = float(parts[3]); y1 = float(parts[4]); x2 = float(parts[5]); y2 = float(parts[6])
                    bbox = [x1, y1, x2, y2]
                except Exception:
                    continue

            if class_map and class_id is not None:
                class_name = class_map.get(class_id, str(class_id))
            else:
                class_name = class_token if class_id is None else str(class_id)

            detections.setdefault(frame_id, []).append({
                'id': obj_id,
                'class': class_name,
                'bbox': bbox
            })
    return detections

def read_kitti_annotations(filepath):
    """
    Read KITTI-style annotations that may include a leading frame_id and track_id.
    Expected token layout (space separated):
      frame_id track_id type truncated occluded alpha left top right bottom h w l x y z [score]
    Returns: dict(frame_id -> list of dicts with keys:
        'track_id','type','truncated','occluded','alpha','bbox',[left,top,right,bottom],
        'dimensions':[h,w,l],'location':[x,y,z],'score'
    """
    annotations = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            # basic sanity
            try:
                frame_id = int(parts[0])
            except Exception:
                continue

            # optional track id
            try:
                track_id = int(parts[1])
            except Exception:
                track_id = None

            # class/type
            class_name = parts[2] if len(parts) > 2 else 'Unknown'

            # defaults
            truncated = 0.0
            occluded = 0
            alpha = None
            left = top = right = bottom = None
            dimensions = [None, None, None]  # h, w, l
            location = [None, None, None]    # x, y, z
            score = None

            # parse the expected positions if present
            # positions: 3:truncated, 4:occluded, 5:alpha, 6:left, 7:top, 8:right, 9:bottom
            if len(parts) >= 10:
                try:
                    truncated = float(parts[3])
                except Exception:
                    truncated = 0.0
                try:
                    occluded = int(float(parts[4]))
                except Exception:
                    occluded = 0
                try:
                    alpha = float(parts[5])
                except Exception:
                    alpha = None
                try:
                    left = float(parts[6]); top = float(parts[7])
                    right = float(parts[8]); bottom = float(parts[9])
                except Exception:
                    left = top = right = bottom = None

            # dimensions h,w,l at indices 10,11,12
            if len(parts) >= 13:
                try:
                    dimensions = [float(parts[10]), float(parts[11]), float(parts[12])]
                except Exception:
                    dimensions = [None, None, None]

            # location x,y,z at indices 13,14,15
            if len(parts) >= 16:
                try:
                    location = [float(parts[13]), float(parts[14]), float(parts[15])]
                except Exception:
                    location = [None, None, None]

            # optional score at index 16
            if len(parts) >= 17:
                try:
                    score = float(parts[16])
                except Exception:
                    score = None

            bbox = None
            if None not in (left, top, right, bottom):
                bbox = [left, top, right, bottom]

            entry = {
                'track_id': track_id,
                'type': class_name,
                'truncated': truncated,
                'occluded': occluded,
                'alpha': alpha,
                'bbox': bbox,
                'dimensions': dimensions,
                'location': location,
                'score': score
            }

            annotations.setdefault(frame_id, []).append(entry)

    return annotations

def debug_sample_iou(annotations, detections, max_examples=10):
    """
    Print examples of GT vs detection IoUs for frames where class names overlap.
    Useful to spot coordinate or class mismatches.
    """
    printed = 0
    # ensure we intersect key sets, not lists (avoid TypeError)
    ann_keys = set(annotations.keys()) if hasattr(annotations, "keys") else set(annotations)
    det_keys = set(detections.keys()) if hasattr(detections, "keys") else set(detections)
    frames = sorted(ann_keys & det_keys)
    for fid in frames:
        if printed >= max_examples:
            break
        gt_objs = annotations.get(fid, [])
        det_objs = detections.get(fid, [])
        # build lists with normalized class and bbox
        gt_list = []
        for g in gt_objs:
            cls = (g.get('type') or g.get('class') or '').strip().lower()
            if not g.get('bbox'):
                continue
            gt_list.append((cls, to_xyxy(g['bbox']), g))
        det_list = []
        for d in det_objs:
            cls = (d.get('class') or d.get('type') or '').strip().lower()
            if not d.get('bbox'):
                continue
            det_list.append((cls, to_xyxy(d['bbox']), d))

        # for each GT try to find best detection of same class
        for gt_cls, gt_bb, gt_orig in gt_list:
            best_iou = 0.0
            best_det = None
            for det_cls, det_bb, det_orig in det_list:
                if gt_cls != det_cls:
                    continue
                i = iou(gt_bb, det_bb)
                if i > best_iou:
                    best_iou = i
                    best_det = (det_cls, det_bb, det_orig)
            if best_det is not None:
                print(f"[Frame {fid}] class='{gt_cls}' best IoU={best_iou:.3f}")
                print(f"  GT bbox:  {list(map(lambda x: round(x,3), gt_bb))}")
                print(f"  DET bbox: {list(map(lambda x: round(x,3), best_det[1]))}")
                printed += 1
                if printed >= max_examples:
                    break

def visualize_frame_comparison(frame_id, annotations, detections, image_dir, out_dir=None, show_image=False):
    """
    Draws GT and detection boxes for a single frame and saves the visualization.
    Clamps detection boxes to image bounds using x,y,w,h semantics converted to corners.
    """
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(image_dir), "visualizations")
    os.makedirs(out_dir, exist_ok=True)

    candidates = [
        os.path.join(image_dir, f"{frame_id:06d}.png"),
        os.path.join(image_dir, f"{frame_id:06d}.jpg"),
        os.path.join(image_dir, f"{frame_id:06d}.jpeg"),
        os.path.join(image_dir, f"{frame_id}.png"),
        os.path.join(image_dir, f"{frame_id}.jpg"),
    ]
    img_path = next((p for p in candidates if os.path.exists(p)), None)
    if img_path is None:
        print(f"No image found for frame {frame_id} in {image_dir}")
        return None

    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image: {img_path}")
        return None

    h_img, w_img = img.shape[:2]

    # draw ground-truth boxes (green) - GT bboxes assumed already in corner format or cx,cy,w,h convertible by to_xyxy
    for obj in annotations.get(frame_id, []):
        if not obj.get('bbox'):
            continue
        x1, y1, x2, y2 = to_xyxy(obj['bbox'])
        # clamp GT as well (defensive)
        x1 = max(0, min(int(round(x1)), w_img - 1))
        y1 = max(0, min(int(round(y1)), h_img - 1))
        x2 = max(x1 + 1, min(int(round(x2)), w_img))
        y2 = max(y1 + 1, min(int(round(y2)), h_img))
        cls = (obj.get('type') or obj.get('class') or "")[:20]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"GT:{cls}", (x1, max(12, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # draw detection boxes (red) - detections interpreted as x,y,w,h -> converted earlier in read_detections_file to corners
    for obj in detections.get(frame_id, []):
        if not obj.get('bbox'):
            continue
        x1, y1, x2, y2 = to_xyxy(obj['bbox'])
        # Clamp bbox to image bounds using provided semantics
        x1 = max(0, min(int(round(x1)), w_img - 1))
        y1 = max(0, min(int(round(y1)), h_img - 1))
        x2 = max(x1 + 1, min(int(round(x2)), w_img))
        y2 = max(y1 + 1, min(int(round(y2)), h_img))
        cls = (obj.get('class') or obj.get('type') or "")[:20]
        conf = obj.get('confidence') or obj.get('score') or None
        label = f"DET:{cls}" + (f" {conf:.2f}" if conf is not None else "")
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, label, (x1, min(img.shape[0]-4, y2+14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    out_path = os.path.join(out_dir, f"vis_frame_{frame_id:06d}.png")
    cv2.imwrite(out_path, img)
    print(f"Wrote visualization: {out_path}")

    if show_image:
        cv2.imshow(f"frame_{frame_id}", img)
        cv2.waitKey(0)
        cv2.destroyWindow(f"frame_{frame_id}")

    return out_path

# call debug before computing metrics in main
if __name__ == "__main__":

    ## File paths
    Sequence = "seq1"  # "seq1" or "seq2"

    base_dir = os.path.dirname(os.path.abspath(__file__))
    seq_dir = os.path.join(base_dir, "Labels", Sequence)

    labels_path = os.path.join(seq_dir, "labels.txt")
    outputs_path = os.path.join(seq_dir, "output.txt")

    # Load annotations and detections
    class_map = {0: 'Cyclist', 1: 'Car', 2: 'Pedestrian'}
    annotations_data = read_kitti_annotations(labels_path)
    detections_data = read_detections_file(outputs_path, class_map=class_map)

    # Debugging sample IoUs for frames present in both GT and detections
    print("Debugging sample IoUs for frames present in both GT and detections...")
    debug_sample_iou(annotations_data, detections_data, max_examples=20)

    # Compute metrics
    results = compute_map(detections_data, annotations_data, iou_threshold=0.5)
    print("Evaluation Results:")
    print(f"mAP: {results['mAP']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")


    image_dir = os.path.join(os.path.dirname(__file__), "Labels", Sequence, "data")
    visual_path = visualize_frame_comparison(8, annotations_data, detections_data, image_dir)
    print("Visualization saved at:", visual_path)

