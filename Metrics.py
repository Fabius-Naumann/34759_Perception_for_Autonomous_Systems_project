from sklearn.metrics import precision_recall_fscore_support


def IoU (boxA, boxB):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    boxA: list or tuple of four floats [x1, y1, x2, y2] representing the first bounding box
    boxB: list or tuple of four floats [x1, y1, x2, y2] representing the second bounding box

    Returns:
    float: IoU value between 0 and 1
    """
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    # Compute the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the IoU
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def mean_average_precision(y_true, y_preds, iou_threshold=0.5):
    """
    Compute the Mean Average Precision (mAP) for object detection.

    Parameters:
    y_true: list of lists of ground truth bounding boxes for each image
    y_preds: list of lists of predicted bounding boxes with confidence scores for each image
    iou_threshold: float, IoU threshold to consider a prediction as true positive

    Returns:
    float: mAP value between 0 and 1
    """
    average_precisions = []

    for true_boxes, pred_boxes in zip(y_true, y_preds):
        true_positives = 0
        false_positives = 0
        false_negatives = len(true_boxes)

        matched = []

        for pred_box, score in sorted(pred_boxes, key=lambda x: x[1], reverse=True):
            best_iou = 0
            best_match = -1

            for i, true_box in enumerate(true_boxes):
                if i in matched:
                    continue
                iou = IoU(pred_box, true_box)
                if iou > best_iou:
                    best_iou = iou
                    best_match = i

            if best_iou >= iou_threshold:
                true_positives += 1
                matched.append(best_match)
                false_negatives -= 1
            else:
                false_positives += 1

        precision = true_positives / (true_positives + false_positives + 1e-6)
        recall = true_positives / (true_positives + false_negatives + 1e-6)

        average_precisions.append(precision)

    mAP = sum(average_precisions) / len(average_precisions) if average_precisions else 0.0
    return mAP


def compute_metrics(detections, annotations, iou_threshold=0.5):
    """ 
    Compute precision, recall, and F1 score for object detection.   
    Parameters:
        detections: list of lists of predicted bounding boxes for each image
        annotations: list of lists of ground truth bounding boxes for each image
        iou_threshold: float, IoU threshold to consider a prediction as true positive
    Returns:
        tuple: precision, recall, and F1 score
    """
    y_true = []
    y_pred = []

    for det, ann in zip(detections, annotations):
        detected = [False] * len(det)
        true_positive = 0
        false_positive = 0
        false_negative = len(ann)

        for a in ann:
            matched = False
            for d in det:
                iou = IoU(d[:4], a)
                if iou >= iou_threshold:
                    matched = True
                    break
            if matched:
                true_positive += 1
                false_negative -= 1
            else:
                false_positive += 1

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / len(ann) if len(ann) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        y_true.extend([1] * len(ann) + [0] * false_positive)
        y_pred.extend([1] * true_positive + [0] * (false_positive + false_negative))

    precision_avg, recall_avg, f1_score_avg, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

    return precision_avg, recall_avg, f1_score_avg


def read_kitti_annotations(file_path):
    """
    Read KITTI format annotations file and extract bounding boxes.
    
    KITTI format columns:
    0: type, 1: truncated, 2: occluded, 3: alpha, 4-7: bbox (left, top, right, bottom),
    8-10: dimensions, 11-13: location, 14: rotation_y, 15: score (optional)
    
    Parameters:
    file_path: string, path to the annotations file
    
    Returns:
    dict: dictionary with frame_id as key and list of detections as value
          Each detection: {'type': str, 'bbox': [x1, y1, x2, y2], 'truncated': int, 
                          'occluded': int, 'confidence': float}
    """
    annotations = {}
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                
                # Extract frame ID and object information
                frame_id = int(parts[0])
                obj_type = parts[2]
                truncated = int(parts[3])
                occluded = int(parts[4])
                
                # Extract bounding box coordinates
                bbox = [float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])]
                
                # Optional: confidence score (if available)
                confidence = float(parts[-1]) if len(parts) > 15 else 1.0
                
                # Store in dictionary
                if frame_id not in annotations:
                    annotations[frame_id] = []
                
                annotations[frame_id].append({
                    'type': obj_type,
                    'bbox': bbox,
                    'truncated': truncated,
                    'occluded': occluded,
                    'confidence': confidence
                })
        
        return annotations
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return {}
    except Exception as e:
        print(f"Error reading file: {e}")
        return {}


def format_for_metrics(annotations_dict):
    """
    Convert parsed annotations to format compatible with compute_metrics function.
    
    Parameters:
    annotations_dict: dict returned from read_kitti_annotations()
    
    Returns:
    list: list of bounding box lists, one per frame
    """
    sorted_frames = sorted(annotations_dict.keys())
    formatted = []
    
    for frame_id in sorted_frames:
        frame_boxes = [det['bbox'] for det in annotations_dict[frame_id]]
        formatted.append(frame_boxes)
    
    return formatted


if __name__ == "__main__":
    # Read KITTI format annotations
    annotations_data = read_kitti_annotations("path/to/your/annotations.txt")
    annotations = format_for_metrics(annotations_data)
    
    detections = None  # Replace with actual detections
    
    # Calculate mAP
    mAP = mean_average_precision(detections, annotations)
    print(f"Mean Average Precision (mAP): {mAP:.4f}")

    # Calculate precision, recall, and F1 score
    precision, recall, f1_score = compute_metrics(detections, annotations)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")