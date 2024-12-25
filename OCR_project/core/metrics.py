from libs.libs import *
from engines.ocr_engine import apply_ocr
from config.config import *
from core.shared_utils import parse_label_files
import time
import psutil

def monitor_resources(func):
    """Decorator to monitor CPU, memory, and runtime."""
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        start_time = time.time()
        memory_before = process.memory_info().rss / (1024 ** 2)  # in MB

        result = func(*args, **kwargs)

        end_time = time.time()
        memory_after = process.memory_info().rss / (1024 ** 2)  # in MB

        print(f"Runtime: {end_time - start_time:.2f} seconds")
        print(f"Memory Usage: {memory_after - memory_before:.2f} MB")
        
        return result
    return wrapper


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU)."""
    x1 = max(box1[0][0], box2[0][0])
    y1 = max(box1[0][1], box2[0][1])
    x2 = min(box1[2][0], box2[2][0])
    y2 = min(box1[2][1], box2[2][1])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2][0] - box1[0][0]) * (box1[2][1] - box1[0][1])
    box2_area = (box2[2][0] - box2[0][0]) * (box2[2][1] - box2[0][1])
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0
    return inter_area / union_area


def calculate_distance(box1, box2):
    """Calculate Euclidean distance between centers of two bounding boxes."""
    center1 = [(box1[0][0] + box1[2][0]) / 2, (box1[0][1] + box1[2][1]) / 2]
    center2 = [(box2[0][0] + box2[2][0]) / 2, (box2[0][1] + box2[2][1]) / 2]
    return sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)


def calculate_metrics(image_folder, label_folder, num_files, framework,
                      use_cropped=False, iou_threshold=0.3):
    """
    Calculates relevant metrics (CER/WER/Accuracy or IoU-based metrics) depending on
    whether we're using cropped text regions or the full image, for any chosen framework.
    """
    annotations = parse_label_files(label_folder, num_files)
    runtimes = []
    cer_list = []
    wer_list = []
    exact_matches = 0
    iou_scores = []
    distances = []
    
    # For object-detection style metrics
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for idx, (label_file, annotation) in enumerate(annotations.items()):
        image_file = f"im{idx + 1:04d}.jpg"
        image_path = os.path.join(image_folder, image_file)

        if not os.path.exists(image_path):
            print(f"Image file not found: {image_file}")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        start_time = time.time()

        # -------------------------------
        # Case 1: Use Cropped Regions
        # -------------------------------
        if use_cropped:
            skipped_count = 0  # Keep track of how many bounding boxes cannot be predicted
            # For each bounding box & text, compute CER/WER and exact match
            for coords, true_text in annotation:
                result = apply_ocr(image, framework, coords)  
                # 'result' for EasyOCR/PaddleOCR is typically list-of-lists
                # 'result' for Tesseract might be raw string 
                
                # If result is string (e.g., Tesseract raw), just take it directly
                if isinstance(result, str):
                    predicted_text = result.strip().lower()
                else:
                    # Usually a list with (bbox, text, conf) for the first item
                    predicted_text = result[0][-2] if result else ""
                    predicted_text = predicted_text.strip().lower()

                true_text = true_text.strip().lower()

                # Skip empty
                if not true_text or not predicted_text:
                    skipped_count += 1
                    continue

                cer_list.append(cer(true_text, predicted_text))
                wer_list.append(wer(true_text, predicted_text))
                if true_text == predicted_text:
                    exact_matches += 1

            print(f"{skipped_count} bounding boxes had empty text (true or predicted).")
        # -------------------------------
        # Case 2: Full-Image BBoxes
        # -------------------------------
        else:
            # We'll attempt bounding-box metrics
            result = apply_ocr(image, framework)
            
            # If Tesseract returns a string, skip bounding-box IoU
            if isinstance(result, str):
                print("Tesseract string output detected; skipping IoU-based metrics.")
                pass
            else:
                matched_gt_boxes = set()
                for detection in result:
                    bbox, text, confidence = detection
                    best_iou = 0
                    best_gt_box = None

                    for coords, true_text in annotation:
                        iou_val = calculate_iou(
                            bbox,
                            np.array(coords).reshape((4, 2))
                        )
                        if iou_val > best_iou:
                            best_iou = iou_val
                            best_gt_box = coords

                    if best_iou >= iou_threshold:
                        true_positive += 1
                        matched_gt_boxes.add(tuple(best_gt_box))
                        distances.append(
                            calculate_distance(bbox, np.array(best_gt_box).reshape((4, 2)))
                        )
                        iou_scores.append(best_iou)
                    else:
                        false_positive += 1

                for coords, _ in annotation:
                    if tuple(coords) not in matched_gt_boxes:
                        false_negative += 1

        end_time = time.time()
        runtimes.append(end_time - start_time)

    # -------------------------------
    # Final metrics
    # -------------------------------
    avg_runtime = np.mean(runtimes) if runtimes else 0.0
    metrics = {"Avg Runtime (s)": avg_runtime}

    if use_cropped:
        # Character Error Rate / Word Error Rate / Exact match
        total_samples = sum([len(ann) for ann in annotations.values()])
        metrics.update({
            "Avg CER (%)": np.mean(cer_list) * 100 if cer_list else 0,
            "Avg WER (%)": np.mean(wer_list) * 100 if wer_list else 0,
            "Exact Match Accuracy (%)":
                (exact_matches / total_samples) * 100 if total_samples else 0,
        })
    else:
        # IoU-based, plus distances between matched boxes
        precision = (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive) else 0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative) else 0
        )
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) else 0
        )
        
        metrics.update({
            "Avg IoU (%)": np.mean(iou_scores) * 100 if iou_scores else 0,
            "Avg Distance Between Centers": np.mean(distances) if distances else 0,
            "Precision (%)": precision * 100,
            "Recall (%)": recall * 100,
            "F1 Score (%)": f1_score * 100,
        })

    return metrics