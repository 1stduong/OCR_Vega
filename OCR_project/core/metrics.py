from libs.libs import *
from config.config import *

def calculate_metrics(ground_truths, predictions):
    """
    Calculates CER, WER, and Exact Match Accuracy for OCR results.

    Args:
        ground_truths (list of str): Ground truth texts.
        predictions (list of str): Predicted texts from the OCR framework.

    Returns:
        dict: Calculated metrics: CER, WER, and Exact Match Accuracy.
    """
    def calculate_edit_distance(ref, hyp):
        """Calculates the edit distance between reference and hypothesis."""
        matcher = SequenceMatcher(None, ref, hyp)
        return sum([tag[2] for tag in matcher.get_opcodes() if tag[0] != 'equal'])

    total_cer = 0
    total_wer = 0
    exact_matches = 0

    for gt, pred in zip(ground_truths, predictions):
        gt_words = gt.split()
        pred_words = pred.split()

        # Calculate CER
        char_distance = calculate_edit_distance(gt, pred)
        total_cer += char_distance / max(len(gt), 1)

        # Calculate WER
        word_distance = calculate_edit_distance(gt_words, pred_words)
        total_wer += word_distance / max(len(gt_words), 1)

        # Exact Match
        if gt == pred:
            exact_matches += 1

    num_samples = len(ground_truths)
    metrics = {
        "Avg CER (%)": (total_cer / num_samples) * 100 if num_samples > 0 else 0,
        "Avg WER (%)": (total_wer / num_samples) * 100 if num_samples > 0 else 0,
        "Exact Match Accuracy (%)": (exact_matches / num_samples) * 100 if num_samples > 0 else 0
    }
    return metrics

# def calculate_iou(box1, box2):
#     """Calculate Intersection over Union (IoU)."""
#     # Ensure box1 and box2 are numpy arrays of shape (4, 2)
#     box1 = np.array(box1).reshape((4, 2)) if not isinstance(box1, np.ndarray) else box1
#     box2 = np.array(box2).reshape((4, 2)) if not isinstance(box2, np.ndarray) else box2

#     x1 = max(box1[0][0], box2[0][0])
#     y1 = max(box1[0][1], box2[0][1])
#     x2 = min(box1[2][0], box2[2][0])
#     y2 = min(box1[2][1], box2[2][1])

#     inter_area = max(0, x2 - x1) * max(0, y2 - y1)
#     box1_area = (box1[2][0] - box1[0][0]) * (box1[2][1] - box1[0][1])
#     box2_area = (box2[2][0] - box2[0][0]) * (box2[2][1] - box2[0][1])
#     union_area = box1_area + box2_area - inter_area

#     if union_area == 0:
#         return 0
#     return inter_area / union_area

# def calculate_distance(box1, box2):
#     """Calculate Euclidean distance between centers of two bounding boxes."""
#     center1 = [(box1[0][0] + box1[2][0]) / 2, (box1[0][1] + box1[2][1]) / 2]
#     center2 = [(box2[0][0] + box2[2][0]) / 2, (box2[0][1] + box2[2][1]) / 2]
#     return np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

# def safe_cer(true_text, predicted_text):
#     if not true_text or not predicted_text:
#         return None
#     return cer(true_text, predicted_text)

# def safe_wer(true_text, predicted_text):
#     if not true_text or not predicted_text:
#         return None
#     return wer(true_text, predicted_text)

# def calculate_metrics_with_results(ocr_results, label_folder, use_cropped=False, iou_threshold=0.3):
#     """
#     Calculate metrics using precomputed OCR results.

#     Args:
#         ocr_results (dict): Results from OCR application, containing predictions and metadata.
#         label_folder (str): Path to label files.
#         use_cropped (bool): Whether cropped regions were used.
#         iou_threshold (float): IoU threshold for bounding box metrics.

#     Returns:
#         dict: Computed metrics.
#     """
#     # Parse annotations
#     annotations = parse_label_files(label_folder, len(ocr_results.get("predictions", [])))
#     if not annotations:
#         print("No annotations found. Ensure label files are correctly formatted and available.")
#         return {"Error": "No annotations"}

#     cer_list = [], wer_list = [],exact_matches = 0
#     iou_scores = [], distances = []
#     true_positive = 0, false_positive = 0, false_negative = 0

#     for prediction in ocr_results["predictions"]:
#         image_file = prediction.get("image_file", "")
#         annotation = annotations.get(image_file, [])

#         if not annotation:
#             print(f"No annotation found for image: {image_file}")
#             continue

#         if use_cropped:
#             # Handle cropped text regions
#             for coords, true_text in annotation:
#                 predicted_text = prediction.get("predicted_text", "").strip().lower()
#                 true_text = true_text.strip().lower()

#                 if not predicted_text or not true_text:
#                     continue

#                 # CER and WER calculations
#                 cer_value = safe_cer(true_text, predicted_text)
#                 wer_value = safe_wer(true_text, predicted_text)
#                 if cer_value is not None:
#                     cer_list.append(cer_value)
#                 if wer_value is not None:
#                     wer_list.append(wer_value)

#                 if true_text == predicted_text:
#                     exact_matches += 1
#         else:
#             # Handle bounding boxes
#             matched_gt_boxes = set()
#             try:
#                 detection_bbox = prediction.get("bbox", [])
#                 # Ensure bbox is a single list of four points
#                 if isinstance(detection_bbox, list) and len(detection_bbox) == 4 and all(isinstance(point, (list, tuple)) and len(point) == 2 for point in detection_bbox):
#                     bbox = np.array(detection_bbox).reshape((4, 2))
#                 else:
#                     raise ValueError(f"Invalid bbox format: {detection_bbox}")

#                 best_iou = 0
#                 best_gt_box = None

#                 for coords, true_text in annotation:
#                     try:
#                         true_coords = np.array(coords).reshape((4, 2))  # Normalize annotation bbox
#                         iou_val = calculate_iou(bbox, true_coords)
#                     except Exception as e:
#                         print(f"Error calculating IoU for bbox: {bbox}, coords: {coords}. Error: {e}")
#                         continue

#                     if iou_val > best_iou:
#                         best_iou = iou_val
#                         best_gt_box = true_coords

#                 if best_iou >= iou_threshold:
#                     true_positive += 1
#                     matched_gt_boxes.add(tuple(best_gt_box.flatten()))
#                     distances.append(calculate_distance(bbox, best_gt_box))
#                     iou_scores.append(best_iou)
#                 else:
#                     false_positive += 1

#                 for coords, true_text in annotation:
#                     predicted_text = prediction.get("predicted_text", "").strip().lower()
#                     true_text = true_text.strip().lower()

#                     # CER and WER calculations for full-image bounding boxes
#                     cer_value = safe_cer(true_text, predicted_text)
#                     wer_value = safe_wer(true_text, predicted_text)
#                     if cer_value is not None:
#                         cer_list.append(cer_value)
#                     if wer_value is not None:
#                         wer_list.append(wer_value)

#                     if true_text == predicted_text:
#                         exact_matches += 1

#                 for coords, _ in annotation:
#                     try:
#                         if tuple(np.array(coords).flatten()) not in matched_gt_boxes:
#                             false_negative += 1
#                     except Exception as e:
#                         print(f"Error processing annotation coords: {coords}. Error: {e}")

#             except Exception as e:
#                 print(f"Error processing detection bbox for image: {image_file}. Error: {e}")

#     # Final metrics computation
#     metrics = {}
#     if use_cropped:
#         total_samples = sum(len(ann) for ann in annotations.values())
#         metrics.update({
#             "Avg CER (%)": np.mean(cer_list) * 100 if cer_list else 0,
#             "Avg WER (%)": np.mean(wer_list) * 100 if wer_list else 0,
#             "Exact Match Accuracy (%)": (exact_matches / total_samples) * 100 if total_samples else 0,
#         })
#     else:
#         precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0
#         recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0
#         f1_score = (2 * (precision * recall) / (precision + recall)) if (precision + recall) else 0

#         metrics.update({
#             "Avg IoU (%)": np.mean(iou_scores) * 100 if iou_scores else 0,
#             "Avg Distance Between Centers": np.mean(distances) if distances else 0,
#             "Precision (%)": precision * 100,
#             "Recall (%)": recall * 100,
#             "F1 Score (%)": f1_score * 100,
#             "Avg CER (%)": np.mean(cer_list) * 100 if cer_list else 0,
#             "Avg WER (%)": np.mean(wer_list) * 100 if wer_list else 0,
#             "Exact Match Accuracy (%)": (exact_matches / len(ocr_results["predictions"])) * 100 if len(ocr_results["predictions"]) else 0,
#         })

#     print(f"Metrics calculated: {metrics}")
#     return metrics
