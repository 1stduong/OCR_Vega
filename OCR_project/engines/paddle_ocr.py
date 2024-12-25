from libs.libs import *
from core.ocr_utils import parse_label_files, crop_text_regions
from config.config import image_folder, label_folder, num_files
from paddleocr import PaddleOCR
import os
import cv2
import numpy as np

def apply_paddle_ocr(image_folder, label_folder, num_files, use_cropped=False, mode="normal", framework="paddle_ocr"):
    """
    Applies PaddleOCR to images and optionally draws bounding boxes in test mode.
    """
    # Initialize PaddleOCR
    ocr = PaddleOCR(lang='vi', use_angle_cls=False, show_log=False)

    annotations = parse_label_files(label_folder, num_files)
    results = {"total": 0, "correct": 0, "incorrect": 0, "predictions": []}

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

        print(f"Processing image: {image_file}, Shape: {image.shape}, Dtype: {image.dtype}")

        # Preprocessing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        denoised = cv2.medianBlur(thresh, 3)

        predicted_image = image.copy()
        original_image = image.copy()

        if use_cropped:
            # Cropped Regions Mode
            for coords, true_text in annotation:
                cropped = crop_text_regions(image, coords)
                if cropped is None or cropped.size == 0:
                    print(f"Skipping invalid cropped region for {image_file}")
                    continue

                rec_result = ocr.ocr(cropped, det=False, rec=True, cls=False)

                if rec_result and len(rec_result[0]) > 0:
                    predicted_text, confidence = rec_result[0][0]
                else:
                    predicted_text, confidence = "", 0.0

                true_text = true_text.strip().lower()
                predicted_text = predicted_text.strip().lower()

                print(f"True: {true_text}, Predicted (PaddleOCR): {predicted_text}")

                results["total"] += 1
                if true_text == predicted_text:
                    results["correct"] += 1
                else:
                    results["incorrect"] += 1

                results["predictions"].append({
                    "image_file": image_file,
                    "true_text": true_text,
                    "predicted_text": predicted_text,
                    "confidence": confidence
                })

        else:
            # Full-Image Mode
            ocr_result = ocr.ocr(denoised, det=True, rec=True, cls=False)

            for line in ocr_result:
                bbox_points = line[0]
                (text, confidence) = line[1]

                print(f"Bounding Box: {bbox_points}, Predicted Text: {text}, Confidence: {confidence}")

                if mode == "test":
                    pts = np.array(bbox_points, dtype=np.int32)
                    cv2.polylines(predicted_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                    text_x, text_y = pts[0]
                    cv2.putText(
                        predicted_image,
                        text,
                        (int(text_x), int(text_y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA
                    )

                results["total"] += 1
                results["predictions"].append({
                    "image_file": image_file,
                    "bbox": bbox_points,
                    "predicted_text": text,
                    "confidence": confidence
                })

            if mode == "test":
                predicted_out_name = f"test/{framework}_predicted_bboxes_{image_file}"
                predicted_out_path = os.path.join("/home/nhduong141103/VegaCop/OCR/OCR_project", predicted_out_name)
                cv2.imwrite(predicted_out_path, predicted_image)
                print(f"Saved predicted bounding box image to: {predicted_out_path}")

                for coords, text in annotation:
                    pts = np.array(coords, dtype=np.int32).reshape((4, 2))
                    cv2.polylines(original_image, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
                original_out_name = f"test/original_bboxes_{image_file}"
                original_out_path = os.path.join("/home/nhduong141103/VegaCop/OCR/OCR_project", original_out_name)
                cv2.imwrite(original_out_path, original_image)
                print(f"Saved original bounding box image to: {original_out_path}")

    return results