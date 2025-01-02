from libs.libs import *
from core.ocr_utils import *
from config.config import *

def tesseract_ocr(image, labels=None, mode="original", draw_boxes=False, save_path=None):
    """
    Applies Tesseract OCR to a preprocessed image and optionally draws bounding boxes.

    Args:
        image (numpy.ndarray): The preprocessed image.
        labels (list): Ground truth labels (used for validation).
        mode (str): Mode of operation ("original" or "cropped").
        draw_boxes (bool): Whether to draw predicted bounding boxes on the image.
        save_path (str): Path to save the image with bounding boxes. If None, the image is not saved.

    Returns:
        dict: OCR results.
    """
    results = {
        "image_file": None,
        "ground_truth": "",
        "predicted_text": "",
        "predicted_boxes": []
    }

    if mode == "original":
        # Apply Tesseract OCR
        ocr_results = pytesseract.image_to_data(image, lang='vie', output_type=Output.DICT)
        n_boxes = len(ocr_results['text'])
        for i in range(n_boxes):
            text = ocr_results['text'][i].strip().lower()
            confidence = int(ocr_results['conf'][i])
            if text and confidence > 0:  # Filter out low-confidence or empty results
                x, y, w, h = (
                    ocr_results['left'][i],
                    ocr_results['top'][i],
                    ocr_results['width'][i],
                    ocr_results['height'][i]
                )
                bbox = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
                results["predicted_boxes"].append({"bbox": bbox, "text": text, "confidence": confidence})
                results["predicted_text"] += f"{text} "

        # Add ground truth if labels are provided
        if labels:
            results["ground_truth"] = " ".join([label.lower() for _, label in labels])

        # Draw bounding boxes if required
        if draw_boxes:
            for box in results["predicted_boxes"]:
                x1, y1 = int(box["bbox"][0][0]), int(box["bbox"][0][1])
                x2, y2 = int(box["bbox"][2][0]), int(box["bbox"][2][1])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
                cv2.putText(
                    image, f"{box['text']} ({box['confidence']}%)",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                )

            # Save the image only if save_path is provided
            if save_path:
                cv2.imwrite(save_path, image)
                print(f"Saved image with bounding boxes to {save_path}")

    elif mode == "cropped":
        # Cropped mode doesn't require bounding boxes
        predicted_text = pytesseract.image_to_string(image, lang='vie').strip().lower()
        results["ground_truth"] = labels.lower() if labels else ""
        results["predicted_text"] = predicted_text

    return results