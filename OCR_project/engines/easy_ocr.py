from libs.libs import *
from core.ocr_utils import *
from config.config import *

def easy_ocr(image, labels=None, mode="original", draw_boxes=False, save_path=None):
    """
    Applies EasyOCR to a preprocessed image.

    Args:
        image (numpy.ndarray): The preprocessed image.
        labels (list): Ground truth labels. For "original" mode, it's the full text. For "cropped," it's a single label.
        mode (str): Mode of operation ("original" or "cropped").
        draw_boxes (bool): Whether to draw predicted bounding boxes (only for "original").
        save_path (str): Path to save the image with drawn bounding boxes (only for "original").

    Returns:
        dict: OCR results. Structure depends on the mode:
              For "original" mode: {
                  "image_file": str,
                  "ground_truth": str,
                  "predicted_text": str,
                  "predicted_boxes": list of [bbox, text, confidence],
              }
              For "cropped" mode: {
                  "image_file": str,
                  "sub_image_file": str,
                  "ground_truth": str,
                  "predicted_text": str,
              }
    """
    reader = easyocr.Reader(['vi'])

    if mode == "original":
        results = {
            "image_file": None,
            "ground_truth": "",
            "predicted_text": "",
            "predicted_boxes": []
        }

        # Apply EasyOCR
        ocr_results = reader.readtext(image, detail=1)
        for bbox, text, confidence in ocr_results:
            text = text.lower()
            results["predicted_boxes"].append({"bbox": bbox, "text": text, "confidence": confidence})
            results["predicted_text"] += f"{text} "

        # Set ground truth
        if labels:
            results["ground_truth"] = " ".join([label.lower() for _, label in labels])

        # Draw bounding boxes if requested
        if draw_boxes:
            for bbox, text, confidence in ocr_results:
                pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                text_x, text_y = int(pts[0][0][0]), int(pts[0][0][1]) - 10
                cv2.putText(image, f"{text} ({confidence:.2f})", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                
        # Save the image with bounding boxes
        if save_path:
            cv2.imwrite(save_path, image)
            print(f"Saved image with bounding boxes to {save_path}")

        return results

    elif mode == "cropped":
        results = {
            "ground_truth": "",
            "predicted_text": ""
        }

        # Apply EasyOCR
        ocr_results = reader.readtext(image, detail=0)
        predicted_text = ocr_results[0].lower() if ocr_results else ""

        # Store results
        results["ground_truth"] = labels.lower() if labels else ""
        results["predicted_text"] = predicted_text

        return results

    else:
        raise ValueError("Invalid mode. Choose 'original' or 'cropped'.")