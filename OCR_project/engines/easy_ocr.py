from libs.libs import *
from core.ocr_utils import parse_label_files, crop_text_regions
from config.config import image_folder, label_folder, num_files
import os

def apply_easy_ocr(image_folder, label_folder, num_files, use_cropped=False, mode="normal", framework="easy_ocr"):
    """
    Applies EasyOCR to images and optionally draws bounding boxes in test mode.
    Includes preprocessing directly in the function.
    """
    reader = easyocr.Reader(['vi'])
    annotations = parse_label_files(label_folder, num_files)
    results = {"total": 0, "correct": 0, "incorrect": 0, "predictions": []}

    for idx, (label_file, annotation) in enumerate(annotations.items()):
        image_file = f"im{idx + 1:04d}.jpg"
        image_path = os.path.join(image_folder, image_file)

        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        print(f"Processing image: {image_file}, Shape: {image.shape}, Dtype: {image.dtype}")

        # Preprocessing
        print(f"Applying preprocessing to image: {image_file}")
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Otsu's thresholding
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Denoise with median blur
        denoised = cv2.medianBlur(thresh, 3)

        predicted_image = image.copy()
        original_image = image.copy()

        if not use_cropped:
            # Full-Image Mode
            result = reader.readtext(denoised)

            for detection in result:
                bbox, text, confidence = detection
                print(f"Bounding Box: {bbox}, Predicted Text: {text}, Confidence: {confidence}")

                # Draw bounding boxes only in test mode
                if mode == "test":
                    # Predicted bounding boxes
                    pts = np.array(bbox, dtype=np.int32)
                    cv2.polylines(predicted_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                    # Add predicted text
                    text_x, text_y = pts[0]
                    cv2.putText(
                        predicted_image,
                        text,
                        (text_x, text_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA
                    )

            # Save images only in test mode
            if mode == "test":
                # Save predicted bounding boxes
                predicted_out_name = f"test/{framework}_predicted_bboxes_{image_file}"
                predicted_out_path = os.path.join("/home/nhduong141103/VegaCop/OCR/OCR_project", predicted_out_name)
                cv2.imwrite(predicted_out_path, predicted_image)
                print(f"Saved predicted bounding box image to: {predicted_out_path}")

                # Save original bounding boxes
                for coords, text in annotation:
                    pts = np.array(coords, dtype=np.int32).reshape((4, 2))
                    cv2.polylines(original_image, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
                original_out_name = f"test/original_bboxes_{image_file}"
                original_out_path = os.path.join("/home/nhduong141103/VegaCop/OCR/OCR_project", original_out_name)
                cv2.imwrite(original_out_path, original_image)
                print(f"Saved original bounding box image to: {original_out_path}")

    return results
