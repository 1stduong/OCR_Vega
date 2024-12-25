from libs.libs import *
from core.ocr_utils import parse_label_files, crop_text_regions
from config.config import image_folder, label_folder, num_files
import pytesseract
from pytesseract import Output

def apply_tesseract_ocr(image_folder, label_folder, num_files, use_cropped=False, mode="normal", framework="tesseract_ocr"):
    """
    Applies Tesseract OCR to images and optionally saves images in test mode.
    Includes Vietnamese language support.
    """
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
        print(f"Applying preprocessing to image: {image_file}")
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply Otsu's thresholding
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Denoise with median blur
        denoised = cv2.medianBlur(thresh, 3)
        predicted_image = image.copy()
        original_image = image.copy()

        if use_cropped:
            # Cropped Regions Mode
            for coords, true_text in annotation:
                x, y, w, h = cv2.boundingRect(np.array(coords).reshape(4, 2))
                cropped = image[y:y+h, x:x+w]

                if cropped is None or cropped.size == 0:
                    print(f"Skipping invalid cropped region for {image_file}")
                    continue

                # Specify Vietnamese language
                predicted_text = pytesseract.image_to_string(cropped, lang='vie').strip().lower()
                true_text = true_text.strip().lower()

                print(f"True: {true_text}, Predicted (Tesseract): {predicted_text}")

                # Update statistics
                results["total"] += 1
                if true_text == predicted_text:
                    results["correct"] += 1
                else:
                    results["incorrect"] += 1

                # Append prediction info
                results["predictions"].append({
                    "image_file": image_file,
                    "true_text": true_text,
                    "predicted_text": predicted_text
                })

        else:
            # Full-Image Mode
            print(f"Running Tesseract on preprocessed full image: {image_file}")
            # Specify Vietnamese language
            data = pytesseract.image_to_data(denoised, lang='vie', output_type=Output.DICT)
            print(f"Raw Tesseract Output for {image_file}: {data}")
            n_boxes = len(data['level'])

            for i in range(n_boxes):
                confidence = int(data['conf'][i])
                text = data['text'][i].strip()

                # Skip low-confidence or empty detections
                if confidence < 50 or not text:
                    print(f"Skipping low-confidence or empty text: Text='{text}', Confidence={confidence}")
                    continue

                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                print(f"Bounding Box: ({x}, {y}, {w}, {h}), Text: {text}, Confidence: {confidence}")

                # Draw bounding boxes only in test mode
                if mode == "test":
                    cv2.rectangle(predicted_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        predicted_image,
                        text,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA
                    )

                # Append predictions to results
                results["total"] += 1
                results["predictions"].append({
                    "image_file": image_file,
                    "bbox": [(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
                    "predicted_text": text,
                    "confidence": confidence
                })

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