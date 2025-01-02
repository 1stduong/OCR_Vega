from core.ocr_utils import *
from config.config import *
from engines.easy_ocr import easy_ocr
from engines.tesseract_ocr import tesseract_ocr
from core.metrics import calculate_metrics

def main():
    mode = input("Select mode (original/cropped): ").strip().lower()
    if mode not in {"original", "cropped"}:
        print("Invalid choice. Please select 'original' or 'cropped'.")
        return

    framework = input("Select OCR framework (easyocr/tesseract): ").strip().lower()
    if framework not in {"easyocr", "tesseract"}:
        print("Invalid choice. Please select 'easyocr' or 'tesseract'.")
        return

    save_choice = input("Do you want to save images? (yes/no): ").strip().lower()
    if save_choice not in {"yes", "no"}:
        print("Invalid choice. Defaulting to no.")
        save_choice = "no"

    # Set save path based on user choice
    output_folder = output_folder_original if mode == "original" else output_folder_cropped
    save_path = output_folder if save_choice == "yes" else None

    ground_truths = []
    predictions = []

    if mode == "original":
        annotations = parse_label_files(label_folder, num_files)
        for image_file, bbox_labels in annotations.items():
            image_path = os.path.join(image_folder, image_file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_file}")
                continue

            # Preprocess the image
            processed_image = preprocess_image(image, preprocess_type="original")

            # Determine the save path for this image
            image_save_path = (
                os.path.join(save_path, f"predicted_{image_file}")
                if save_path
                else None
            )

            # Apply OCR
            ocr_function = easy_ocr if framework == "easyocr" else tesseract_ocr
            ocr_results = ocr_function(
                processed_image,
                labels=bbox_labels,
                mode="original",
                draw_boxes=True if save_choice == "yes" else False,
                save_path=image_save_path
            )
            ground_truths.append(ocr_results["ground_truth"])
            predictions.append(ocr_results["predicted_text"])

    elif mode == "cropped":
        cropped_data = crop_images_from_folder(preprocess=True)
        for image_file, crops in cropped_data.items():
            for idx, (processed_crop, label) in enumerate(crops):
                # Determine the save path for cropped images
                sub_image_save_path = (
                    os.path.join(save_path, f"{os.path.splitext(image_file)[0]}_crop_{idx}.jpg")
                    if save_path
                    else None
                )

                # Apply OCR to cropped region
                ocr_function = easy_ocr if framework == "easyocr" else tesseract_ocr
                ocr_results = ocr_function(
                    processed_crop,
                    labels=label,
                    mode="cropped",
                    save_path=sub_image_save_path
                )
                ground_truths.append(ocr_results["ground_truth"])
                predictions.append(ocr_results["predicted_text"])

                # Save cropped images if save_path is valid
                if sub_image_save_path:
                    cv2.imwrite(sub_image_save_path, processed_crop)
                    print(f"Saved cropped image to {sub_image_save_path}")

    # Calculate and display metrics
    metrics = calculate_metrics(ground_truths, predictions)
    print("\nMetrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.2f}%")

if __name__ == "__main__":
    main()