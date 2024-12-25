from engines.easy_ocr import apply_easy_ocr
from engines.paddle_ocr import apply_paddle_ocr
from engines.tesseract_ocr import apply_tesseract_ocr
from config.config import image_folder, label_folder, num_files
from core.ocr_utils import process_pipeline

def main():
    # Ask user for the OCR framework
    framework = input("Enter framework (easy_ocr/paddle_ocr/tesseract_ocr): ").strip().lower()

    # Map framework to appropriate function
    if framework == "easy_ocr":
        apply_ocr_fn = apply_easy_ocr
    elif framework == "paddle_ocr":
        apply_ocr_fn = apply_paddle_ocr
    elif framework == "tesseract_ocr":
        apply_ocr_fn = apply_tesseract_ocr
    else:
        print(f"Unknown framework: {framework}")
        return

    # Ask user if they want to apply cropped text regions
    cropped_choice = input("Apply cropped regions? (y/n): ").strip().lower()
    if cropped_choice == 'y':
        use_cropped = True
        mode = "normal"  # Automatically set to normal for cropped
    else:
        use_cropped = False
        # Ask user for 'mode': normal or test
        mode = input("Enter mode (normal/test): ").strip().lower()

    # Define a wrapper to pass mode to the OCR function
    def apply_ocr_wrapper(img_folder, lbl_folder, n_files, use_crop):
        return apply_ocr_fn(
            image_folder=img_folder,
            label_folder=lbl_folder,
            num_files=n_files,
            use_cropped=use_crop,
            mode=mode  # Pass mode explicitly
        )

    # Run the pipeline
    process_pipeline(
        framework=framework,
        image_folder=image_folder,
        label_folder=label_folder,
        num_files=num_files,
        apply_ocr_fn=apply_ocr_wrapper,
        use_cropped=use_cropped
    )

if __name__ == "__main__":
    main()
