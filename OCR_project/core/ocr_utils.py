from libs.libs import *
from core.metrics import monitor_resources, calculate_metrics
from config.config import *
from core.shared_utils import *
from engines.ocr_engine import apply_ocr

# Generalized OCR Application
def crop_text_regions(image, coords):
    """Crop text regions using bounding box coordinates."""
    h_img, w_img, _ = image.shape
    pts = np.array(coords, dtype=np.int32).reshape((4, 2))
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect

    # Clamp coordinates to fit within the image
    x = max(0, x)
    y = max(0, y)
    w = min(w, w_img - x)
    h = min(h, h_img - y)

    # Validate crop dimensions
    if w <= 0 or h <= 0:
        print(f"Invalid crop dimensions: x={x}, y={y}, w={w}, h={h}")
        return None

    cropped = image[y:y+h, x:x+w]
    return cropped

@monitor_resources
def process_pipeline(framework, image_folder, label_folder, num_files, apply_ocr_fn, use_cropped=False, iou_threshold=0.3):
    """
    Generalized pipeline to run OCR and calculate metrics.

    Args:
        framework (str): Name of the OCR framework (e.g., 'easyocr', 'paddleocr', 'tesseract').
        image_folder (str): Path to images.
        label_folder (str): Path to labels.
        num_files (int): Number of files to process.
        apply_ocr_fn (callable): OCR function specific to the framework.
        use_cropped (bool): Whether to use cropped regions or full images.
        iou_threshold (float): IoU threshold for bounding box metrics.
    """
    print(f"Running pipeline for {framework}...")

    # Step 1: Apply OCR
    print(f"Step 1: Applying {framework} OCR...")
    ocr_results = apply_ocr_fn(image_folder, label_folder, num_files, use_cropped)
    if ocr_results is None:
        print("OCR results are empty. Skipping metric calculations.")
        return

    print(f"OCR results for {framework}: {ocr_results}")

    # Step 2: Calculate Metrics
    print("Step 2: Calculating Metrics...")
    metrics = calculate_metrics(
        image_folder=image_folder,
        label_folder=label_folder,
        num_files=num_files,
        use_cropped=use_cropped,
        iou_threshold=iou_threshold,
        framework=framework  # Explicit keyword argument
    )
    print(f"Metrics for {framework}: {metrics}")

    return metrics