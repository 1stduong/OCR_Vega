from libs.libs import *
#from core.metrics import monitor_resources, calculate_metrics_with_results
from config.config import *
#from engines.ocr_engine import apply_ocr

def parse_label_files(label_folder, num_files):
    """
    Function;
        Parses a label file to extract bounding boxes and corresponding labels.
        
    Args:
        label_file (str): Path to the label file.
        
    Returns:
        annotations (dict): A dictionary where keys are image filenames and values are lists of bounding boxes and labels.
    Example: gt_1.txt -> im0001.jpg
    """
    
    if not os.path.exists(label_folder):
        raise FileNotFoundError(f"Folder not found: {label_folder}")
    
    annotations = {}
    label_files = natsorted(os.listdir(label_folder))[:num_files]  # limit by num_files

    #Skip not related format files
    for label_file in label_files:
        if not label_file.startswith("gt_") or not label_file.endswith(".txt"):
            print(f"Skipping unrelated file: {label_file}")
            continue
        
        #Extract numeric part and map to image filename
        try:
            file_index = int(label_file.split('_')[1].split('.')[0]) # Extract number from gt_1.txt
            image_file = f"im{file_index:04d}.jpg"  # Format as im0001.jpg
        except ValueError as e:
            print(f"Invalid label file name format: {label_file}. Error: {e}")
            continue
        
        #Parse label file
        file_path = os.path.join(label_folder, label_file)
        with open(file_path, 'r', encoding='utf-8') as file:
            annotation = []
            for line_num, line in enumerate(file, start=1):
                data = line.strip().split(',')
                if len(data) < 9:
                    print(f"Skipping invalid line in {label_file}: {line}")
                    continue
                try:
                    coords = list(map(int, data[:8]))  # First 8 values are bounding box coords
                    text = data[8]  # 9th value is the text
                    annotation.append((coords, text))
                except ValueError as e:
                    print(f"Error parsing line {line_num} in {label_file}: {e}")
        if annotation:
            annotations[image_file] = annotation
        else:
            print(f"No valid annotations found in {label_file}")
            
    #print(f"Parsed annotations: {annotations.keys()}")
    return annotations

def preprocess_image(image, preprocess_type="original"):
    """
    Preprocess the image for OCR.

    Args:
        image (numpy.ndarray): The input image to preprocess.
        preprocess_type (str): The type of preprocessing. Options are "original" or "cropped".

    Returns:
        numpy.ndarray: The preprocessed image.
    """
    if preprocess_type == "original":
        # Preprocessing for original images (full image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        denoised = cv2.medianBlur(thresh, 3)  # Remove noise
        return denoised
    elif preprocess_type == "cropped":
        # Preprocessing for cropped images (focused text regions)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (128, 32), interpolation=cv2.INTER_AREA)  # Normalize size
        normalized = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX)
        return normalized
    else:
        raise ValueError(f"Unknown preprocess_type: {preprocess_type}")

def crop_images_from_folder(preprocess=True, output_folder=None):
    """
    Splits an image into multiple cropped images based on bounding boxes in the label file
    and preprocesses the cropped images if required.

    Args:
        preprocess (bool): Whether to preprocess cropped images.
        output_folder (str): Optional path to save the cropped and processed images.

    Returns:
        dict: A dictionary where keys are image filenames and values are lists of processed cropped images with labels.
    """
    cropped_data = {}
    total_crops = 0
    total_labels = 0

    annotations = parse_label_files(label_folder, num_files)
    #print(f"Parsed annotations: {annotations.keys()}")

    for image_file, bbox_labels in annotations.items():
        image_path = os.path.join(image_folder, image_file)
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_file}")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        image_crops = []

        for idx, (bbox, label) in enumerate(bbox_labels):
            pts = np.array(bbox, dtype=np.int32).reshape((4, 2))
            rect = cv2.boundingRect(pts)
            x, y, w, h = rect
            x = max(0, x)
            y = max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)

            cropped = image[y:y+h, x:x+w]
            if cropped.size == 0:
                print(f"Skipping empty crop for bounding box {idx} in {image_file}")
                continue

            # Preprocess the cropped image if preprocess is enabled
            if preprocess:
                cropped = preprocess_image(cropped, preprocess_type="cropped")

            image_crops.append((cropped, label))
            total_crops += 1

            if output_folder:
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                label_safe = label.replace(" ", "_").replace("/", "_")
                output_path = os.path.join(output_folder, f"{image_file.split('.')[0]}_{idx}_{label_safe}.jpg")
                cv2.imwrite(output_path, cropped)

        cropped_data[image_file] = image_crops
        total_labels += len(bbox_labels)
        print(f"Processed {len(image_crops)} crops for {image_file}")

    print("\nSummary:")
    print(f"Total cropped images: {total_crops}")
    print(f"Total labels processed: {total_labels}")

    return cropped_data