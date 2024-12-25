from libs.libs import *

def apply_ocr(image, framework, coords=None):
    """
    Unified OCR function for different frameworks.
    coords might be [x1, y1, x2, y2, x3, y3, x4, y4]
    or it might be None (full image).
    """
    # If we have polygon coords, convert them to a bounding rect.
    if coords:
        # coords is length 8, i.e. [x1,y1,x2,y2,x3,y3,x4,y4]
        # Reshape to (4,2) and compute bounding rect.
        pts = np.array(coords, dtype=np.int32).reshape((4, 2))
        x, y, w, h = cv2.boundingRect(pts)
        
        # Crop the image
        cropped = image[y:y+h, x:x+w]
        if cropped is None or cropped.size == 0:
            raise ValueError("Invalid or empty cropped image in apply_ocr.")
        image = cropped

    if image is None or image.size == 0:
        raise ValueError("Invalid or empty image passed to apply_ocr.")

    # Handle various frameworks
    if framework == "easy_ocr":
        reader = easyocr.Reader(['vi'])
        return reader.readtext(image)
    elif framework == "paddle_ocr":
        from paddleocr import PaddleOCR
        reader = PaddleOCR()
        return reader.ocr(image)
    elif framework == "tesseract_ocr":
        import pytesseract
        # You can return just a string or bounding boxes from tesseract
        return pytesseract.image_to_string(image)
    else:
        raise ValueError(f"Unknown framework: {framework}")