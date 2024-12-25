from libs.libs import *

use_cropped=True

# File paths
image_folder = "/home/nhduong141103/VegaCop/OCR/OCR_project/data/images/"
label_folder = "/home/nhduong141103/VegaCop/OCR/OCR_project/data/labels/"
test_folder = "/home/nhduong141103/VegaCop/OCR/OCR_project/test/"
font_path = "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"

#File count
num_files = 5 #len([f for f in os.listdir(image_folder) if f.endswith('.jpg')])