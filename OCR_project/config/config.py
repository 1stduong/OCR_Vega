from libs.libs import *

# File paths
image_folder = "/home/nhduong141103/VegaCop/OCR/OCR_project/data/images/"
label_folder = "/home/nhduong141103/VegaCop/OCR/OCR_project/data/labels_preprocess/"
output_folder_cropped = "/home/nhduong141103/VegaCop/OCR/OCR_project/data/output_cropped/"
output_folder_original = "/home/nhduong141103/VegaCop/OCR/OCR_project/data/output_original/"
font_path = "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"

#File count
num_files = 10 #len([f for f in os.listdir(image_folder) if f.endswith('.jpg')])