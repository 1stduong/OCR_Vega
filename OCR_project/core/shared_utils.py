from libs.libs import *

def parse_label_files(label_folder, num_files):
    """
    Parse the label files and return annotations in a dict:
    {
       "label_file_1.txt": [
          ([x1,y1,x2,y2,x3,y3,x4,y4], text1),
          ([x1,y1,x2,y2,x3,y3,x4,y4], text2),
          ...
       ],
       "label_file_2.txt": [...],
       ...
    }
    """
    annotations = {}
    label_files = natsorted(os.listdir(label_folder))[:num_files]  # limit by num_files
    for label_file in label_files:
        file_path = os.path.join(label_folder, label_file)
        with open(file_path, 'r', encoding='utf-8') as file:
            annotation = []
            for line in file:
                data = line.strip().split(',')
                coords = list(map(int, data[:8]))  # first 8 values are bounding box coords
                text = data[8]  # 9th value is the text
                annotation.append((coords, text))
        annotations[label_file] = annotation
    return annotations