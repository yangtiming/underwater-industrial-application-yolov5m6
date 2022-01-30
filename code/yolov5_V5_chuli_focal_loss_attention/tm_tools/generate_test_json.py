import json
import os
from glob import glob
from tqdm import tqdm
from PIL import Image

underwater_classes = ["holothurian", "echinus", "scallop", "starfish"]
label_ids = {name: i + 1 for i, name in enumerate(underwater_classes)}


def save(images, annotations):
    ann = {}
    ann['type'] = 'instances'
    ann['images'] = images
    ann['annotations'] = annotations

    categories = []
    for k, v in label_ids.items():
        categories.append({"name": k, "id": v})
    ann['categories'] = categories
    json.dump(ann, open('./val.json', 'w'))


def test_dataset(im_dir):
    im_list = glob(os.path.join(im_dir, '*.jpg'))
    idx = 1
    image_id = 1
    images = []
    annotations = []
    for im_path in tqdm(im_list):
        
        im = Image.open(im_path)
        w, h = im.size
        image = {'file_name': os.path.basename(im_path), 'width': w, 'height': h, 'id': image_id}
        images.append(image)
        labels = [[10, 10, 20, 20]]
        for label in labels:
            bbox = [label[0], label[1], label[2], label[3]]
            seg = []
            ann = {'segmentation': [seg], 'area': bbox[2] * bbox[3], 'iscrowd': 0, 'image_id': image_id,
                   'bbox': bbox, 'category_id': 1, 'id': idx, 'ignore': 0}
                    
            idx += 1
            annotations.append(ann)
        image_id += 1
    save(images, annotations)


if __name__ == '__main__':
    test_dir = '../../../test/'
    print("generate test json label file.")
    test_dataset(test_dir)