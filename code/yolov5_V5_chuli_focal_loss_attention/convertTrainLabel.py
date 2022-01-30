import xml.etree.ElementTree as ET
import os
import shutil
import random


classes = ["holothurian", "echinus", "scallop", "starfish"]  # 类别
def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_id, img_dir , box_dir):
    in_file = open('tcdata/train/box/%s.xml' % (image_id))
    out_file = open('%s/%s.txt' % (box_dir, image_id), 'w')  # 生成txt格式文件
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    shutil.copy(os.path.join(image_dir,image_id+".jpg"), os.path.join(img_dir,image_id+".jpg"))




image_dir = "tcdata/train/image/"
box_dir = "tcdata/train/box/"


train_img_dir = "user_data/tmp_data/images/train"
train_box_dir = "user_data/tmp_data/labels/train"
val_img_dir = "user_data/tmp_data/images/val"
val_box_dir = "user_data/tmp_data/labels/val"


if not os.path.exists(train_img_dir):
    os.makedirs(train_img_dir)

if not os.path.exists(train_box_dir):
    os.makedirs(train_box_dir)

if not os.path.exists(val_img_dir):
    os.makedirs(val_img_dir)

if not os.path.exists(val_box_dir):
    os.makedirs(val_box_dir)


id_list = os.listdir(image_dir)
flod = 0
random.shuffle(id_list)

for img_id in id_list:
    print(img_id.split(".")[0])
    if flod % 9 == 0:
        convert_annotation(img_id.split(".")[0], val_img_dir, val_box_dir)
        flod = 0
    else:
        convert_annotation(img_id.split(".")[0], train_img_dir, train_box_dir)

    flod += 1


#
# image_ids_train = open(
#     '/Users/youxinlin/Desktop/datasets/dataset-floats/ImageSets/Main/test.txt').read().strip().split()  # list格式只有000000 000001
#
# # image_ids_val = open('/home/*****/darknet/scripts/VOCdevkit/voc/list').read().strip().split()
#
#
# list_file_train = open('test.txt', 'w')
# # list_file_val = open('val.txt', 'w')
#
#
# for image_id in image_ids_train:
#     list_file_train.write('../../tcdata/train/image/dataset-floats/%s.jpg\n' % (image_id))
#     convert_annotation(image_id)