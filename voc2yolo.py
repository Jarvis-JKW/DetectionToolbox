# -*- coding: utf-8 -*-
import os
import random
import argparse
from tqdm import tqdm
import xml.etree.ElementTree as ET

"""
使用说明:
1. 文件按如下格式放置：
    datasets
    ├── voc2yolo.py
    └── your_dataset
        ├── images
        ├── xmls
        ├── labels
        ├── train.txt (被用来训练的图片路径)
        ├── val.txt (被用以评估训练效果的图片路径)
        └── test.txt (非必需)

2. 在datasets路径下打开终端或Powershell，使用命令:
    python voc2yolo.py --img_path ./your_dataset/path_to_images --xml_path ./your_dataset/path_to_xmls --label_path ./your_dataset/path_to_labels --txt_path ./your_dataset
"""

def splitImg():
    """
    划分训练集、验证集和测试集
    """
    trainval_percent = opt.trainval
    train_percent    = opt.train

    xml_names = os.listdir(opt.xml_path)
    num = len(xml_names)
    xml_indexs = range(num)

    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(xml_indexs, tv)
    train = random.sample(trainval, tr)

    ftrainval = open(opt.txt_path + '/trainval.txt', 'w')
    ftest     = open(opt.txt_path + '/test.txt',     'w')
    ftrain    = open(opt.txt_path + '/train.txt',    'w')
    fval      = open(opt.txt_path + '/val.txt',      'w')

    for idx in xml_indexs:
        xml_name_no_ext = xml_names[idx][:-4] + '\n'
        if idx in trainval:
            ftrainval.write(xml_name_no_ext)
            if idx in train:
                ftrain.write(xml_name_no_ext)
            else:
                fval.write(xml_name_no_ext)
        else:
            ftest.write(xml_name_no_ext)

    ftrainval.close()
    ftest.close()
    ftrain.close()
    fval.close()

    print(print(f"{tr} images for training, {tv-tr} images for validation, {num-tv} images for testing, {num} images in total."))


def normBox(size, box):
    """
    归一化坐标
    """
    scaleW = 1. / size[0]
    scaleH = 1. / size[1]
    centerX = (box[0] + box[1]) / 2.0 - 1
    centerY = (box[2] + box[3]) / 2.0 - 1
    boxW = box[1] - box[0]
    boxH = box[3] - box[2]
    centerX *= scaleW
    centerY *= scaleH
    boxW *= scaleW
    boxH *= scaleH
    return centerX, centerY, boxW, boxH


def cvtLabels(xml_name_no_ext:str):
    fxml = open(f"{opt.xml_path}/{xml_name_no_ext}.xml", encoding="utf-8")
    ftxt = open(f"{opt.label_path}/{xml_name_no_ext}.txt", 'w')

    tree = ET.parse(fxml)
    root = tree.getroot()

    size = root.find("size")
    width  = int(size.find("width").text)
    height = int(size.find("height").text)\

    for obj in root.iter("object"):
        difficult  = obj.find("difficult").text  # 如果被标注difficult将被跳过!!!
        class_name = obj.find("name").text
        if class_name not in class_names or int(difficult):  
            continue
        class_idx = class_names.index(class_name)
        box = obj.find("bndbox")
        box = [float(box.find("xmin").text), float(box.find("xmax").text),
               float(box.find("ymin").text), float(box.find("ymax").text)]

        if box[1] > width:  box[1] = width
        if box[3] > height: box[3] = height

        # Normalization for YOLO
        box = normBox((width, height), box)
        ftxt.write(' '.join([str(class_idx)] + [f"{x:.6f}" for x in box]) + '\n')


if __name__ == '__main__':
    # 列表中的类名应与xml文件一致，将类名移出列表以过滤不需要训练的类, 对应(*.yaml)文件中的names顺序
    class_names = ['kiwi', 'col']

    parser = argparse.ArgumentParser()
    parser.add_argument("--txt_path",   default="./kiwi",         type=str,   help="train.txt, val.txt, test.txt")
    parser.add_argument("--img_path",   default="./kiwi/images",  type=str,   help="图片路径")
    parser.add_argument("--xml_path",   default="./kiwi/xmls",    type=str,   help="Pascal VOC (*.xml) 标签路径")
    parser.add_argument("--label_path", default="./kiwi/labels",  type=str,   help="输出YOLO(*.txt) 标签路径")
    parser.add_argument("--ext_name",   default="jpg",            type=str,   help="图片扩展名")
    parser.add_argument('--trainval',   default=1.0,              type=float, help='test_percent = 1.0 - trainval_percent')
    parser.add_argument('--train',      default=0.9,              type=float, help='val_percent  = trainval_percent * (1.0 - train_percent)')
    parser.add_argument("--sort",       action="store_true",                  help="按数字排序以避免图片名按字符串排序如1.jpg, 10.jpg, 11.jpg")
    opt = parser.parse_args()

    splitImg()

    if not os.path.exists(opt.label_path):
        os.makedirs(opt.label_path)

    work_dir = str(os.getcwd())
    sets = ["train", "val", "test"]
    for image_set in sets:
        try:
            ftxt = open(f"{opt.txt_path}/{image_set}.txt", 'r')

            if opt.sort:
                try:
                    image_names = list(map(int, ftxt.read().strip().split()))
                    image_names.sort()
                    image_names = list(map(str, image_names))
                except ValueError:
                    print(f'''\033[1;31mERROR: Only number-like filename could be sorted.\033[0m''')
                    exit(-1)
            else:
                image_names = ftxt.read().strip().split()
            ftxt.close()


        except FileNotFoundError:
            print(f'''\033[1;33mWARNING: File "{opt.txt_path}/{image_set}.txt" Not Found.\033[0m''')
            continue

        else:
            list_file = open(f"{opt.txt_path}/{image_set}.txt", "w")

            pbar = tqdm(image_names)
            for image_name in pbar:
                list_file.write(f"./images/{image_name}.{opt.ext_name}\n".replace("\\", "/"))
                cvtLabels(image_name)
            pbar.close()

            list_file.close()

    print("Done!")
