# -*- coding: utf-8 -*
import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from xml.etree.ElementTree import Element, SubElement, tostring
import xml.dom.minidom as minidom


def extractJson(json_file:str, class_names:list):
    with open(json_file) as f:
        data = json.load(f)

    imageHeight = data['imageHeight']
    imageWidth = data['imageWidth']
    imagePath = data['imagePath']  # only basename was written by labelme

    bndboxes = []
    for shape in data['shapes']:
        if shape['label'] not in class_names:
            continue
        else:
            points = np.array(shape['points'], dtype=int)
            xmin, xmax = points[:, 0].min(), points[:, 0].max()
            ymin, ymax = points[:, 1].min(), points[:, 1].max()
            if xmax < xmin or ymax < ymin:
                print(f'''\033[1;33m unexpected value in xmin, xmax, ymin, ymax:{xmin, xmax, ymin, ymax}\033[0m''')
                continue
            elif xmin < 0 or xmax > imageWidth:
                print('''\033[1;33m unexpected value in xmin, xmax:{xmin, xmax}\033[0m''')
                continue
            bndboxes.append([shape['label'], xmin, ymin, xmax, ymax])

    return imagePath, (imageHeight, imageWidth), bndboxes


def formatXml(elem):
    rough_string = tostring(elem, "utf-8")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="	")


def writeXml(img_name, img_size, bndboxes:list, xml_file:str):
    annotation = Element("annotation")

    filename = SubElement(annotation, "filename")
    filename.text = img_name

    size = SubElement(annotation, "size")
    width = SubElement(size, "width")
    width.text = str(img_size[1])
    height = SubElement(size, "height")
    height.text = str(img_size[0])
    depth = SubElement(size, "depth")
    depth.text = "3"

    for i in range(len(bndboxes)):
        obj = SubElement(annotation, "object")
    
        name = SubElement(obj, "name")
        name.text = bndboxes[i][0]
    
        bndbox = SubElement(obj, "bndbox")
    
        xmin = SubElement(bndbox, "xmin")
        xmin.text = str(bndboxes[i][1])
    
        ymin = SubElement(bndbox, "ymin")
        ymin.text = str(bndboxes[i][2])
    
        xmax = SubElement(bndbox, "xmax")
        xmax.text = str(bndboxes[i][3])
    
        ymax = SubElement(bndbox, "ymax")
        ymax.text = str(bndboxes[i][4])
    
    with open(xml_file, "w") as f:
        f.write(formatXml(annotation))


if __name__ == "__main__":
    class_names = ['square', 'star']  # Put your class names to be converted

    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", default=None, type=str, help="The directory where your semantic segmentation annotation files (*.JSON) are")
    parser.add_argument("--xml_path",  default=None, type=str, help="The directory where your object detection annotation files (*.xml) are")
    opt = parser.parse_args()

    json_files = os.listdir(opt.json_path)

    if opt.json_path is None:
        raise ValueError("Param json_path cannot be None or NULL!")

    if opt.xml_path is None:
        opt.xml_path = os.path.join(opt.json_path, "xmls")
        print(f'''\033[1;33mWARNING: No specified annotation files directory, set to default: "{opt.xml_path}".\033[0m''')

    if not os.path.exists(opt.xml_path):
        os.makedirs(opt.xml_path)

    pbar = tqdm(json_files)
    for json_file in pbar:
        if json_file.lower().endswith("json"):
            img_name, img_size, bndboxes = extractJson(os.path.join(opt.json_path, json_file), class_names)
            pbar.set_postfix_str(f"Image: {img_name}, size:{img_size}, {len(bndboxes)} instances.")
            writeXml(img_name, img_size, bndboxes, os.path.join(opt.xml_path, json_file.split('.')[0] + ".xml"))

    pbar.close()
    print("Done!")
