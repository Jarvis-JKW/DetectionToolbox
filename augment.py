# -*- coding: utf-8 -*
import os
import argparse
import numpy as np
from json import dumps
from tqdm import tqdm
from shutil import copyfile
from xml.dom import minidom
from PIL import Image, ImageEnhance
from xml.etree.ElementTree import Element, SubElement, tostring, parse

"""
Augment your dataset
For normal YOLO object detection
Run command python augment.py --img_dir './path/to/images' --labels_dir './path/to/annotations' --opt_dir './path/to/output' --bright --contrast --color  --fliplr --fliptb --rot90 --rot180 --rot270  --rotany --angle 45
"""

FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1

# Counter clock
ROTATE_90 = 2
ROTATE_180 = 3
ROTATE_270 = 4


def rotate(img, root, angle, fill_color=(128, 128, 128)):
    """
    Rotate the image at any angle but keep the canvas.
    :param img: PIL Image object.
    :param root: .
    :param angle: Angle in degrees (counter-clockwise).
    :param fill_color: Color to fill empty spaces after rotation.
    :return img_rotated: PIL Image object.
    :return bboxes: [[class_name:str, x_min:str(int), y_min:str(int), x_max:str(int), y_max:str(int)]]
    """
    # Get image size
    width, height = img.size

    # extract bounding boxes from xml root
    bboxes = []
    class_names = []
    for obj in root.iter("object"):
        class_names.append(obj.find("name").text)
        box = obj.find("bndbox")
        bboxes.append([float(box.find("xmin").text), float(box.find("xmax").text),
                       float(box.find("ymin").text), float(box.find("ymax").text)])

    # Rotate the image
    img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=fill_color)

    # Convert to radians and compute
    theta = np.radians(angle)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    bboxes = np.array(bboxes)

    # Calculate by considering the point of rotation center as the original point
    bboxes[:, :2] -= width / 2
    bboxes[:, 2:] -= height / 2

    # Formula to follow
    # x_new = x_norm * cos \theta + y_norm * sin \theta + width / 2
    # y_new = y_norm * cos \theta - x_norm * sin \theta + height / 2
    top_left = np.stack([bboxes[:, 0] * cos_theta + bboxes[:, 2] * sin_theta + width / 2,
                         bboxes[:, 2] * cos_theta - bboxes[:, 0] * sin_theta + height / 2], axis=1)

    top_right = np.stack([bboxes[:, 1] * cos_theta + bboxes[:, 2] * sin_theta + width / 2,
                          bboxes[:, 2] * cos_theta - bboxes[:, 1] * sin_theta + height / 2], axis=1)

    btm_left = np.stack([bboxes[:, 0] * cos_theta + bboxes[:, 3] * sin_theta + width / 2,
                         bboxes[:, 3] * cos_theta - bboxes[:, 0] * sin_theta + height / 2], axis=1)
            
    btm_right = np.stack([bboxes[:, 1] * cos_theta + bboxes[:, 3] * sin_theta + width / 2,
                          bboxes[:, 3] * cos_theta - bboxes[:, 1] * sin_theta + height / 2], axis=1)

    stacks = np.stack([top_left, top_right, btm_left, btm_right], axis=1)

    x_min = np.min(stacks[:, :, 0], axis=1)
    x_max = np.max(stacks[:, :, 0], axis=1)
    y_min = np.min(stacks[:, :, 1], axis=1)
    y_max = np.max(stacks[:, :, 1], axis=1)

    # Add limits for new bounding boxes
    x_min[x_min < 0] = 0
    x_max[x_max > width - 1] = width - 1
    y_min[y_min < 0] = 0
    y_max[y_max > height - 1] = height - 1

    # Delete the bounding boxes out of the canva.
    del_idx = []
    del_idx += np.where(x_min > width - 1)[0].tolist()
    del_idx += np.where(x_max < 0)[0].tolist()
    del_idx += np.where(y_min > height - 1)[0].tolist()
    del_idx += np.where(y_max < 0)[0].tolist()

    bboxes = np.stack([x_min, y_min, x_max, y_max], axis=1).astype(int)
    bboxes = np.delete(bboxes, list(set(del_idx)), axis=0)

    # Add class_names to bboxes
    bboxes = [[class_name, *bbox] for class_name, bbox in zip(class_names, bboxes)]

    return img, bboxes


def change_brightness(img):
    """
    Change the image brightness according to the given factor, or a random factor between the limits. Default limits: [0.7, 1.3].
    :param img: PIL.Image object.
    :return: Augmented PIL.Image object.
    """
    factor = np.random.uniform(opt.brt_lower, opt.brt_upper) if opt.brt_factor is None else opt.brt_factor
    return ImageEnhance.Brightness(img).enhance(factor)


def change_contrast(img):
    """
    Change the image contrast according to the given factor, or a random factor between the limits. Default limits: [0.7, 1.3].
    :param img: PIL.Image object.
    :return: Augmented PIL.Image object.
    """
    factor = np.random.uniform(opt.ctrs_lower, opt.ctrs_upper) if opt.ctrs_factor is None else opt.ctrs_factor
    return ImageEnhance.Contrast(img).enhance(factor)


def change_color(img):
    """
    Change the image color according to the given factor, or a random factor between the limits. Default limits: [0.7, 1.3].
    :param img: PIL.Image object.
    :return: Augmented PIL.Image object.
    """
    factor = np.random.uniform(opt.color_lower, opt.color_upper) if opt.color_factor is None else opt.color_factor
    return ImageEnhance.Color(img).enhance(factor)


def transpose(img, root, method:int):
    """
    Flip or rotate the image canva and bounding boxes.
    :param img: PIL.Image object.
    :param root: Root element of ElementTree of annotation file(*.xml)
    :param method: FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM, ROTATE_90, ROTATE_180, ROTATE_270
    :return img: Augmented PIL.Image object
    :return bboxes: [[class_name, xmin, ymin, xmax, ymax], ...]
    :return size: (height, width)
    """
    size = root.find("size")
    width  = int(size.find("width").text)
    height = int(size.find("height").text)

    # In some annotations, the size of images were annotated to 0
    # The following codes to cope with the situations
    if width == 0 or height == 0:
        print(f'''\033[1;33mWARNING: Annotations file is abnormal: "width: {width}, height: {height}. Attempt to correct using image".\033[0m''')
        width, height = img.size

    # Extract bounding boxes
    bboxes = []
    class_names = []
    for obj in root.iter("object"):
        class_names.append(obj.find("name").text)
        box = obj.find("bndbox")
        bboxes.append([float(box.find("xmin").text), float(box.find("xmax").text),
                       float(box.find("ymin").text), float(box.find("ymax").text)])

    # Transpose the image
    img = img.transpose(method)

    # Convert to NumPy array to calculate
    bboxes = np.array(bboxes)

    if method == FLIP_LEFT_RIGHT:
        # x_min' = width - x_max, x_max' = width - x_min
        bboxes[:, 0], bboxes[:, 1] = width - bboxes[:, 1], width - bboxes[:, 0]

    elif method == FLIP_TOP_BOTTOM:
        # y_min' = height - y_max, y_max' = height - y_min
        bboxes[:, 2], bboxes[:, 3] = height - bboxes[:, 3], height - bboxes[:, 2]

    elif method == ROTATE_90:
        # x_min' = y_min, x_max' = y_max
        # y_min' = width - x_max, y_max' = width - x_min
        bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3] = bboxes[:, 2], bboxes[:, 3], width - bboxes[:, 1], width - bboxes[:, 0]
        width, height = img.size

    elif method == ROTATE_180:
        # x_min' = width - x_max, x_max' = width - x_min
        # y_min' = height - y_max, y_max' = height - y_min
        bboxes[:, 0], bboxes[:, 1] = width - bboxes[:, 1], width - bboxes[:, 0]
        bboxes[:, 2], bboxes[:, 3] = height - bboxes[:, 3], height - bboxes[:, 2]

    elif method == ROTATE_270:
        # x_min' = height - y_max, x_max' = height - y_min
        # y_min' = x_min, y_max' = x_max
        # WARNING: this code lead a bug, found and fixed by Jarvis_jkw, 12/19/2024.
        # bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3] = height - bboxes[:, 3], height - bboxes[:, 2], bboxes[:, 0], bboxes[:, 1]
        # Unexpected result: y_min' = x_min' = height - y_max, y_max' = x_max' = height - y_min
        bboxes[:, 2], bboxes[:, 3], bboxes[:, 0], bboxes[:, 1] = bboxes[:, 0], bboxes[:, 1], height - bboxes[:, 3], height - bboxes[:, 2]
        width, height = img.size

    # [[x_min, x_max, y_min. y_max] * n] -> [[class_name, x_min, y_min, x_max, y_max]]
    bboxes = bboxes[:, [0, 2, 1, 3]].astype(int)
    bboxes = bboxes.tolist()
    bboxes = [[class_name, *bbox] for class_name, bbox in zip(class_names, bboxes)]

    return img, bboxes, (height, width)


def augment_one(img_path, label_path):
    """
    Augment one image.
    :param img_path: Image path of image to augment.
    :param label_path: File path of correspond Pascal VOC annotations (*.xml).
    """
    img = Image.open(img_path)
    filename, ext = os.path.splitext(os.path.basename(img_path))

    if any([opt.fliplr, opt.fliptb, opt.rot90, opt.rot180, opt.rot270, opt.rotany]):
        fxml = open(f"{label_path}", encoding="utf-8")
        root = parse(fxml).getroot()
        fxml.close()

        if opt.fliplr:
            img_flipped, bboxes, size = transpose(img, root, FLIP_LEFT_RIGHT)
            img_flipped.save(os.path.join(opt.opt_dir, 'Images', filename + '_fliplr' + ext), quality=80)
            writeXml(filename + '_fliplr' + ext, size, bboxes, os.path.join(opt.opt_dir, 'Annotations', filename + '_fliplr' + '.xml'))

        if opt.fliptb:
            img_flipped, bboxes, size = transpose(img, root, FLIP_TOP_BOTTOM)
            img_flipped.save(os.path.join(opt.opt_dir, 'Images', filename + '_fliptb' + ext), quality=80)
            writeXml(filename + '_fliptb' + ext, size, bboxes, os.path.join(opt.opt_dir, 'Annotations',filename + '_fliptb' + '.xml'))
        
        if opt.rot90:
            img_flipped, bboxes, size = transpose(img, root, ROTATE_90)
            img_flipped.save(os.path.join(opt.opt_dir, 'Images', filename + '_rot90' + ext), quality=80)
            writeXml(filename + '_rot90' + ext, size, bboxes, os.path.join(opt.opt_dir, 'Annotations',filename + '_rot90' + '.xml'))
        
        if opt.rot180:
            img_flipped, bboxes, size = transpose(img, root, ROTATE_180)
            img_flipped.save(os.path.join(opt.opt_dir, 'Images', filename + '_rot180' + ext), quality=80)
            writeXml(filename + '_rot180' + ext, size, bboxes, os.path.join(opt.opt_dir, 'Annotations',filename + '_rot180' + '.xml'))
        
        if opt.rot270:
            img_flipped, bboxes, size = transpose(img, root, ROTATE_270)
            img_flipped.save(os.path.join(opt.opt_dir, 'Images', filename + '_rot270' + ext), quality=80)
            writeXml(filename + '_rot270' + ext, size, bboxes, os.path.join(opt.opt_dir, 'Annotations',filename + '_rot270' + '.xml'))
        
        if opt.rotany:
            angle = int(np.random.randint(1, 360)) if opt.angle is None else opt.angle
            img_rotated, bboxes = rotate(img, root, angle)
            img_rotated.save(os.path.join(opt.opt_dir, 'Images', filename + f'_rota{angle}' + ext), quality=80)
            writeXml(filename + f'_rota{angle}' + ext, (img.size[1], img.size[0]), bboxes, os.path.join(opt.opt_dir, 'Annotations', filename + f'_rota{angle}' + '.xml'))

    if opt.bright:
        change_brightness(img).save(os.path.join(opt.opt_dir, 'Images', filename + '_brt' + ext), quality=80)
        copyfile(label_path, os.path.join(opt.opt_dir, 'Annotations', filename + '_brt' + '.xml'))

    if opt.contrast:
        change_contrast(img).save(os.path.join(opt.opt_dir, 'Images', filename + '_ctrs' + ext), quality=80)
        copyfile(label_path, os.path.join(opt.opt_dir, 'Annotations', filename + '_ctrs' + '.xml'))
    
    if opt.color:
        change_color(img).save(os.path.join(opt.opt_dir, 'Images', filename + '_color' + ext), quality=80)
        copyfile(label_path, os.path.join(opt.opt_dir, 'Annotations', filename + '_color' + '.xml'))


def isImage(filename):
    """
    Check if a file is an image based on its extension.
    :param filename: Name of the file to check.
    :return: True if the file is an image, False otherwise.
    """
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}  # ".gif"
    return any(filename.lower().endswith(ext) for ext in img_exts)


def formatXml(elem):
    """
    Add indents according to the level.
    """
    rough_string = tostring(elem, "utf-8")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="	")


def writeXml(img_name:str, img_size, bndboxes:list, xml_file:str):
    """
    Write the information to the specific file.
    img_name: image basename with extension.
    img_size: height(int), width(int)
    bndboxes: [[class_name, x_min, y_min, x_max, y_max], ...]
    """
    annotation = Element("annotation")

    filename = SubElement(annotation, "filename")
    filename.text = img_name

    size = SubElement(annotation, "size")
    width = SubElement(size, "width")
    width.text = str(img_size[1])
    height = SubElement(size, "height")
    height.text = str(img_size[0])
    depth = SubElement(size, "depth")
    depth.text = "3"  # If the image is grayscale, edit it to the "1" (str type).

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
        f.write(formatXml(annotation))  # Save xml


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir',   type=str, default=None , help='images directory')
    parser.add_argument('--labels_dir', type=str, default=None, help='annotations directory')
    parser.add_argument('--opt_dir', type=str, default=None, help='output directory')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    parser.add_argument('--bright', action='store_true', help='enable brightness changing')
    parser.add_argument('--brt_factor', type=float, default=None, help='given a brightness factor, or use the random value')
    parser.add_argument('--brt_lower', type=float, default=0.7, help='brightness lower')
    parser.add_argument('--brt_upper', type=float, default=1.3, help='brightness upper')

    parser.add_argument('--contrast', action='store_true', help='enable contrast changing')
    parser.add_argument('--ctrs_factor', type=float, default=None, help='given a contrast factor, or use the random value')
    parser.add_argument('--ctrs_lower', type=float, default=0.7, help='contrast lower')
    parser.add_argument('--ctrs_upper', type=float, default=1.3, help='contrast upper')
    
    parser.add_argument('--color', action='store_true', help='enable color changing')
    parser.add_argument('--color_factor', type=float, default=None, help='given a color factor, or use the random value')
    parser.add_argument('--color_lower', type=float, default=0.7, help='color lower')
    parser.add_argument('--color_upper', type=float, default=1.3, help='color upper')

    parser.add_argument('--fliplr', action='store_true', help='Horizontal flip')
    parser.add_argument('--fliptb', action='store_true', help='Vertical flip')
    parser.add_argument('--rot90', action='store_true', help='Rotate 90° counter-clockwise')
    parser.add_argument('--rot180', action='store_true', help='Rotate 180° counter-clockwise')
    parser.add_argument('--rot270', action='store_true', help='Rotate 270° counter-clockwise')

    parser.add_argument('--rotany', action='store_true', help='Rotate images counterclockwise by any degree while keeping the canvas intact')
    parser.add_argument('--angle', type=int, default=None, help='given a angle, or use the random integer between 1 to 359')

    opt = parser.parse_args()
    if opt.opt_dir is None:
        opt.opt_dir = opt.img_dir
        print(f'''\033[1;33mWARNING: No specified output files directory, set to default: "{opt.img_dir}".\033[0m''')

    # Set the seed
    np.random.seed(opt.seed)

    # Create the output directories
    os.makedirs(os.path.join(opt.opt_dir, 'Images'), exist_ok=True)
    os.makedirs(os.path.join(opt.opt_dir, 'Annotations'), exist_ok=True)

    print(dumps(vars(opt), indent=4))

    img_names = [x for x in os.listdir(opt.img_dir) if isImage(x)]
    pbar = tqdm(img_names)
    for img_name in pbar:
        pbar.set_postfix_str(f"Image: {img_name}")
        augment_one(os.path.join(opt.img_dir, img_name), os.path.join(opt.labels_dir, os.path.splitext(img_name)[0] + '.xml'))
    pbar.close()

    print('Done!')
