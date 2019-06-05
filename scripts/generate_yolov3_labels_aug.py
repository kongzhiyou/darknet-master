"""
generate labels for yolo format with data augmentation

author:will
date: 2018.08.17
"""

import xml.etree.ElementTree as ET
import os
import glob
import tensorlayer as tl
import cv2
import numpy as np

mission_name = "/Users/peter/will/package"
sets = ['train', 'test']
# classes = list(map(str, range(1, 8)))
classes = {'yellow_package': 1, 'orange_package': 2, 'blue_package': 3,'white_bag': 4,'full_tray': 5,'m_forklift': 6}
class_label = {k:v for v,k in classes.items()}

need_augmentation = True

aug_ops = ['flip', 'crop']

crop_fraction = np.arange(0.95, 0.55, -0.05)
print(crop_fraction)


def augmentation(image, bbox, size, cls_list, list_file):
    # crop
    w, h = size
    img = tl.vis.read_image(image)image_1264
    _img = img
    _bbox = bbox
    for frac in crop_fraction:
        for op in aug_ops:
            if op == 'crop':
                im_new, clas, coords = tl.prepro.obj_box_crop(img, cls_list,
                                                              bbox, wrg=int(w * frac), hrg=int(h * frac),
                                                              is_rescale=True, is_center=True, is_random=False)
                _img = im_new
                _bbox = coords
            elif op == 'flip':
                im_new, coords = tl.prepro.obj_box_left_right_flip(_img, _bbox, is_rescale=True, is_center=True,
                                                                   is_random=False)
                clas = cls_list

            base_name = os.path.basename(image)
            img_new_name = base_name[:-4] + '_' + op + str(frac) + '_' '.jpg'
            img_new_path = os.path.join(os.path.dirname(image), img_new_name)
            tl.vis.save_image(im_new, img_new_path)
            list_file.write(img_new_path + '\n')

            label_dir = os.path.join(os.path.dirname(os.path.dirname(image)), 'labels')
            label_file = open(os.path.join(label_dir, os.path.basename(img_new_name)[:-4] + '.txt'), 'w')

            for cls, bb in zip(clas, coords):
                label_file.write(str(cls) + " " + " ".join([str(a) for a in bb]) + '\n')

            label_file.close()


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
    return x, y, w, h


def convert_annotation(data_name, set_name, im_id):
    in_file = open('%s/%s/Annotations/%s.xml' % (data_name, set_name, im_id))
    out_file = open('%s/%s/labels/%s.txt' % (data_name, set_name, image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    cls_list = []
    bbox_list = []

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in class_label.keys():
            continue
        cls_id = classes.index(class_label[cls])
        cls_list.append(cls_id)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
             float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        bbox_list.append(bb)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    return (w, h), cls_list, bbox_list


if __name__ == '__main__':
    for data_set in sets:
        if not os.path.exists('%s/%s/labels/' % (mission_name, data_set)):
            os.makedirs('%s/%s/labels/' % (mission_name, data_set))
        list_file = open('%s_%s.txt' % (mission_name, data_set), 'w')
        image_path_list = glob.glob("%s/%s/JPEGImages/*.jp*g" % (mission_name, data_set))

        for image_path in image_path_list:
            # YOLO v3 seem not support jpeg format, need convert to jpg
            if image_path.endswith('jpeg'):
                print('convert {} to jpg format'.format(image_path))
                img = cv2.imread(image_path)
                os.remove(image_path)
                image_path = image_path[:-4] + 'jpg'
                cv2.imwrite(image_path, img)

            list_file.write(image_path + "\n")
            image_id = os.path.basename(image_path)[:-4]
            size, cls_list, bbox_list = convert_annotation(mission_name, data_set, image_id)

            if need_augmentation and data_set == 'train':
                augmentation(image_path, bbox_list, size, cls_list, list_file)

        list_file.close()
