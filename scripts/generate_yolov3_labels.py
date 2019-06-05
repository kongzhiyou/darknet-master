import xml.etree.ElementTree as et
import os
import cv2
import glob

mission_name = "/Users/peter/data/wljr/data/sld/all_data" #数据主目录里面应该包含test,train两个文件夹，两个文件夹中都有Annotations JPEGImages
sets = ["train", "test"]
# classes = ["1", "2", "3", "4", "5"]
classes = {'others': 1,'tray_package':2,'forklift':3,'h_forklift':4}
class_index = {k: v for v, k in classes.items()}


#将标注的objectx,y,w,h做归一化处理
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

#输出label.txt文件
def convert_annotation(data_name, set_name, xml_root, img_id):
    with open('%s/%s/labels/%s.txt' % (data_name, set_name, img_id), 'w') as out_file:
        size = xml_root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in xml_root.iter('object'):
            cls = obj.find('name').text
            if cls.isdecimal():
                cls_index = int(cls)
            else:
                cls_index = classes[cls]   #根据类的名称返回类索引


            if cls_index not in class_index:
                continue

            xml_box = obj.find('bndbox')
            b = (float(xml_box.find('xmin').text), float(xml_box.find('xmax').text),
                 float(xml_box.find('ymin').text), float(xml_box.find('ymax').text))
            bb = convert((w, h), b)  # 调用convert方法将原始点的像素转化成比例

            out_file.write(str(cls_index-1) + " " + " ".join([str(a) for a in bb]) + '\n')


if __name__ == '__main__':
    for data_set in sets:
        if not os.path.exists('%s/%s/labels/' % (mission_name, data_set)):
            os.makedirs('%s/%s/labels/' % (mission_name, data_set))
        list_file = open('%s_%s.txt' % (mission_name, data_set), 'w') # 生成一个项目索引txt文件，里面是已经转换的图片的信息
        xml_path_list = glob.glob('%s/%s/Annotations/*.xml' % (mission_name, data_set))
        for xml_path in xml_path_list:
            tree = et.parse(xml_path)
            root = tree.getroot()
            image_name = xml_path.split('/')[-1].split('.')[0]+'.jpg'
            #image_name = root.find('filename').text
            image_path = os.path.join('%s/%s/JPEGImages' % (mission_name, data_set), image_name)

            if image_path.endswith('jpeg'):   #图片的扩展名
                img = cv2.imread(image_path)
                os.remove(image_path)
                image_path = image_path[:-4] + 'jpg'
                cv2.imwrite(image_path, img)

            list_file.write(image_path + "\n")
            image_id = os.path.splitext(image_name)[0]
            convert_annotation(mission_name, data_set, root, image_id)

        list_file.close()