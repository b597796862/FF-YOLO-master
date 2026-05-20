import tensorflow as tf
import xml.etree.ElementTree as ET


# 创建一个示例函数来生成XML内容
def generate_xml(image_id, bboxes):
    # 创建XML根元素
    annotation = ET.Element("annotation")

    # 创建一个folder元素
    folder = ET.SubElement(annotation, "folder")
    folder.text = "VOC2012"

    # 创建一个filename元素
    filename = ET.SubElement(annotation, "filename")
    filename.text = image_id.replace(".xml",".jpg")

    # 创建一个size元素
    size = ET.SubElement(annotation, "size")
    width = ET.SubElement(size, "width")
    width.text = "512"  # 假设图片宽度为512像素
    height = ET.SubElement(size, "height")
    height.text = "512"  # 假设图片高度为512像素
    depth = ET.SubElement(size, "depth")
    depth.text = "3"  # 假设图片深度为3

    # 创建一个object列表
    for bbox in bboxes:
        object = ET.SubElement(annotation, "object")
        name = ET.SubElement(object, "name")
        name.text = bbox[0]  # 假设bbox格式为[类别, xmin, ymin, xmax, ymax]
        pose = ET.SubElement(object, "pose")
        pose.text = "Unspecified"
        truncated = ET.SubElement(object, "truncated")
        truncated.text = "0"
        difficult = ET.SubElement(object, "difficult")
        difficult.text = "0"
        bndbox = ET.SubElement(object, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(int(bbox[1]))  # xmin
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(int(bbox[2]))  # ymin
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(int(bbox[3]))  # xmax
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(int(bbox[4]))  # ymax

    return annotation


# # 使用函数生成XML文档
# image_id = "image1"
# bboxes = [("car", 100, 100, 200, 200), ("person", 50, 50, 150, 150)]
# annotation = generate_xml(image_id, bboxes)
#
# # 将ElementTree转换为字符串并写入文件
# xml_str = ET.tostring(annotation, encoding='utf8', method='xml')
# with open(f"{image_id}.xml", 'wb') as f:
#     f.write(xml_str)