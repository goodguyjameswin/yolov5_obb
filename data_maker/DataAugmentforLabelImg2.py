# -*- coding=utf-8 -*-

# 包括:
#     1. 裁剪(需改变bbox)
#     2. 平移(需改变bbox)
#     3. 改变亮度
#     4. 加噪声
#     5. 旋转角度(需要改变bbox)
#     6. 镜像(需要改变bbox)
#     7. cutout
#  注意:
#     random.seed(),相同的seed,产生的随机数是一样的!!


import time
import random
import copy
import cv2
import os
import math
import numpy as np
from skimage.util import random_noise
from lxml import etree, objectify
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon  # 多边形
import argparse


def rotate_xy(x, y, angle, cx, cy):  # 角度单位是弧度
    """
    点(x,y) 绕(cx,cy)点旋转
    """
    # print(cx,cy)
    # angle = angle * pi / 180
    x_new = (x - cx) * math.cos(angle) - (y - cy) * math.sin(angle) + cx
    y_new = (x - cx) * math.sin(angle) + (y - cy) * math.cos(angle) + cy
    return x_new, y_new


# 显示旋转图片
def show_pic(img, bboxes=None, rot_mat=None):
    '''
    输入:
        img:图像array
        bboxes:图像的所有boudning box list, 格式为[[cx, cy, w, h, angle]....]
        names:每个box对应的名称
    '''
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        x_tl, y_tl = rotate_xy(bbox[0] - bbox[2] / 2, bbox[1] - bbox[3] / 2, bbox[4], bbox[0], bbox[1])
        x_tr, y_tr = rotate_xy(bbox[0] + bbox[2] / 2, bbox[1] - bbox[3] / 2, bbox[4], bbox[0], bbox[1])
        x_bl, y_bl = rotate_xy(bbox[0] - bbox[2] / 2, bbox[1] + bbox[3] / 2, bbox[4], bbox[0], bbox[1])
        x_br, y_br = rotate_xy(bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2, bbox[4], bbox[0], bbox[1])
        cv2.line(img, (int(x_tl), int(y_tl)), (int(x_tr), int(y_tr)), (0, 255, 0), 3)
        cv2.line(img, (int(x_tr), int(y_tr)), (int(x_br), int(y_br)), (0, 255, 0), 3)
        cv2.line(img, (int(x_br), int(y_br)), (int(x_bl), int(y_bl)), (0, 255, 0), 3)
        cv2.line(img, (int(x_bl), int(y_bl)), (int(x_tl), int(y_tl)), (0, 255, 0), 3)
        # if rot_mat is not None:
        #     tl = np.dot(rot_mat, np.array([x_tl, y_tl, 1]))
        #     tr = np.dot(rot_mat, np.array([x_tr, y_tr, 1]))
        #     bl = np.dot(rot_mat, np.array([x_bl, y_bl, 1]))
        #     br = np.dot(rot_mat, np.array([x_br, y_br, 1]))
        #     cv2.line(img, (int(tl[0]), int(tl[1])), (int(tr[0]), int(tr[1])), (0, 255, 0), 3)
        #     cv2.line(img, (int(tr[0]), int(tr[1])), (int(br[0]), int(br[1])), (0, 255, 0), 3)
        #     cv2.line(img, (int(br[0]), int(br[1])), (int(bl[0]), int(bl[1])), (0, 255, 0), 3)
        #     cv2.line(img, (int(bl[0]), int(bl[1])), (int(tl[0]), int(tl[1])), (0, 255, 0), 3)
        # else:
        #     cv2.line(img, (int(x_tl), int(y_tl)), (int(x_tr), int(y_tr)), (0, 255, 0), 3)
        #     cv2.line(img, (int(x_tr), int(y_tr)), (int(x_br), int(y_br)), (0, 255, 0), 3)
        #     cv2.line(img, (int(x_br), int(y_br)), (int(x_bl), int(y_bl)), (0, 255, 0), 3)
        #     cv2.line(img, (int(x_bl), int(y_bl)), (int(x_tl), int(y_tl)), (0, 255, 0), 3)

    cv2.namedWindow('pic', 0)  # 1表示原图
    cv2.moveWindow('pic', 0, 0)
    cv2.resizeWindow('pic', 1200, 800)  # 可视化的图片大小
    cv2.imshow('pic', img)
    # cv2.imwrite("./test_result/0001" + ".jpg", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 图像均为cv2读取
class DataAugmentForObjectDetection():
    def __init__(self, rotation_rate=0.5, max_rotation_angle=360,
                 crop_rate=0.5, shift_rate=0.5, change_light_rate=0.5,
                 add_noise_rate=0.5, flip_rate=0.5,
                 cutout_rate=0.5, cut_out_length=10, cut_out_holes=2, cut_out_threshold=0.,
                 is_addNoise=True, is_changeLight=True, is_cutout=True, is_rotate_img_bbox=True,
                 is_crop_img_bboxes=True, is_shift_pic_bboxes=True, is_filp_pic_bboxes=True):

        # 配置各个操作的属性
        self.rotation_rate = rotation_rate
        self.max_rotation_angle = max_rotation_angle
        self.crop_rate = crop_rate
        self.shift_rate = shift_rate
        self.change_light_rate = change_light_rate
        self.add_noise_rate = add_noise_rate
        self.flip_rate = flip_rate
        self.cutout_rate = cutout_rate

        self.cut_out_length = cut_out_length
        self.cut_out_holes = cut_out_holes
        self.cut_out_threshold = cut_out_threshold

        # 是否使用某种增强方式
        self.is_addNoise = is_addNoise
        self.is_changeLight = is_changeLight
        self.is_cutout = is_cutout
        self.is_rotate_img_bbox = is_rotate_img_bbox
        self.is_crop_img_bboxes = is_crop_img_bboxes
        self.is_shift_pic_bboxes = is_shift_pic_bboxes
        self.is_filp_pic_bboxes = is_filp_pic_bboxes

    # 加噪声
    def _addNoise(self, img):
        '''
        输入:
            img:图像array
        输出:
            加噪声后的图像array,由于输出的像素是在[0,1]之间,所以得乘以255
        '''
        # return cv2.GaussianBlur(img, (11, 11), 0)
        return random_noise(img, mode='gaussian', seed=int(time.time()), clip=True) * 255

    # 调整亮度
    def _changeLight(self, img):
        alpha = random.uniform(0.35, 1)
        blank = np.zeros(img.shape, img.dtype)
        return cv2.addWeighted(img, alpha, blank, 1 - alpha, 0)

    # cutout
    def _cutout(self, img, bboxes, length=100, n_holes=1, threshold=0.5):
        '''
        Randomly mask out one or more patches from an image.
        Args:
            img : a 3D numpy array,(h,w,c)
            bboxes : 框的格式为[[cx, cy, w, h, angle]....]
            n_holes (int): Number of patches to cut out of each image.
            length (int): The length (in pixels) of each square patch.
        '''

        def cal_iou(boxA, boxB):
            '''
            boxA, boxB为两个框，返回iou
            boxB为bouding box
            '''
            # determine the (x, y)-coordinates of the intersection rectangle
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            if xB <= xA or yB <= yA:
                return 0.0

            # compute the area of intersection rectangle
            interArea = (xB - xA + 1) * (yB - yA + 1)

            # compute the area of both the prediction and ground-truth
            # rectangles
            boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
            iou = interArea / float(boxBArea)
            return iou

        # 得到h和w
        if img.ndim == 3:
            h, w, c = img.shape
        else:
            _, h, w, c = img.shape
        mask = np.ones((h, w, c), np.float32)
        for n in range(n_holes):
            chongdie = True  # 看切割的区域是否与box重叠太多
            y1, y2, x1, x2 = 0., 0., 0., 0.
            while chongdie:
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - length // 2, 0, h)
                # numpy.clip(a, a_min, a_max, out=None), clip这个函数将将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于
                # a_max，小于a_min,的就使得它等于a_min
                y2 = np.clip(y + length // 2, 0, h)
                x1 = np.clip(x - length // 2, 0, w)
                x2 = np.clip(x + length // 2, 0, w)

                chongdie = False
                for box in bboxes:
                    box_r = math.sqrt(box[2] ** 2 + box[3] ** 2) / 2
                    rbox = [box[0] - box_r, box[1] - box_r, box[0] + box_r, box[1] + box_r]
                    if cal_iou([x1, y1, x2, y2], rbox) > threshold:  # 阈值越小，重叠区域面积越小
                        chongdie = True
                        break
            mask[int(y1): int(y2), int(x1): int(x2), :] = 0.
        img = img * mask
        # show_pic(img, bboxes)
        return img

    # 旋转
    def _rotate_img_bbox(self, img, bboxes, angle=30, scale=1.):  # 为统一角度，规定角度为顺时针旋转0至360度
        '''
        输入:
            img:图像array,(h,w,c)
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为格式为[cx, cy, w, h, angle],要确保是数值
            angle:旋转角度
            scale:默认1
        输出:
            rot_img:旋转后的图像array
            rot_bboxes:旋转后的boundingbox坐标list
        '''
        # ---------------------- 旋转图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        # 角度变弧度
        rangle = np.deg2rad(angle)  # angle in radians
        # now calculate new image width and height
        nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
        nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
        # ask OpenCV for the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), -angle, scale)  # 为统一角度，输入参数angle取负值
        # calculate the move from the old center to the new center combined
        # with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        # the move only affects the translation, so update the translation
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # 仿射变换
        rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)
        # ---------------------- 矫正bbox坐标 ----------------------
        # rot_mat是最终的旋转矩阵
        # 获取原始bbox的中心点，然后转将中心点换到旋转后的坐标系下
        rot_bboxes = list()
        for bbox in bboxes:
            point1 = np.dot(rot_mat, np.array([bbox[0], bbox[1], 1]))
            # 加入list中
            # rot_bboxes.append([point1[0], point1[1], bbox[2], bbox[3], bbox[4] + rangle])
            rot_bboxes.append([point1[0], point1[1], bbox[2] * scale,
                               bbox[3] * scale, (bbox[4] + rangle) % (math.pi * 2)])
        return rot_img, rot_bboxes

    # 裁剪
    def _crop_img_bboxes(self, img, bboxes):
        '''
        裁剪后的图片要包含所有的框
        输入:
            img:图像array
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[cx, cy, w, h, angle],要确保是数值
        输出:
            crop_img:裁剪后的图像array
            crop_bboxes:裁剪后的bounding box的坐标list
        '''
        # ---------------------- 裁剪图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        cx_min = w  # 裁剪后的包含所有目标框的最小的框
        cx_max = 0
        cy_min = h
        cy_max = 0

        for bbox in bboxes:
            bbox_r = math.sqrt(bbox[2] ** 2 + bbox[3] ** 2) / 2

            cx_min = min(bbox[0] - bbox_r, cx_min)
            cy_min = min(bbox[1] - bbox_r, cy_min)
            cx_max = max(bbox[0] + bbox_r, cx_max)
            cy_max = max(bbox[1] + bbox_r, cy_max)

        # 确保不要越界
        cx_min = max(0, cx_min)
        cy_min = max(0, cy_min)
        cx_max = min(w, cx_max)
        cy_max = min(h, cy_max)

        d_to_left = cx_min  # 包含所有目标框的最小框到左边的距离
        d_to_right = w - cx_max  # 包含所有目标框的最小框到右边的距离
        d_to_top = cy_min  # 包含所有目标框的最小框到顶端的距离
        d_to_bottom = h - cy_max  # 包含所有目标框的最小框到底部的距离

        # 随机扩展这个最小框
        crop_x_min = int(cx_min - random.uniform(0, d_to_left))
        crop_y_min = int(cy_min - random.uniform(0, d_to_top))
        crop_x_max = int(cx_max + random.uniform(0, d_to_right))
        crop_y_max = int(cy_max + random.uniform(0, d_to_bottom))

        # 随机扩展这个最小框 , 防止别裁的太小
        # crop_x_min = int(x_min - random.uniform(d_to_left//2, d_to_left))
        # crop_y_min = int(y_min - random.uniform(d_to_top//2, d_to_top))
        # crop_x_max = int(x_max + random.uniform(d_to_right//2, d_to_right))
        # crop_y_max = int(y_max + random.uniform(d_to_bottom//2, d_to_bottom))

        # 确保不要越界
        crop_x_min = max(0, crop_x_min)
        crop_y_min = max(0, crop_y_min)
        crop_x_max = min(w, crop_x_max)
        crop_y_max = min(h, crop_y_max)

        crop_img = copy.deepcopy(img)
        crop_img = crop_img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

        # 判断图片是的等宽高的，否则填充短边
        crop_h = crop_img.shape[0]
        crop_w = crop_img.shape[1]
        crop_bboxes = list()
        if crop_h != crop_w:
            if crop_h < crop_w:
                pad = (crop_w - crop_h) // 2
                crop_img = cv2.copyMakeBorder(crop_img, pad, crop_w - crop_h - pad, 0, 0,
                                              cv2.BORDER_CONSTANT, value=(0, 0, 0))
                for bbox in bboxes:
                    crop_bboxes.append([bbox[0] - crop_x_min, bbox[1] - crop_y_min + pad, bbox[2], bbox[3], bbox[4]])
            else:
                pad = (crop_h - crop_w) // 2
                crop_img = cv2.copyMakeBorder(crop_img, 0, 0, pad, crop_h - crop_w - pad,
                                              cv2.BORDER_CONSTANT, value=(0, 0, 0))
                for bbox in bboxes:
                    crop_bboxes.append([bbox[0] - crop_x_min + pad, bbox[1] - crop_y_min, bbox[2], bbox[3], bbox[4]])
        # ---------------------- 裁剪boundingbox ----------------------
        # 裁剪后的boundingbox坐标计算
        else:
            for bbox in bboxes:
                crop_bboxes.append([bbox[0] - crop_x_min, bbox[1] - crop_y_min, bbox[2], bbox[3], bbox[4]])

        return crop_img, crop_bboxes

    # 平移
    def _shift_pic_bboxes(self, img, bboxes):
        '''
        平移后的图片要包含所有的框
        输入:
            img:图像array
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[cx, cy, w, h, angle],要确保是数值
        输出:
            shift_img:平移后的图像array
            shift_bboxes:平移后的bounding box的坐标list
        '''
        # ---------------------- 平移图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        cx_min = w  # 裁剪后的包含所有目标框的最小的框
        cx_max = 0
        cy_min = h
        cy_max = 0

        for bbox in bboxes:
            bbox_r = math.sqrt(bbox[2] ** 2 + bbox[3] ** 2) / 2

            cx_min = min(bbox[0] - bbox_r, cx_min)
            cy_min = min(bbox[1] - bbox_r, cy_min)
            cx_max = max(bbox[0] + bbox_r, cx_max)
            cy_max = max(bbox[1] + bbox_r, cy_max)

        # 确保不要越界
        cx_min = max(0, cx_min)
        cy_min = max(0, cy_min)
        cx_max = min(w, cx_max)
        cy_max = min(h, cy_max)

        d_to_left = cx_min  # 包含所有目标框的最大左移动距离
        d_to_right = w - cx_max  # 包含所有目标框的最大右移动距离
        d_to_top = cy_min  # 包含所有目标框的最大上移动距离
        d_to_bottom = h - cy_max  # 包含所有目标框的最大下移动距离

        x = random.uniform(-(d_to_left - 1) / 3, (d_to_right - 1) / 3)
        y = random.uniform(-(d_to_top - 1) / 3, (d_to_bottom - 1) / 3)

        M = np.float32([[1, 0, x], [0, 1, y]])  # x为向左或右移动的像素值,正为向右负为向左; y为向上或者向下移动的像素值,正为向下负为向上
        shift_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        # ---------------------- 平移boundingbox ----------------------
        shift_bboxes = list()
        for bbox in bboxes:
            shift_bboxes.append([bbox[0] + x, bbox[1] + y, bbox[2], bbox[3], bbox[4]])

        return shift_img, shift_bboxes

    # 镜像
    def _filp_pic_bboxes(self, img, bboxes):
        '''
            平移后的图片要包含所有的框
            输入:
                img:图像array
                bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[cx, cy, w, h, angle],要确保是数值
            输出:
                flip_img:平移后的图像array
                flip_bboxes:平移后的bounding box的坐标list
        '''
        # ---------------------- 翻转图像 ----------------------

        flip_img = copy.deepcopy(img)
        h, w, _ = img.shape

        sed = random.random()

        if 0 < sed < 0.33:  # 0.33的概率水平翻转，0.33的概率垂直翻转,0.33是对角反转
            flip_img = cv2.flip(flip_img, 0)  # _flip_x
            inver = 0
        elif 0.33 < sed < 0.66:
            flip_img = cv2.flip(flip_img, 1)  # _flip_y
            inver = 1
        else:
            flip_img = cv2.flip(flip_img, -1)  # flip_x_y
            inver = -1

        # ---------------------- 调整boundingbox ----------------------
        flip_bboxes = list()
        for bbox in bboxes:
            if inver == 0:
                flip_bboxes.append([bbox[0], h - bbox[1], bbox[2], bbox[3], (2 * math.pi) - bbox[4]])
            elif inver == 1:
                flip_bboxes.append([w - bbox[0], bbox[1], bbox[2], bbox[3], (2 * math.pi) - bbox[4]])
            elif inver == -1:
                flip_bboxes.append([w - bbox[0], h - bbox[1], bbox[2], bbox[3], (bbox[4] + math.pi) % (2 * math.pi)])

        return flip_img, flip_bboxes

    # 图像增强方法
    def dataAugment(self, img, bboxes):
        '''
        图像增强
        输入:
            img:图像array
            bboxes:该图像的所有框坐标
        输出:
            img:增强后的图像
            bboxes:增强后图片对应的box
        '''
        change_num = 0  # 改变的次数
        # print('------')
        while change_num < 1:  # 默认至少有一种数据增强生效

            if self.is_rotate_img_bbox:
                if random.random() > self.rotation_rate:  # 旋转
                    change_num += 1
                    # angle = random.uniform(-self.max_rotation_angle, self.max_rotation_angle)
                    angle = random.uniform(0, self.max_rotation_angle)
                    scale = random.uniform(0.5, 1.0)
                    # angle = 30
                    # scale = 1.
                    img, bboxes = self._rotate_img_bbox(img, bboxes, angle, scale)

            if self.is_shift_pic_bboxes:
                if random.random() < self.shift_rate:  # 平移
                    change_num += 1
                    img, bboxes = self._shift_pic_bboxes(img, bboxes)

            if self.is_changeLight:
                if random.random() > self.change_light_rate:  # 改变亮度
                    change_num += 1
                    img = self._changeLight(img)

            if self.is_addNoise:
                if random.random() < self.add_noise_rate:  # 加噪声
                    change_num += 1
                    img = self._addNoise(img)
            if self.is_cutout:
                if random.random() < self.cutout_rate:  # cutout
                    change_num += 1
                    img = self._cutout(img, bboxes, length=self.cut_out_length, n_holes=self.cut_out_holes,
                                       threshold=self.cut_out_threshold)
            if self.is_filp_pic_bboxes:
                if random.random() < self.flip_rate:  # 翻转
                    change_num += 1
                    img, bboxes = self._filp_pic_bboxes(img, bboxes)

            if self.is_crop_img_bboxes:
                if random.random() > self.crop_rate:  # 裁剪
                    change_num += 1
                    img, bboxes = self._crop_img_bboxes(img, bboxes)

        return img, bboxes


# xml解析工具
class ToolHelper():
    # 从xml文件中提取bounding box信息, 格式为[[cx, cy, w, h, angle, name]]
    def parse_xml(self, path):
        '''
        输入：
            xml_path: xml的文件路径
        输出：
            从xml文件中提取bounding box信息, 格式为[[cx, cy, w, h, angle, name]]
        '''
        tree = ET.parse(path)
        root = tree.getroot()
        objs = root.findall('object')
        coords = list()
        for ix, obj in enumerate(objs):
            name = obj.find('name').text
            box = obj.find('robndbox')
            cx = int(float(box[0].text))
            cy = int(float(box[1].text))
            w = int(float(box[2].text))
            h = int(float(box[3].text))
            angle = float(box[4].text)
            coords.append([cx, cy, w, h, angle, name])
        return coords

    # 保存图片结果
    def save_img(self, file_name, save_folder, img):
        cv2.imwrite(os.path.join(save_folder, file_name), img)

    # 保存xml结果
    def save_xml(self, file_name, save_folder, img_info, height, width, channel, bboxs_info):
        '''
        :param file_name:文件名
        :param save_folder:#保存的xml文件的结果
        :param height:图片的信息
        :param width:图片的宽度
        :param channel:通道
        :return:
        '''
        folder_name, img_name = img_info  # 得到图片的信息

        E = objectify.ElementMaker(annotate=False)

        anno_tree = E.annotation(
            E.folder(folder_name),
            E.filename(img_name),
            E.path(os.path.join(folder_name, img_name)),
            E.source(
                E.database('Unknown'),
            ),
            E.size(
                E.width(width),
                E.height(height),
                E.depth(channel)
            ),
            E.segmented(0),
        )

        labels, bboxs = bboxs_info  # 得到边框和标签信息
        for label, box in zip(labels, bboxs):
            anno_tree.append(
                E.object(
                    E.name(label),
                    E.pose('Unspecified'),
                    E.truncated('0'),
                    E.difficult('0'),
                    E.robndbox(
                        E.cx(box[0]),
                        E.cy(box[1]),
                        E.w(box[2]),
                        E.h(box[3]),
                        E.angle(box[4])
                    )
                ))

        etree.ElementTree(anno_tree).write(os.path.join(save_folder, file_name), pretty_print=True)

    # 保存txt结果
    def save_txt(self, file_name, save_folder, bboxs_info):
        '''
        :param file_name:文件名
        :param save_folder:#保存的xml文件的结果
        :param height:图片的信息
        :param width:图片的宽度
        :param channel:通道
        :return:
        '''
        labels, bboxs = bboxs_info  # 得到边框和标签信息
        with open(os.path.join(save_folder, file_name), 'w') as t:
            for label, box in zip(labels, bboxs):
                # rect = ((box[0], box[1]), (box[2], box[3]), box[-1])
                # poly = np.float32(cv2.boxPoints(rect))
                x_tl, y_tl = rotate_xy(box[0] - box[2] / 2, box[1] - box[3] / 2, box[4], box[0], box[1])
                x_tr, y_tr = rotate_xy(box[0] + box[2] / 2, box[1] - box[3] / 2, box[4], box[0], box[1])
                x_br, y_br = rotate_xy(box[0] + box[2] / 2, box[1] + box[3] / 2, box[4], box[0], box[1])
                x_bl, y_bl = rotate_xy(box[0] - box[2] / 2, box[1] + box[3] / 2, box[4], box[0], box[1])
                t.write(' '.join([str(x_tl), str(y_tl), str(x_tr), str(y_tr),
                                  str(x_br), str(y_br), str(x_bl), str(y_bl),
                                  label]) + '\n')


if __name__ == '__main__':

    need_aug_num = 400  # 每张图片需要增强的次数

    is_endwidth_dot = True  # 文件是否以.jpg或者png结尾

    dataAug = DataAugmentForObjectDetection()  # 数据增强工具类

    toolhelper = ToolHelper()  # 工具

    # 获取相关参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_img_path', type=str, default='data/Images')
    parser.add_argument('--source_xml_path', type=str, default='data/Annotations')
    parser.add_argument('--save_img_path', type=str, default='data/Images2')
    parser.add_argument('--save_xml_path', type=str, default='data/Annotations2')
    parser.add_argument('--save_txt_path', type=str, default='data/labels')
    args = parser.parse_args()
    source_img_path = args.source_img_path  # 图片原始位置
    source_xml_path = args.source_xml_path  # xml的原始位置

    save_img_path = args.save_img_path  # 图片增强结果保存文件
    save_xml_path = args.save_xml_path  # xml增强结果保存文件
    save_txt_path = args.save_txt_path  # txt增强结果保存文件

    # 如果保存文件夹不存在就创建
    if not os.path.exists(save_img_path):
        os.mkdir(save_img_path)

    if not os.path.exists(save_xml_path):
        os.mkdir(save_xml_path)

    if not os.path.exists(save_txt_path):
        os.mkdir(save_txt_path)

    for parent, _, files in os.walk(source_img_path):
        files.sort()
        for file in files:
            cnt = 0
            pic_path = os.path.join(parent, file)
            xml_path = os.path.join(source_xml_path, file[:-4] + '.xml')
            values = toolhelper.parse_xml(xml_path)  # 解析得到box信息，格式为[[cx, cy, w, h, angle]...]
            coords = [v[:5] for v in values]  # 得到框
            labels = [v[-1] for v in values]  # 对象的标签

            # 如果图片是有后缀的
            if is_endwidth_dot:
                # 找到文件的最后名字
                dot_index = file.rfind('.')
                _file_prefix = file[:dot_index]  # 文件名的前缀
                _file_suffix = file[dot_index:]  # 文件名的后缀
            img = cv2.imread(pic_path)

            # show_pic(img, coords)  # 显示原图
            while cnt < need_aug_num:  # 继续增强
                auged_img, auged_bboxes = dataAug.dataAugment(img, coords)
                # auged_bboxes_int = np.array(auged_bboxes).astype(np.int32)
                # show_pic(auged_img, auged_bboxes)  # 增强后的图
                height, width, channel = auged_img.shape  # 得到图片的属性
                img_name = '{}_{}{}'.format(_file_prefix, cnt + 1, _file_suffix)  # 图片保存的信息
                toolhelper.save_img(img_name, save_img_path,
                                    auged_img)  # 保存增强图片

                toolhelper.save_xml('{}_{}.xml'.format(_file_prefix, cnt + 1),
                                    save_xml_path, (save_img_path, img_name), height, width, channel,
                                    (labels, auged_bboxes))  # 保存xml文件

                toolhelper.save_txt('{}_{}.txt'.format(_file_prefix, cnt + 1),
                                    save_txt_path, (labels, auged_bboxes))
                # show_pic(auged_img, auged_bboxes)  # 增强后的图
                print(img_name)
                cnt += 1  # 继续增强下一张
