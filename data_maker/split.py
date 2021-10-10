import os
import glob
import random
import shutil
import cv2
from PIL import Image
import sys
import codecs
import shapely.geometry as shgeo


import numpy as np


def split_train_val(data, save_dir, val_ratio):
    np.random.seed(32)
    shuffled_indices = np.random.permutation(len(data))
    # shuffled_indices = list(range(len(data)))
    # random.shuffle(shuffled_indices)
    val_set_size = int(len(data) * val_ratio)
    val_indices = shuffled_indices[:val_set_size]
    train_indices = shuffled_indices[val_set_size:]
    val_dir = os.path.join(save_dir, "val")
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    for val in val_indices:
        val_json = data[val]
        val_image = val_json.replace('.json', '.jpg')
        json_name = os.path.split(val_json)[1]
        image_name = json_name.replace('.json', '.jpg')
        shutil.copy(val_json, os.path.join(val_dir, json_name))
        shutil.copy(val_image, os.path.join(val_dir, image_name))
    train_dir = os.path.join(save_dir, "train")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    for train in train_indices:
        train_json = data[train]
        train_image = train_json.replace('.json', '.jpg')
        json_name = os.path.split(train_json)[1]
        image_name = json_name.replace('.json', '.jpg')
        shutil.copy(train_json, os.path.join(train_dir, json_name))
        shutil.copy(train_image, os.path.join(train_dir, image_name))
    # return data.iloc[train_indices], data.iloc[val_indices]


def GetFileFromThisRootDir(dir, ext=None):
    allfiles = []
    needExtFilter = (ext != None)
    for root, dirs, files in os.walk(dir):
        for filespath in files:
            filepath = os.path.join(root, filespath)
            extension = os.path.splitext(filepath)[1][1:]
            if needExtFilter and extension in ext:
                allfiles.append(filepath)
            elif not needExtFilter:
                allfiles.append(filepath)
    return allfiles


def parse_dota_poly(filename):
    """
        parse the dota ground truth in the format:
        [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    """
    objects = []
    # print('filename:', filename)
    f = []
    if (sys.version_info >= (3, 5)):
        fd = open(filename, 'r')
        f = fd
    elif (sys.version_info >= 2.7):
        fd = codecs.open(filename, 'r')
        f = fd
    # count = 0
    while True:
        line = f.readline()
        # count = count + 1
        # if count < 2:
        #     continue
        if line:
            splitlines = line.strip().split(' ')
            object_struct = {}
            ### clear the wrong name after check all the data
            # if (len(splitlines) >= 9) and (splitlines[8] in classname):
            if (len(splitlines) < 9):
                continue
            if (len(splitlines) >= 9):
                object_struct['name'] = splitlines[8]
            if (len(splitlines) == 9):
                object_struct['difficult'] = '0'
            elif (len(splitlines) >= 10):
                # if splitlines[9] == '1':
                # if (splitlines[9] == 'tr'):
                #     object_struct['difficult'] = '1'
                # else:
                object_struct['difficult'] = splitlines[9]
                # else:
                #     object_struct['difficult'] = 0
            object_struct['poly'] = [(float(splitlines[0]), float(splitlines[1])),
                                     (float(splitlines[2]), float(splitlines[3])),
                                     (float(splitlines[4]), float(splitlines[5])),
                                     (float(splitlines[6]), float(splitlines[7]))
                                     ]
            gtpoly = shgeo.Polygon(object_struct['poly'])
            object_struct['area'] = gtpoly.area
            # poly = list(map(lambda x:np.array(x), object_struct['poly']))
            # object_struct['long-axis'] = max(distance(poly[0], poly[1]), distance(poly[1], poly[2]))
            # object_struct['short-axis'] = min(distance(poly[0], poly[1]), distance(poly[1], poly[2]))
            # if (object_struct['long-axis'] < 15):
            #     object_struct['difficult'] = '1'
            #     global small_count
            #     small_count = small_count + 1
            objects.append(object_struct)
        else:
            break
    return objects


def parse_longsideformat(filename):  # filename=??.txt
    """
        parse the longsideformat ground truth in the format:
        objects[i] : [classid, x_c, y_c, longside, shortside, theta]
    """
    objects = []
    f = []
    if (sys.version_info >= (3, 5)):
        fd = open(filename, 'r')
        f = fd
    elif (sys.version_info >= 2.7):
        fd = codecs.open(filename, 'r')
        f = fd
    # count = 0
    while True:
        line = f.readline()
        if line:
            splitlines = line.strip().split(' ')
            object_struct = {}
            ### clear the wrong name after check all the data
            # if (len(splitlines) >= 9) and (splitlines[8] in classname):
            if (len(splitlines) < 6) or (len(splitlines) > 6):
                print('labels长度不为6,出现错误,与预定形式不符')
                continue
            object_struct = [int(splitlines[0]), float(splitlines[1]),
                             float(splitlines[2]), float(splitlines[3]),
                             float(splitlines[4]), float(splitlines[5])
                             ]
            objects.append(object_struct)
        else:
            break
    return objects


## trans dota format to  (cls, c_x, c_y, Longest side, short side, angle:[0,179))
def dota2LongSideFormat(imgpath, txtpath, dstpath, extractclassname):
    """
    trans dota farmat to longside format
    :param imgpath: the path of images
    :param txtpath: the path of txt in dota format
    :param dstpath: the path of txt in YOLO format
    :param extractclassname: the category you selected
    """
    if os.path.exists(dstpath):
        shutil.rmtree(dstpath)  # delete output folder
    os.makedirs(dstpath)  # make new output folder
    filelist = GetFileFromThisRootDir(txtpath)  # fileist=['/.../P0005.txt', ..., /.../P000?.txt]
    for fullname in filelist:  # fullname='/.../P000?.txt'
        objects = parse_dota_poly(fullname)
        '''
        objects =
        [{'name': 'ship', 
          'difficult': '1', 
          'poly': [(1054.0, 1028.0), (1063.0, 1011.0), (1111.0, 1040.0), (1112.0, 1062.0)], 
          'area': 1159.5
          },
          ...
        ]
        '''
        name = os.path.splitext(os.path.basename(fullname))[0]  # name='P000?'
        img_fullname = os.path.join(imgpath, name + '.jpg')  # img_fullname='/.../P000?.png'
        img = Image.open(img_fullname)
        img_w, img_h = img.size
        # print img_w,img_h
        with open(os.path.join(dstpath, name + '.txt'), 'w') as f_out:
            num_gt = 0
            for i, obj in enumerate(objects):
                num_gt = num_gt + 1  # 为当前有效gt计数
                poly = obj['poly']  # poly=[(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
                poly = np.float32(np.array(poly))
                # 四点坐标归一化
                poly[:, 0] = poly[:, 0] / img_w
                poly[:, 1] = poly[:, 1] / img_h

                rect = cv2.minAreaRect(poly)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
                # box = np.float32(cv2.boxPoints(rect))  # 返回rect四个点的值

                c_x = rect[0][0]
                c_y = rect[0][1]
                w = rect[1][0]
                h = rect[1][1]
                theta = rect[-1]  # Range for angle is [-90，0)

                trans_data = cvminAreaRect2longsideformat(c_x, c_y, w, h, theta)
                if not trans_data:
                    if theta != 90:  # Θ=90说明wh中有为0的元素，即gt信息不完整，无需提示异常，直接删除
                        print('opencv表示法转长边表示法出现异常,已将第%d个box排除,问题出现在该图片中:%s' % (i, img_fullname))
                    num_gt = num_gt - 1
                    continue
                else:
                    # range:[-180，0)
                    c_x, c_y, longside, shortside, theta_longside = trans_data

                bbox = np.array((c_x, c_y, longside, shortside))

                if (sum(bbox <= 0) + sum(bbox[:2] >= 1)) >= 1:  # 0<xy<1, 0<side<=1
                    print('bbox[:2]中有>= 1的元素,bbox中有<= 0的元素,已将第%d个box排除,问题出现在该图片中:%s' % (i, img_fullname))
                    print('出问题的longside形式数据:[%.16f, %.16f, %.16f, %.16f, %.1f]' % (
                        c_x, c_y, longside, shortside, theta_longside))
                    num_gt = num_gt - 1
                    continue
                if (obj['name'] in extractclassname):
                    id = extractclassname.index(obj['name'])  # id=类名的索引 比如'plane'对应id=0
                else:
                    print('预定类别中没有类别:%s;已将该box排除,问题出现在该图片中:%s' % (obj['name'], fullname))
                    num_gt = num_gt - 1
                    continue
                theta_label = int(theta_longside + 180.5)  # range int[0,180] 四舍五入
                if theta_label == 180:  # range int[0,179]
                    theta_label = 179
                # outline='id x y longside shortside Θ'

                # final check
                if id > 15 or id < 0:
                    print('id problems,问题出现在该图片中:%s' % (i, img_fullname))
                    print('出问题的longside形式数据:[%.16f, %.16f, %.16f, %.16f, %.1f]' % (
                        c_x, c_y, longside, shortside, theta_longside))
                if theta_label < 0 or theta_label > 179:
                    print('id problems,问题出现在该图片中:%s' % (i, img_fullname))
                    print('出问题的longside形式数据:[%.16f, %.16f, %.16f, %.16f, %.1f]' % (
                        c_x, c_y, longside, shortside, theta_longside))
                outline = str(id) + ' ' + ' '.join(list(map(str, bbox))) + ' ' + str(theta_label)
                f_out.write(outline + '\n')  # 写入txt文件中并加上换行符号 \n

        if num_gt == 0:
            os.remove(os.path.join(dstpath, name + '.txt'))  #
            os.remove(img_fullname)
            os.remove(fullname)
            print('%s 图片对应的txt不存在有效目标,已删除对应图片与txt' % img_fullname)
    print('已完成文件夹内DOTA数据形式到长边表示法的转换')


def cvminAreaRect2longsideformat(x_c, y_c, width, height, theta):
    '''
    trans minAreaRect(x_c, y_c, width, height, θ) to longside format(x_c, y_c, longside, shortside, θ)
    两者区别为:
            当opencv表示法中width为最长边时（包括正方形的情况），则两种表示方法一致
            当opencv表示法中width不为最长边 ，则最长边表示法的角度要在opencv的Θ基础上-90度
    @param x_c: center_x
    @param y_c: center_y
    @param width: x轴逆时针旋转碰到的第一条边
    @param height: 与width不同的边
    @param theta: x轴逆时针旋转与width的夹角，由于原点位于图像的左上角，逆时针旋转角度为负 [-90, 0)
    @return:
            x_c: center_x
            y_c: center_y
            longside: 最长边
            shortside: 最短边
            theta_longside: 最长边和x轴逆时针旋转的夹角，逆时针方向角度为负 [-180, 0)
    '''
    '''
    意外情况:(此时要将它们恢复符合规则的opencv形式：wh交换，Θ置为-90)
    竖直box：box_width < box_height  θ=0
    水平box：box_width > box_height  θ=0
    '''
    if theta == 0:
        theta = -90
        buffer_width = width
        width = height
        height = buffer_width

    if theta > 0:
        if theta != 90:  # Θ=90说明wh中有为0的元素，即gt信息不完整，无需提示异常，直接删除
            print('θ计算出现异常，当前数据为：%.16f, %.16f, %.16f, %.16f, %.1f;超出opencv表示法的范围：[-90,0)' % (
                x_c, y_c, width, height, theta))
        return False

    if theta < -90:
        print(
            'θ计算出现异常，当前数据为：%.16f, %.16f, %.16f, %.16f, %.1f;超出opencv表示法的范围：[-90,0)' % (x_c, y_c, width, height, theta))
        return False

    if width != max(width, height):  # 若width不是最长边
        longside = height
        shortside = width
        theta_longside = theta - 90
    else:  # 若width是最长边(包括正方形的情况)
        longside = width
        shortside = height
        theta_longside = theta

    if longside < shortside:
        print('旋转框转换表示形式后出现问题：最长边小于短边;[%.16f, %.16f, %.16f, %.16f, %.1f]' % (
            x_c, y_c, longside, shortside, theta_longside))
        return False
    if (theta_longside < -180 or theta_longside >= 0):
        print('旋转框转换表示形式时出现问题:θ超出长边表示法的范围：[-180,0);[%.16f, %.16f, %.16f, %.16f, %.1f]' % (
            x_c, y_c, longside, shortside, theta_longside))
        return False

    return x_c, y_c, longside, shortside, theta_longside


def drawLongsideFormatimg(imgpath, txtpath, dstpath, extractclassname, thickness=2):
    """
    根据labels绘制边框(label_format:classid, x_c_normalized, y_c_normalized, longside_normalized, shortside_normalized, Θ)
    :param imgpath: the path of images
    :param txtpath: the path of txt in longside format
    :param dstpath: the path of image_drawed
    :param extractclassname: the category you selected
    """
    if os.path.exists(dstpath):
        shutil.rmtree(dstpath)  # delete output folder
    os.makedirs(dstpath)  # make new output folder
    # 设置画框的颜色    colors = [[178, 63, 143], [25, 184, 176], [238, 152, 129],....,[235, 137, 120]]随机设置RGB颜色
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(extractclassname))]
    filelist = GetFileFromThisRootDir(txtpath)  # fileist=['/.../P0005.txt', ..., /.../P000?.txt]
    for fullname in filelist:  # fullname='/.../P000?.txt'
        objects = parse_longsideformat(fullname)
        '''
        objects[i] = [classid, x_c_normalized, y_c_normalized, longside_normalized, shortside_normalized, theta]
        '''
        name = os.path.splitext(os.path.basename(fullname))[0]  # name='P000?'
        img_fullname = os.path.join(imgpath, name + '.jpg')  # img_fullname='/.../P000?.png'
        img_savename = os.path.join(dstpath, name + '_.jpg')  # img_fullname='/.../_P000?.png'
        img = Image.open(img_fullname)  # 图像被打开但未被读取
        img_w, img_h = img.size
        img = cv2.imread(img_fullname)  # 读取图像像素
        for i, obj in enumerate(objects):
            # obj = [classid, x_c_normalized, y_c_normalized, longside_normalized, shortside_normalized, float:0-179]
            class_index = obj[0]
            # rect=[(x_c,y_c),(w,h),Θ] Θ:flaot[0-179]  -> (-180,0)
            rect = longsideformat2cvminAreaRect(obj[1], obj[2], obj[3], obj[4], (obj[5] - 179.9))
            # poly = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
            poly = np.float32(cv2.boxPoints(rect))  # 返回rect对应的四个点的值 normalized

            # 四点坐标反归一化 取整
            poly[:, 0] = poly[:, 0] * img_w
            poly[:, 1] = poly[:, 1] * img_h
            poly = np.int0(poly)

            # 画出来
            cv2.drawContours(image=img,
                             contours=[poly],
                             contourIdx=-1,
                             color=colors[int(class_index)],
                             thickness=thickness)
            # cv2.line(img, (int(poly[0][0]), int(poly[0][1])), (int(poly[1][0]), int(poly[1][1])), (0, 255, 0), 3)
            # cv2.line(img, (int(poly[1][0]), int(poly[1][1])), (int(poly[2][0]), int(poly[2][1])), (0, 255, 0), 3)
            # cv2.line(img, (int(poly[2][0]), int(poly[2][1])), (int(poly[3][0]), int(poly[3][1])), (0, 255, 0), 3)
            # cv2.line(img, (int(poly[3][0]), int(poly[3][1])), (int(poly[0][0]), int(poly[0][1])), (0, 255, 0), 3)
        cv2.imshow("debug", img)
        cv2.waitKey()
        cv2.imwrite(img_savename, img)

    # time.sleep()


def longsideformat2cvminAreaRect(x_c, y_c, longside, shortside, theta_longside):
    '''
    trans longside format(x_c, y_c, longside, shortside, θ) to minAreaRect(x_c, y_c, width, height, θ)
    两者区别为:
            当opencv表示法中width为最长边时（包括正方形的情况），则两种表示方法一致
            当opencv表示法中width不为最长边 ，则最长边表示法的角度要在opencv的Θ基础上-90度
    @param x_c: center_x
    @param y_c: center_y
    @param longside: 最长边
    @param shortside: 最短边
    @param theta_longside: 最长边和x轴逆时针旋转的夹角，逆时针方向角度为负 [-180, 0)
    @return: ((x_c, y_c),(width, height),Θ)
            x_c: center_x
            y_c: center_y
            width: x轴逆时针旋转碰到的第一条边最长边
            height: 与width不同的边
            theta: x轴逆时针旋转与width的夹角，由于原点位于图像的左上角，逆时针旋转角度为负 [-90, 0)
    '''
    if (theta_longside >= -180 and theta_longside < -90):  # width is not the longest side
        width = shortside
        height = longside
        theta = theta_longside + 90
    else:
        width = longside
        height = shortside
        theta = theta_longside

    if theta < -90 or theta >= 0:
        print('当前θ=%.1f，超出opencv的θ定义范围[-90, 0)' % theta)

    return ((x_c, y_c), (width, height), theta)


my_classnames = ['steel', 'null']

if '__main__' == __name__:
    # jsons = glob.glob("./mydata/data/*.json")
    # save_dir = "./mydata"
    # split_train_val(jsons, save_dir, 0.1)
    dota2LongSideFormat('./data/Images2',
                        './data/labels',
                        './data/yolo_labels',
                        my_classnames)

    drawLongsideFormatimg(imgpath='data/Images2',
                          txtpath='data/yolo_labels',
                          dstpath='data/draw_longside_img',
                          extractclassname=my_classnames)
