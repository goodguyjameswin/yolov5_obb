import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_labels,
    xyxy2xywh, plot_one_rotated_box, strip_optimizer, set_logging)
from shapely.geometry import Polygon, MultiPoint
from utils.torch_utils import select_device, load_classifier, time_synchronized
# from utils.evaluation_utils import rbox2txt


def iou_rotated_rect(line1, line2):
    a = np.array(line1).reshape(4, 2)
    poly1 = Polygon(a).convex_hull
    b = np.array(line2).reshape(4, 2)
    poly2 = Polygon(b).convex_hull
    union_poly = np.concatenate((a, b))
    if not poly1.intersects(poly2):  # if there is no intersection between two polygons
        iou = 0
    else:
        inter_area = poly1.intersection(poly2).area  # intersection area
        print(inter_area)
        union_area = MultiPoint(union_poly).convex_hull.area
        print(union_area)
        if union_area == 0:
            iou = 0
        iou = float(inter_area) / union_area
    return iou


def nms_rotated_rect(dets, scores, thresh):
    obbs = dets[:, 0:-1]
    x1 = np.min(obbs[:, 0::2], axis=1)
    y1 = np.min(obbs[:, 1::2], axis=1)
    x2 = np.max(obbs[:, 0::2], axis=1)
    y2 = np.max(obbs[:, 1::2], axis=1)
    # scores = dets[:, 8]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    polys = []
    for i in range(len(dets)):
        # tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
        #                                    dets[i][2], dets[i][3],
        #                                    dets[i][4], dets[i][5],
        #                                    dets[i][6], dets[i][7]])
        tm_polygon = [dets[i][0], dets[i][1], dets[i][2], dets[i][3],
                      dets[i][4], dets[i][5], dets[i][6], dets[i][7]]
        polys.append(tm_polygon)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        # ovr = []
        i = order[0]
        keep.append(i)
        # if order.size == 0:
        #     break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # w = np.maximum(0.0, xx2 - xx1 + 1)
        # h = np.maximum(0.0, yy2 - yy1 + 1)
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        hbb_inter = w * h
        hbb_ovr = hbb_inter / (areas[i] + areas[order[1:]] - hbb_inter)
        # h_keep_inds = np.where(hbb_ovr == 0)[0]
        h_inds = np.where(hbb_ovr > 0)[0]
        tmp_order = order[h_inds + 1]
        for j in range(tmp_order.size):
            # iou = polyiou.iou_poly(polys[i], polys[tmp_order[j]])
            iou = iou_rotated_rect(polys[i], polys[tmp_order[j]])
            hbb_ovr[h_inds[j]] = iou
            # ovr.append(iou)
            # ovr_index.append(tmp_order[j])

        # ovr = np.array(ovr)
        # ovr_index = np.array(ovr_index)
        # print('ovr: ', ovr)
        # print('thresh: ', thresh)
        # try:
        #     if math.isnan(ovr[0]):
        #         pdb.set_trace()
        # except:
        #     pass
        inds = np.where(hbb_ovr <= thresh)[0]

        # order_obb = ovr_index[inds]
        # print('inds: ', inds)
        # order_hbb = order[h_keep_inds + 1]
        order = order[inds + 1]
        # pdb.set_trace()
        # order = np.concatenate((order_obb, order_hbb), axis=0).astype(np.int)
    return keep


def longsideformat2poly(x_c, y_c, longside, shortside, theta_longside):
    '''
    trans longside format(x_c, y_c, longside, shortside, θ) θ ∈ [0-179]    to  poly
    @param x_c: center_x   tensor
    @param y_c: center_y   tensor
    @param longside: 最长边  tensor
    @param shortside: 最短边  tensor
    @param theta_longside: 最长边和x轴逆时针旋转的夹角，逆时针方向角度为负 [0, 180)  tensor
    @return: poly shape(8)   numpy
    '''
    # Θ:flaot[0-179]  -> (-180,0)
    rect = longsideformat2cvminAreaRect(x_c, y_c, longside, shortside, (theta_longside - 179.9))
    # poly = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
    poly = np.double(cv2.boxPoints(rect))  # 返回rect对应的四个点的值 normalized
    poly.shape = 8
    return poly


def rotate_non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False,
                               without_iouthres=False):
    """
    Performs Rotate-Non-Maximum Suppression (RNMS) on inference results；
    @param prediction: size=(batch_size, num, [xywh,score,num_classes,num_angles])
    @param conf_thres: 置信度阈值
    @param iou_thres:  IoU阈值
    @param merge: None
    @param classes: None
    @param agnostic: 进行nms是否将所有类别框一视同仁，默认False
    @param without_iouthres : 本次nms不做iou_thres的标志位  默认为False
    @return:
            output：经nms后的旋转框(batch_size, num_conf_nms, [xywhθ,conf,classid]) θ∈[0,179]
    """
    # prediction :(batch_size, num_boxes, [xywh,score,num_classes,num_angles])
    nc = prediction[0].shape[1] - 5 - 180  # number of classes = no - 5 -180
    class_index = nc + 5
    # xc : (batch_size, num_boxes) 对应位置为1说明该box超过置信度
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 500  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections 要求冗余检测
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    # output: (batch_size, ?)
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x ： (num_boxes, [xywh, score, num_classes, num_angles])
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # 取出数组中索引为True的的值即将置信度符合条件的boxes存入x中   x -> (num_confthres_boxes, no)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:class_index] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        angle = x[:, class_index:]  # angle.size=(num_confthres_boxes, [num_angles])
        # torch.max(angle,1) 返回每一行中最大值的那个元素，且返回其索引（返回最大元素在这一行的列索引）
        angle_value, angle_index = torch.max(angle, 1, keepdim=True)  # size都为 (num_confthres_boxes, 1)
        # box.size = (num_confthres_boxes, [xywhθ])  θ∈[0,179]
        box = torch.cat((x[:, :4], angle_index), 1)
        if multi_label:
            # nonzero ： 取出每个轴的索引,默认是非0元素的索引（取出括号公式中的为True的元素对应的索引） 将索引号放入i和j中
            # i：num_boxes该维度中的索引号，表示该索引的box其中有class的conf满足要求  length=x中满足条件的所有坐标数量
            # j：num_classes该维度中的索引号，表示某个box中是第j+1类物体的conf满足要求  length=x中满足条件的所有坐标数量
            i, j = (x[:, 5:class_index] > conf_thres).nonzero(as_tuple=False).T
            # 按列拼接  list x：(num_confthres_boxes, [xywhθ]+[conf]+[classid]) θ∈[0,179]
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)

        else:  # best class only
            conf, j = x[:, 5:class_index].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        if without_iouthres:  # 不做nms_iou
            output[xi] = x
            continue

        # Filter by class 按类别筛选
        if classes:
            # list x：(num_confthres_boxes, [xywhθ,conf,classid])
            x = x[(x[:, 6:7] == torch.tensor(classes, device=x.device)).any(1)]  # any（1）函数表示每行满足条件的返回布尔值

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]
        # Batched NMS
        # x：(num_confthres_boxes, [xywhθ,conf,classid]) θ∈[0,179]
        c = x[:, 6:7] * (0 if agnostic else max_wh)  # classes
        # boxes:(num_confthres_boxes, [xy])  scores:(num_confthres_boxes, 1)
        # agnostic用于 不同类别的框仅跟自己类别的目标进行nms   (offset by class) 类别id越大,offset越大
        boxes_xy, box_whthetas, scores = x[:, :2] + c, x[:, 2:5], x[:, 5]
        rects = []
        for i, box_xy in enumerate(boxes_xy):
            rect = longsideformat2poly(box_xy[0], box_xy[1], box_whthetas[i][0], box_whthetas[i][1], box_whthetas[i][2])
            rects.append(rect)
        i = np.array(nms_rotated_rect(np.array(rects), np.array(scores.cpu()), iou_thres))
        # i = nms(boxes, scores)  # i为数组，里面存放着boxes中经nms后的索引

        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]  # 根据nms索引提取x中的框  x.size=(num_conf_nms, [xywhθ,conf,classid]) θ∈[0,179]

        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


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
    if ((theta_longside >= -180) and (theta_longside < -90)):  # width is not the longest side
        width = shortside
        height = longside
        theta = theta_longside + 90
    else:
        width = longside
        height = shortside
        theta = theta_longside

    if (theta < -90) or (theta >= 0):
        print('当前θ=%.1f，超出opencv的θ定义范围[-90, 0)' % theta)

    return ((x_c, y_c), (width, height), theta)


def rbox2txt(rbox, classname, conf, img_name, out_path, pi_format=False):
    """
    将分割图片的目标信息填入原始图片.txt中
    @param robx: rbox:[tensor(x),tensor(y),tensor(l),tensor(s),tensor(θ)]
    @param classname: string
    @param conf: string
    @param img_name: string
    @param path: 文件夹路径 str
    @param pi_format: θ是否为pi且 θ ∈ [-pi/2,pi/2)  False说明 θ∈[0,179]
    """
    if isinstance(rbox, torch.Tensor):
        rbox = rbox.cpu().float().numpy()

    # rbox = np.array(x)
    if pi_format:  # θ∈[-pi/2,pi/2)
        rbox[-1] = (rbox[-1] * 180 / np.pi) + 90  # θ∈[0,179]

    # rect=[(x_c,y_c),(w,h),Θ] Θ:flaot[0-179]  -> (-180,0)
    rect = longsideformat2cvminAreaRect(rbox[0], rbox[1], rbox[2], rbox[3], (rbox[4] - 179.9))
    # poly = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
    poly = np.float32(cv2.boxPoints(rect))  # 返回rect对应的四个点的值
    poly = np.int0(poly).reshape(8)

    splitname = img_name.split('__')  # 分割待merge的图像的名称 eg:['P0706','1','0','_0']
    oriname = splitname[0]  # 获得待merge图像的原图像名称 eg:P706

    # 目标所属图片名称_分割id 置信度 poly classname
    lines = img_name + ' ' + conf + ' ' + ' '.join(list(map(str, poly))) + ' ' + classname
    # 移除之前的输出文件夹,并新建输出文件夹
    if not os.path.exists(out_path):
        os.makedirs(out_path)  # make new output folder

    with open(str(out_path + '/' + oriname) + '.txt', 'a') as f:
        f.writelines(lines + '\n')


def detect(save_img=False):
    '''
    input: save_img_flag
    output(result):
    '''
    # 获取输出文件夹，输入路径，权重，参数等参数
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')

    # Initialize
    set_logging()
    # 获取设备
    device = select_device(opt.device)
    # 移除之前的输出文件夹,并新建输出文件夹
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    # 如果设备为gpu，使用Float16
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    # 加载Float32模型，确保用户设定的输入图片分辨率能整除最大步长s=32(如不能则调整为能整除并返回)
    '''
    model = Model(
                  (model): Sequential(
                                       (0): Focus(...)
                                       (1): Conv(...)
                                            ...
                                       (24): Detect(...)
                    )
    '''
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

    # 设置Float16
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    # 通过不同的输入源来设置不同的数据加载方式
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    # 获取类别名字    names = ['person', 'bicycle', 'car',...,'toothbrush']
    names = model.module.names if hasattr(model, 'module') else model.names
    # 设置画框的颜色    colors = [[178, 63, 143], [25, 184, 176], [238, 152, 129],....,[235, 137, 120]]随机设置RGB颜色
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    # 进行一次前向推理,测试程序是否正常  向量维度（1，3，imgsz，imgsz）
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    """
        path 图片/视频路径  'E:\...\bus.jpg'
        img 进行resize+pad之后的图片   1*3*re_size1*resize2的张量 (3,img_height,img_weight)
        img0 原size图片   (img_height,img_weight,3)          
        cap 当读取图片时为None，读取视频时为视频源   
    """
    for path, img, im0s, vid_cap in dataset:
        print(img.shape)
        img = torch.from_numpy(img).to(device)
        # 图片也设置为Float16
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # 没有batch_size的话则在最前面添加一个轴
        if img.ndimension() == 3:
            # (in_channels,size1,size2) to (1,in_channels,img_height,img_weight)
            img = img.unsqueeze(0)  # 在[0]维增加一个维度

        # Inference
        t1 = time_synchronized()
        """
        model:
        input: in_tensor (batch_size, 3, img_height, img_weight)
        output: 推理时返回 [z,x]
        z tensor: [small+medium+large_inference]  size=(batch_size, 3 * (small_size1*small_size2 + medium_size1*medium_size2 + large_size1*large_size2), nc)
        x list: [small_forward, medium_forward, large_forward]  eg:small_forward.size=( batch_size, 3种scale框, size1, size2, [xywh,score,num_classes]) 
        '''
               
        前向传播 返回pred[0]的shape是(1, num_boxes, nc)
        h,w为传入网络图片的长和宽，注意dataset在检测时使用了矩形推理，所以这里h不一定等于w
        num_boxes = 3 * h/32 * w/32 + 3 * h/16 * w/16 + 3 * h/8 * w/8
        pred[0][..., 0:4] 预测框坐标为xywh(中心点+宽长)格式
        pred[0][..., 4]为objectness置信度
        pred[0][..., 5:5+nc]为分类结果
        pred[0][..., 5+nc:]为Θ分类结果
        """
        # pred : (batch_size, num_boxes, no)  batch_size=1
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        # 进行NMS
        # pred : list[tensor(batch_size, num_conf_nms, [xylsθ,conf,classid])] θ∈[0,179]
        # pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        pred = rotate_non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                          agnostic=opt.agnostic_nms, without_iouthres=False)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # i:image index  det:(num_nms_boxes, [xylsθ,conf,classid]) θ∈[0,179]
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)  # 图片保存路径+图片名字
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            # print(txt_path)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :5] = scale_labels(img.shape[2:], det[:, :5], im0.shape).round()

                # Print results    det:(num_nms_boxes, [xylsθ,conf,classid]) θ∈[0,179]
                for c in det[:, -1].unique():  # unique函数去除其中重复的元素，并按元素（类别）由大到小返回一个新的无元素重复的元组或者列表
                    n = (det[:, -1] == c).sum()  # detections per class  每个类别检测出来的素含量
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string 输出‘数量 类别,’

                # Write results  det:(num_nms_boxes, [xywhθ,conf,classid]) θ∈[0,179]
                for *rbox, conf, cls in reversed(det):  # 翻转list的排列结果,改为类别由小到大的排列
                    # rbox=[tensor(x),tensor(y),tensor(w),tensor(h),tsneor(θ)] θ∈[0,179]
                    # if save_txt:  # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #     with open(txt_path + '.txt', 'a') as f:
                    #         f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        classname = '%s' % names[int(cls)]
                        conf_str = '%.3f' % conf
                        rbox2txt(rbox, classname, conf_str, Path(p).stem, str(out + '/result_txt/result_before_merge'))
                        # plot_one_box(rbox, im0, label=label, color=colors[int(cls)], line_thickness=2)
                        plot_one_rotated_box(rbox, im0, label=label, color=colors[int(cls)], line_thickness=1,
                                             pi_format=False)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results 播放结果
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                    pass
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('   Results saved to %s' % Path(out))

    print('   All Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    """
        weights:训练的权重
        source:测试数据，可以是图片/视频路径，也可以是'0'(电脑自带摄像头),也可以是rtsp等视频流
        output:网络预测之后的图片/视频的保存路径
        img-size:网络输入图片大小
        conf-thres:置信度阈值
        iou-thres:做nms的iou阈值
        device:设置设备
        view-img:是否展示预测之后的图片/视频，默认False
        save-txt:是否将预测的框坐标以txt文件形式保存，默认False
        classes:设置只保留某一部分类别，形如0或者0 2 3
        agnostic-nms:进行nms是否将所有类别框一视同仁，默认False
        augment:推理的时候进行多尺度，翻转等操作(TTA)推理
        update:如果为True，则对所有模型进行strip_optimizer操作，去除pt文件中的优化器等信息，默认为False
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./weights/YOLOv5_DOTAv1.5_OBB.pt',
                        help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='DOTA_demo_view/images',
                        help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='DOTA_demo_view/detection', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=1024, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0,1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', default=False, help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                # 去除pt文件中的优化器等信息
                strip_optimizer(opt.weights)
        else:
            detect()
