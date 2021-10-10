# This file contains modules common to various models
import math
import time
import numpy as np
import torch
import torch.nn as nn
# from utils.general import non_max_suppression

'''
feature map尺寸计算公式： out_size = (in_size + 2*Padding - kernel_size)/strides + 1
卷积计算时map尺寸向下取整
池化计算时map尺寸向上取整
'''


def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False):
    '''
    Performs Non-Maximum Suppression (NMS) on inference results；
    @param prediction:  size=(batch_size, num_boxes, [xywh,score,num_classes,Θ])
    @param conf_thres:
    @param iou_thres:
    @param merge:
    @param classes:
    @param agnostic:
    @return:
            detections with shape: (batch_size, num_nms_boxes, [])
    '''

    # prediction :(batch_size, num_boxes, [xywh,score,num_classes,Θ])
    nc = prediction[0].shape[1] - 5  # number of classes
    class_index = nc + 5
    # xc : (batch_size, num_boxes) 对应位置为1说明该box超过置信度
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image 单帧图片中的最大目标数量
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    # output: (batch_size, ?)
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x ： (num_boxes,[xywh,score,num_classes,Θ])
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # 取出数组中索引为True的的值即将置信度符合条件的boxes存入x中   x -> (num_confthres_boxes, [xywh,score,num_classes,Θ])

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:class_index] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        angle = x[:, class_index:]  # angle.size=(num_confthres_boxes, [num_angles])
        # torch.max(angle,1) 返回每一行中最大值的那个元素，且返回其索引（返回最大元素在这一行的列索引）
        # angle_index为预测的θ类别  θ ∈ int[0,179]
        angle_value, angle_index = torch.max(angle, 1, keepdim=True)  # size都为 (num_confthres_boxes, 1)
        # box.size = (num_confthres_boxes, [xywhθ])  θ ∈ [-pi/2, pi/2) length=180
        box = torch.cat((x[:, :4], (angle_index - 90) * np.pi / 180), 1)

        # Detections matrix nx7 (xywhθ, conf, clsid) θ ∈ [-pi/2, pi/2)
        if multi_label:
            # nonzero ： 取出每个轴的索引,默认是非0元素的索引（取出括号公式中的为True的元素对应的索引） 将索引号放入i和j中
            # x：(num_confthres_boxes, [xywh,score,num_classes,num_angle])
            # i：num_boxes该维度中的索引号，表示该索引的box其中有class的conf满足要求  length=x中满足条件的所有坐标数量
            # j：num_classes该维度中的索引号，表示某个box中是第j+1类物体的conf满足要求  length=x中满足条件的所有坐标数量
            i, j = (x[:, 5:class_index] > conf_thres).nonzero(as_tuple=False).T
            # 按列拼接  list x：(num_confthres_boxes, [xywhθ]+[conf]+[classid]) θ ∈ [-pi/2, pi/2)
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)  # None即新增一个维度 让每个数值单独成为一个维度

        else:  # best class only
            conf, j = x[:, 5:class_index].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class 按类别筛选
        if classes:
            # list x：(num_confthres_boxes, [xywhθ]+[conf]+[classid]) θ ∈ [-pi/2, pi/2)
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
        # x : (num_confthres_boxes, [xywhθ]+[conf]+[classid]) θ ∈ [-pi/2, pi/2)
        c = x[:, 6:7] * (0 if agnostic else max_wh)  # classesid*4096
        boxes, scores = x[:, :5], x[:, 5]  # boxes[x, y, w, h, θ] θ is 弧度制[-pi/2, pi/2)
        boxes[:, :4] = boxes[:, :4] + c  # boxes xywh(offset by class)

        if i.shape[0] > max_det:  # limit detections  限制单帧图片中检测出的目标数量
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

        # output: (batch_size, num_nms_boxes, [x_LT,y_LT,x_RB,y_RB]+conf+class)
        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def autopad(k, p=None):  # kernel, padding
    '''
    自动填充
    返回padding值
        kernel_size 为 int类型时 ：padding = k // 2（整数除法进行一次）
                        否则    : padding = [x // 2 for x in k]
    '''
    # Pad to 'same'
    if p is None:  # k是否为int类型，是则返回True
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    '''
    深度分离卷积层 Depthwise convolution：
        是G（group）CONV的极端情况；
        分组数量等于输入通道数量，即每个通道作为一个小组分别进行卷积，结果联结作为输出，Cin = Cout = g，没有bias项。
        c1 : in_channels
        c2 : out_channels
        k : kernel_size
        s : stride
        act : 是否使用激活函数
        math.gcd() 返回的是最大公约数
    '''
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    '''
    标准卷积层Conv
    包括Conv2d + BN + HardWish激活函数
    (self, in_channels, out_channels, kernel_size, stride, padding, groups, activation_flag)
    p=None时，out_size = in_size/strides
    '''

    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act else nn.Identity()

    def forward(self, x):  # 前向计算（有BN）
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):  # 前向融合计算（无BN）
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    '''
    标准Bottleneck层
        input : input
        output : input + Conv3×3（Conv1×1(input)）
    (self, in_channels, out_channels, shortcut_flag, group, expansion隐藏神经元的缩放因子)
    out_size = in_size
    '''

    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        '''
        若 shortcut_flag为Ture 且 输入输出通道数相等，则返回跳接后的结构：
            x + Conv3×3（Conv1×1(x)）
        否则不进行跳接：
            Conv3×3（Conv1×1(x)）
        '''
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    '''
    标准ottleneckCSP层
    (self, in_channels, out_channels, Bottleneck层重复次数, shortcut_flag, group, expansion隐藏神经元的缩放因子)
    out_size = in_size
    '''

    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))  # CONV + BottleNeck + Conv2d  out_channels = c_
        y2 = self.cv2(x)  # Conv2d   out_channels = c_
        return self.cv4(self.act(
            self.bn(torch.cat((y1, y2), dim=1))))  # concat(y1 + y2) + BN + LeakyReLU + Conv2d  out_channels = c2


class SPP(nn.Module):
    '''
    空间金字塔池化SPP：
    (self, in_channels, out_channels, 池化尺寸strides[3])
    '''

    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        # 建立5×5 9×9 13×13的最大池化处理过程的list
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    '''
    Focus : 把宽度w和高度h的信息整合到c空间中
    (self, in_channels, out_channels, kernel_size, stride, padding, group, activation_flag)
    '''

    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):
        '''
        x(batch_size, channels, height, width) -> y(batch_size, 4*channels, height/2, weight/2)
        '''
        # ::代表[start:end:step], 以2为步长取值
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class Concat(nn.Module):
    '''
    (dimension)
    默认d=1按列拼接 ， d=0则按行拼接
    '''

    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.3  # confidence threshold
    iou = 0.6  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, dimension=1):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class Flatten(nn.Module):
    '''
    在全局平均池化以后使用，去掉2个维度
    (batch_size, channels, size, size) -> (batch_size, channels*size*size)
    '''

    # Use after nn.AdaptiveAvgPool2d(1) to remove last 2 dimensions
    @staticmethod
    def forward(x):
        return x.view(x.size(0), -1)


class Classify(nn.Module):
    '''
    (self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1)
    (batch_size, channels, size, size) -> (batch_size, channels*1*1)
    '''

    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        # 给定输入数据和输出数据的大小，自适应算法能够自动帮助我们计算核的大小和每次移动的步长
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(batch_size,ch_in,1,1) 返回1×1的池化结果
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)  # to x(batch_size,ch_out,1,1)
        self.flat = Flatten()

    def forward(self, x):
        #
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if x is list
        return self.flat(self.conv(z))  # flatten to x(batch_size, ch_out×1×1)
