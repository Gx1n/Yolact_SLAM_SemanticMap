from data import COCODetection, get_label_map, MEANS, COLORS
from yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils.functions import MovingAverage, ProgressBar
from layers.box_utils import jaccard, center_size, mask_iou
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation
import pycocotools

from data import cfg, set_cfg, set_dataset

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import time
import random
import cProfile
import pickle
import json
import os
from collections import defaultdict
from pathlib import Path
from collections import OrderedDict
from PIL import Image


import matplotlib.pyplot as plt
import sys
# if('/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path):
# 	sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

color_cache = defaultdict(lambda: {})
def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=1.0, fps_str=''):
    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape

    save = cfg.rescore_bbox
    cfg.rescore_bbox = True
    t = postprocess(dets_out, w, h, visualize_lincomb = False,crop_masks= True,score_threshold   = 0)
    cfg.rescore_bbox = save
    #得到大于0.5置信度的实例
    effectiveDetect=torch.ge(t[1],0.5)#得到大于0.5置信度的蒙板
    detect_num=effectiveDetect.sum()#得到大于0.5置信度的检测的数量
    idx = t[1].argsort(0, descending=True)[:detect_num] #该图片中取置信度最高的前detect_num个实例
    masks = t[3][idx] #取出置信度最高的detect_num个实例的mask
    classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]] #将这detect_num个实例的类别、置信度、包围框 取出放到numpy数组中
    # person_index = (classes == 0)
    # print(person_index)
    #person_index表示了第几个框是否是person类别
    # if (person_index.sum() > 0):
    #     # 存在person这个类别
    #     for i in classes:
    #         if i==0:
    #             masks = masks[person_index]


        # classes = classes[person_index]
        # scores = scores[person_index]
        # boxes = boxes[person_index]
    # index_person = 0
    # person_found = True
    # # 遍历类别数组，如果遇到person就跳出
    # while (not classes[index_person]):
    #     # 这样当class是0的时候，检测的就是person，就记录下index
    #     index_person += 1
    #     # 如果整个图片都没找到person
    #     if (index_person == idx):
    #         person_found = False
    #         break
    # if (not person_found):
    #     print('----- No person -----')
    #     num_dets_to_consider = 0
    # else:
    #     # 这里加入了一个修改，把除了person之外的其他检测结果屏蔽掉
    #     classes_all, scores_all, boxes_all = classes, scores, boxes
    #     classes = classes_all[index_person]
    #     scores = scores_all[index_person]
    #     boxes = boxes_all[index_person]
    #     #num_dets_to_consider = 1
    #     # print(masks.shape) # torch.Size([10, 480, 360])
    #     masks_all = masks
    #     masks = masks_all[index_person]



    num_dets_to_consider = min(detect_num, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < 0:
            num_dets_to_consider = j
            break

    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)

        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            if not undo_transform:
                # The image might come in as RGB or BRG, depending
                color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color
    #存在实例时

    # mask_ = torch.Tensor(mask_).cuda()
    # cv2.imshow('Debug1', mask_.cpu().numpy())
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    if (num_dets_to_consider):
        masks = masks[:num_dets_to_consider, :, :, None]
        result=[]
        # img_gpu = (masks.sum(dim=0) >= 1).float().expand(-1, -1, 3).contiguous()
        for i in range(num_dets_to_consider):
            _class=class_names[classes[i]]
            if _class=='person':
                msk = masks[i,:,:,None]
                mask = msk.view(1, masks.shape[1], masks.shape[2], masks.shape[3])
                #在expand中的-1表示取当前所在维度的尺寸
                img_gpu = (mask.sum(dim=0) >= 1).float().expand(-1, -1, 3).contiguous()
                result.append(img_gpu)
    else:
        masks = []

    img_gpu = sum(result)
    #print(type(img_gpu))
    cv2.imshow('Debug1', img_gpu.cpu().numpy())
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if (num_dets_to_consider):
        masks = masks[:num_dets_to_consider, :, :, None]
    #     for i in range(num_dets_to_consider):
    #         _class=class_names[classes[i]]
    #         if _class=='person':
    #             msk = masks[i,:,:,None]
    #             mask = msk.view(1, masks.shape[1], masks.shape[2], masks.shape[3])
    #             img_gpu = (mask.sum(dim=0) >= 1).float().expand(-1, -1, 3).contiguous()
    else:
        masks = []
    mask_img = (masks * 255).byte().cpu().numpy()
    mask_img = mask_img[0, :, :, 0]
    # img_crop = img.byte().cpu().numpy()
    # for i in range(3):
    #     img_crop[:,:,i] = img_crop[:,:,i] * (mask_img // 255)
    #debug看一下图像有没有问题，看完再注释掉
    # cv2.namedWindow('Debug2', cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('Debug2', img_crop)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
    masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha
    inv_alph_masks = masks * (-mask_alpha) + 1
    masks_color_summand = masks_color[0]

    if num_dets_to_consider > 1:
        inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
        masks_color_cumul = masks_color[1:] * inv_alph_cumul
        masks_color_summand += masks_color_cumul.sum(dim=0)


    img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand

    # for i in range(num_dets_to_consider):
    #     _class=class_names[classes[i]]
    #     if _class=='person':
    #         msk = masks[i,:,:,None]
    #         mask = msk.view(1, masks.shape[1], masks.shape[2], masks.shape[3])
    #         img_gpu = (mask.sum(dim=0) >= 1).float().expand(-1, -1, 3).contiguous()
    #         img_gpu = img_gpu+(mask.sum(dim=0) >= 1).float().expand(-1, -1, 3).contiguous()

    #debug看一下图像有没有问题，看完再注释掉
    # cv2.namedWindow('Debug1', cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('Debug1', img_gpu.byte().cpu().numpy())
    # #cv2.imshow('Debug1', img_crop)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    img_crop = img_gpu.byte().cpu().numpy()
    for i in range(3):
        img_crop[:,:,i] = img_crop[:,:,i] * (mask_img // 255)
    #debug看一下图像有没有问题，看完再注释掉
    # cv2.namedWindow('Debug2', cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('Debug2', img_crop)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return boxes,classes,scores,masks,img_numpy



print("正在加载模型")
trained_model='/home/x1/catkin_ws/src/Yolact_SLAM/src/yolact/weights/yolact_base_54_800000.pth'
model_path = SavePath.from_str(trained_model)
config = model_path.model_name + '_config'
set_cfg(config)

with torch.no_grad():
    cudnn.fastest = True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    dataset = None        
    print('Loading model...', end='')
    net = Yolact()
    net.load_weights(trained_model)
    net.eval()
    print(' Done.')

    net = net.cuda()
    net.detect.use_fast_nms = True
    net.detect.use_cross_class_nms = False
    cfg.mask_proto_debug = False

print("模型初始化完成")

def yolact(img,w,h):
    with torch.no_grad():
        #读取图片
        frame = torch.from_numpy(img).cuda().float()
        #图像预处理
        batch = FastBaseTransform()(frame.unsqueeze(0)) 
        #预测
        preds = net(batch)
        #提取结果，并绘制到图片上
        results = []
        boxes,classes,scores,masks,img_numpy = prep_display(preds, frame, None, None, undo_transform=False)
        masks = masks.cpu().numpy()
        # masks=cv2.imread(masks,cv2.IMREAD_GRAYSCALE)
        # cv2.imshow( "mask", masks)
        # cv2.waitKey(0)
        results.append({
            "boxes": boxes,
            "class_ids": classes,
            "scores": scores,
            "masks": masks,
        })
        #print(results[0]['boxes'])
        #print(np.array(results[0]['masks']).shape)
        return results,img_numpy
        #矩阵维度转换
        #img_numpy = img_numpy[:, :, (2, 1, 0)]

#def GetDynSeg(image):


if __name__ == '__main__':
    #tmp_str='''
    img_path="/home/x1/catkin_ws/src/Yolact_SLAM/src/yolact/my_image.png"
    img=cv2.imread(img_path,1)
    class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                   'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                   'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                   'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                   'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                   'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                   'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                   'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                   'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                   'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                   'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                   'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                   'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                   'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    # data_path="/home/x1/Data/rgbd_dataset_freiburg3_walking_xyz/rgb"
    # result_path="/home/x1/Data/result"
    # results = []
    # for p in Path(data_path).glob('*'):
    #     path = str(p)
    #     name = os.path.basename(path)
    #     name = '.'.join(name.split('.')[:-1]) + '.png'
    #     img_path=os.path.join(data_path,name)
    #     img=cv2.imread(img_path)
    img=cv2.imread(img_path)
    h, w, _ = img.shape
    #GetDynSeg(img)
    results,img_numpy=yolact(img,h,w)
    #print(results)
    #GetDynSeg(img)
    # save_path=os.path.join(result_path,name)
    #cv2.imshow('test',img_numpy)
    #cv2.waitKey(0)
    #报错
    #TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
    #找到报错文件
    #image_m = r['masks'][:,:,i] # 人的结果
    #image_m = image_m.cpu().numpy()
