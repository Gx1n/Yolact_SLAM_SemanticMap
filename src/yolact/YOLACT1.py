from data import COLORS
from yolact import Yolact
from utils.augmentations import FastBaseTransform
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation
from data import cfg, set_cfg
from collections import defaultdict
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import cv2




color_cache = defaultdict(lambda: {})
class YOLACT:

    #构造函数
    def __init__(self):
        print ('Initializing Yolact network...')
        trained_model='/home/x1/catkin_ws/src/Yolact_SLAM_SemanticMap/src/yolact/weights/yolact_base_54_800000.pth'
        model_path = SavePath.from_str(trained_model)
        config = model_path.model_name + '_config'
        set_cfg(config)

        with torch.no_grad():
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            dataset = None
        print('Loading model...', end='')
        self.net = Yolact()
        self.net.load_weights(trained_model)
        self.net.eval()
        self.net = self.net.cuda()
        self.net.detect.use_fast_nms = True
        self.net.detect.use_cross_class_nms = False
        cfg.mask_proto_debug = False
        print(' Done.')
        #Dilation settings
        self.kernel = np.ones((3, 3), np.uint8)
    #     self.classes=[]
    # def get_color(self,j, on_gpu=None):
    #     color_idx = (self.classes[j] * 5 if False else j * 5) % len(COLORS)
    #
    #     if on_gpu is not None and color_idx in self.color_cache[on_gpu]:
    #         return self.color_cache[on_gpu][color_idx]
    #     else:
    #         color = COLORS[color_idx]
    #         if not True:
    #             # The image might come in as RGB or BRG, depending
    #             color = (color[2], color[1], color[0])
    #         if on_gpu is not None:
    #             color = torch.Tensor(color).to(on_gpu).float() / 255.
    #             color_cache[on_gpu][color_idx] = color
    #         return color

        self.class_names = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
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
                            'scissors', 'teddy bear', 'hair drier', 'toothbrush')

        print("模型初始化完成")

    def prep_display(self, dets_out, img, h, w, undo_transform=True, class_color=True, mask_alpha=1.0, fps_str=''):
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
        effectiveDetect=torch.ge(t[1],0.37)#得到大于0.5置信度的蒙板
        detect_num=effectiveDetect.sum()#得到大于0.5置信度的检测的数量
        idx = t[1].argsort(0, descending=True)[:detect_num] #该图片中取置信度最高的前detect_num个实例
        masks = t[3][idx] #取出置信度最高的detect_num个实例的mask
        classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]] #将这detect_num个实例的类别、置信度、包围框 取出放到numpy数组中

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
        person=[]
        mask1 = np.ones((h, w, 3),dtype =np.uint8)
        #masks = np.zeros((h,w))
        if (num_dets_to_consider):
            masks = masks[:num_dets_to_consider, :, :, None]
            colors = torch.cat(
                [get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)],dim=0)
            masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha
            inv_alph_masks = masks * (-mask_alpha) + 1
            masks_color_summand = masks_color[0]
            # img_gpu = (masks.sum(dim=0) >= 1).float().expand(-1, -1, 3).contiguous()
            for i in range(num_dets_to_consider):
                _class=self.class_names[classes[i]]
                if _class=='person':
                    msk = masks[i,:,:,None]
                    mask = msk.view(1, masks.shape[1], masks.shape[2], masks.shape[3])
                    #在expand中的-1表示取当前所在维度的尺寸img_gpu
                    img_mask = (mask.sum(dim=0) >= 1).float().expand(-1, -1, 3).contiguous()
                    person.append(img_mask)
                else:
                    # mask_img = (masks * 255).byte().cpu().numpy()
                    # mask_img = mask_img[0, :, :, 0]
                    # colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
                    # masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha
                    # inv_alph_masks = masks * (-mask_alpha) + 1
                    # masks_color_summand = masks_color[0]
                    inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
                    masks_color_cumul = masks_color[1:] * inv_alph_cumul
                    masks_color_summand += masks_color_cumul.sum(dim=0)

            img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand
            img_crop = img_gpu.cpu().numpy()
            img_crop = (img_crop * 255).astype(np.uint8)
            # for i in range(3):
            #     img_crop[:, :, i] = img_crop[:, :, i] * (mask_img // 255)
            # cv2.imshow('Debug', img_crop)
            # cv2.waitKey(0)
            if(person):
                img_mask = sum(person)
                img_mask = img_mask.cpu().numpy()
                # print(img_mask.dtype)
                # img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
                img_mask_dil = cv2.dilate(img_mask,self.kernel,iterations=5)
                mask1 = mask1 - img_mask_dil
                img_crop = (mask1 * img_crop).astype(np.uint8)
                #print(img_crop.dtype)
                return (img_mask,img_crop)
            else:
                return (mask1,img_crop)
            # result =[]
            # result = result.append(img_mask,img_crop)

        if num_dets_to_consider == 0:
            #img_numpy = (img_gpu * 255).byte().cpu().numpy()
            img_numpy = (img_gpu * 255).cpu().numpy().astype(np.uint8)
            return (mask1,img_numpy)


        # img_gpu = cv2.cvtColor(img_gpu, cv2.COLOR_RGB2GRAY)
        # cv2.imshow('Debug1', img_gpu.cpu().numpy())
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        # colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
        # masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha
        # inv_alph_masks = masks * (-mask_alpha) + 1
        # masks_color_summand = masks_color[0]
        #
        # if num_dets_to_consider > 1:
        #     inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
        #     masks_color_cumul = masks_color[1:] * inv_alph_cumul
        #     masks_color_summand += masks_color_cumul.sum(dim=0)
        #
        #
        # img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand
        #
        #
        # img_crop = img_gpu.byte().cpu().numpy()
        # for i in range(3):
        #     img_crop[:,:,i] = img_crop[:,:,i] * (mask_img // 255)
        #debug看一下图像有没有问题，看完再注释掉
        # cv2.namedWindow('Debug2', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('Debug2', img_crop)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()



    def GetDynSeg(self,image):
        #print('calling')
        with torch.no_grad():
            #读取图片
            frame = torch.from_numpy(image).cuda().float()
            #print('frame',frame,frame.shape)
            #图像预处理
            batch = FastBaseTransform()(frame.unsqueeze(0))
            #print('batch',batch,batch.shape)
            #预测
            preds = self.net(batch)
            #print('preds',preds)
            result= self.prep_display(preds, frame, None, None, undo_transform=False)
            # print(img_gpu.shape)
            # cv2.imshow('Debug',img_gpu)
            # cv2.waitKey(0)
            #print(type(img_gpu))

            # cv2.imshow('Debug1', img_mask)
            # cv2.imshow('Debug2', img_crop)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            return result

# def seg(image):
#     Net=YOLACT()
#     with torch.no_grad():
#         #读取图片
#         frame = torch.from_numpy(image).cuda().float()
#         #图像预处理
#         batch = FastBaseTransform()(frame.unsqueeze(0))
#         #预测
#         preds = Net.net(batch)
#         img_gpu = Net.prep_display(preds, frame, None, None, undo_transform=False)
#         cv2.imshow('Debug1', img_gpu)
#         return img_gpu

if __name__ == '__main__':
    #tmp_str='''
    img_path="/home/x1/catkin_ws/src/ORB_SLAM2/src/yolact/2.png"
    img=cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
    #mask=np.ones((480,640,3))
    masknet=YOLACT()
    mask,img=masknet.GetDynSeg(img)
    # print(img.dtype)
    # cv2.imshow('Debug1', img)
    # cv2.waitKey(0)
    #print(result)
    # h=image.shape[0]
    # w=image.shape[1]
    # mask=np.ones((h,w),dtype='float32')
    # mask=mask-image
    # mask = np.resize(mask*255,(h,w,3))
    # cv2.imwrite('./',mask)
    # cv2.waitKey(0)
