# DynaSLAM

[[Project]](https://bertabescos.github.io/DynaSLAM/)   [[Paper]](https://arxiv.org/pdf/1806.05620.pdf)



# 主要思想
    
    利用 语义分割信息 和 几何信息得到的 动/静分割信息，剔除部分不可靠的 关键点来使得 跟踪 变得更可靠
    
    使用mask-rcnn获取 语义分割信息 
    
    使用 运动点 判断准则 获取 动/静 mask
    
    结合 语义mask 和 动/静 mask 生成 需要剔除的 mask
    
    在构造帧 的时候 对 提取的关键点 进行滤波，删除 不可靠的 关键点，使得 跟踪更可靠
    
# 思考
    
    1. 是否可以 结合 光流 来生成 动/静 mask ，不过要考虑相机自身的运动引起的光流
    2. 如果用于导航，仅仅依靠orb关键点，数量不够，是否可以 添加 边缘 关键点检测算法


DynaSLAM is a visual SLAM system that is robust in dynamic scenarios for monocular, stereo and RGB-D configurations. Having a static map of the scene allows inpainting the frame background that has been occluded by such dynamic objects.

<img src="imgs/teaser.png" width="900px"/>

DynaSLAM: Tracking, Mapping and Inpainting in Dynamic Scenes   
[Berta Bescos](http://bertabescos.github.io), [José M. Fácil](http://webdiis.unizar.es/~jmfacil/), [Javier Civera](http://webdiis.unizar.es/~jcivera/) and [José Neira](http://webdiis.unizar.es/~jneira/)   
RA-L and IROS, 2018

We provide examples to run the SLAM system in the [TUM dataset](http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) as RGB-D or monocular, and in the [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) as stereo or monocular.


# 运动点 判断准则
    从关键帧数据库（最多20个）中获取当前帧的参考帧：
       差异性： 和当前帧 欧拉角度差平方 + 平移量差平方
               利用各自最大最小值 归一化后，使用加权求和 vDist = 0.7*vDist + 0.3*vRot
       对差异性进行排序： DESCENDING 降序排列
       选取 前面 (差异性最大的) 作为参考帧 (最多5个)
                  
    
    // 提取动态点=============这是不是可以考虑用光流来计算动态点===============
    // 1. 选取 参考帧关键点 深度(0~6m) 计算参考帧下3d点 再变换到 世界坐标系下
    // 2. 保留 当前帧 到 世界点 向量 与 参考帧到世界点向量 夹角 小于30的点， 不会太近的点
    // 3. 保留世界点 反投影到当前帧坐标系下深度值 <7m的点
    // 4. 保留世界点 反投影到当前帧像素坐标系下 浓缩平面( 20～620 & 20～460=)内的点,且该点，当前帧深度!=0
    // 5. 根据投影点深度值和其 周围20×20领域点当前帧深度值 筛选出 深度差值较小的 领域点 的深度值 来更新当前帧 深度值
    // 6. 点投影深度值 和 特征点当前帧下深度 差值过大，且该点周围深度方差小，确定该点为运动点

# 代码修改
    
    Frame.cc Frame.h   根据传入的 mask 对提取的关键点进行滤波，剔除部分不可靠的点
    
    Tracking.cc
       双目/单目   仅仅依靠 语义mask 过滤关键点
       RGBD    结合 语义mask  和 动/静mask 来 过滤关键点
         具体做法  先根据 运动模型 轻量级 跟踪 获取当前帧位姿态，使用 运动点 判断准则 获取 和 动/静mask
         
    其他 添加了 c++ 调用 python 程序的 文件
    双目 左右图的 语义检测，直接将 两张图 拼接在一起 输入到网络，获取的语义结果再分开
    这样 检测时间上不会增加多少，因为都会缩放到 网络固定的尺寸进行检测，不过检测精度有所损失，但是速度快啊，这个idear赞


## Getting Started
- Install ORB-SLAM2 prerequisites: C++11 or C++0x Compiler, Pangolin, **OpenCV 2.4.11** and Eigen3  (https://github.com/raulmur/ORB_SLAM2).
- Install boost libraries with the command `sudo apt-get install libboost-all-dev`.
- Install python3, keras and tensorflow, and download the `mask_rcnn_coco.h5` model from this GitHub repository: https://github.com/matterport/Mask_RCNN/releases. 
- Clone this repo:
```bash
git clone git@github.com:BertaBescos/DynaSLAM.git
cd DynaSLAM
```
```
cd DynaSLAM
chmod +x build.sh
./build.sh
```
- Place the `mask_rcnn_coco.h5` model in the folder `DynaSLAM/src/python/`.

## RGB-D Example on TUM Dataset
- Download a sequence from http://vision.in.tum.de/data/datasets/rgbd-dataset/download and uncompress it.

- Associate RGB images and depth images executing the python script [associate.py](http://vision.in.tum.de/data/datasets/rgbd-dataset/tools):

  ```
  python associate.py PATH_TO_SEQUENCE/rgb.txt PATH_TO_SEQUENCE/depth.txt > associations.txt
  ```
These associations files are given in the folder `./Examples/RGB-D/associations/` for the TUM dynamic sequences.

- Execute the following command. Change `TUMX.yaml` to TUM1.yaml,TUM2.yaml or TUM3.yaml for freiburg1, freiburg2 and freiburg3 sequences respectively. Change `PATH_TO_SEQUENCE_FOLDER` to the uncompressed sequence folder. Change `ASSOCIATIONS_FILE` to the path to the corresponding associations file. `PATH_TO_MASKS` and `PATH_TO_OUTPUT` are optional parameters.

  ```
  ./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUMX.yaml PATH_TO_SEQUENCE_FOLDER ASSOCIATIONS_FILE (PATH_TO_MASKS) (PATH_TO_OUTPUT)
  ```
  
If `PATH_TO_MASKS` and `PATH_TO_OUTPUT` are **not** provided, only the geometrical approach is used to detect dynamic objects. 

If `PATH_TO_MASKS` is provided, Mask R-CNN is used to segment the potential dynamic content of every frame. These masks are saved in the provided folder `PATH_TO_MASKS`. If this argument is `no_save`, the masks are used but not saved. If it finds the Mask R-CNN computed dynamic masks in `PATH_TO_MASKS`, it uses them but does not compute them again.

If `PATH_TO_OUTPUT` is provided, the inpainted frames are computed and saved in `PATH_TO_OUTPUT`.

## Stereo Example on KITTI Dataset
- Download the dataset (grayscale images) from http://www.cvlibs.net/datasets/kitti/eval_odometry.php 

- Execute the following command. Change `KITTIX.yaml`to KITTI00-02.yaml, KITTI03.yaml or KITTI04-12.yaml for sequence 0 to 2, 3, and 4 to 12 respectively. Change `PATH_TO_DATASET_FOLDER` to the uncompressed dataset folder. Change `SEQUENCE_NUMBER` to 00, 01, 02,.., 11. By providing the last argument `PATH_TO_MASKS`, dynamic objects are detected with Mask R-CNN.
```
./Examples/Stereo/stereo_kitti Vocabulary/ORBvoc.txt Examples/Stereo/KITTIX.yaml PATH_TO_DATASET_FOLDER/dataset/sequences/SEQUENCE_NUMBER (PATH_TO_MASKS)
```

## Monocular Example on TUM Dataset
- Download a sequence from http://vision.in.tum.de/data/datasets/rgbd-dataset/download and uncompress it.

- Execute the following command. Change `TUMX.yaml` to TUM1.yaml,TUM2.yaml or TUM3.yaml for freiburg1, freiburg2 and freiburg3 sequences respectively. Change `PATH_TO_SEQUENCE_FOLDER`to the uncompressed sequence folder. By providing the last argument `PATH_TO_MASKS`, dynamic objects are detected with Mask R-CNN.
```
./Examples/Monocular/mono_tum Vocabulary/ORBvoc.txt Examples/Monocular/TUMX.yaml PATH_TO_SEQUENCE_FOLDER (PATH_TO_MASKS)
```

## Monocular Example on KITTI Dataset
- Download the dataset (grayscale images) from http://www.cvlibs.net/datasets/kitti/eval_odometry.php 

- Execute the following command. Change `KITTIX.yaml`by KITTI00-02.yaml, KITTI03.yaml or KITTI04-12.yaml for sequence 0 to 2, 3, and 4 to 12 respectively. Change `PATH_TO_DATASET_FOLDER` to the uncompressed dataset folder. Change `SEQUENCE_NUMBER` to 00, 01, 02,.., 11. By providing the last argument `PATH_TO_MASKS`, dynamic objects are detected with Mask R-CNN.
```
./Examples/Monocular/mono_kitti Vocabulary/ORBvoc.txt Examples/Monocular/KITTIX.yaml PATH_TO_DATASET_FOLDER/dataset/sequences/SEQUENCE_NUMBER (PATH_TO_MASKS)
```


## Citation

If you use DynaSLAM in an academic work, please cite:

    @article{bescos2018dynaslam,
      title={{DynaSLAM}: Tracking, Mapping and Inpainting in Dynamic Environments},
      author={Bescos, Berta, F\'acil, JM., Civera, Javier and Neira, Jos\'e},
      journal={IEEE RA-L},
      year={2018}
     }

## Acknowledgements
Our code builds on [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2).

# DynaSLAM
