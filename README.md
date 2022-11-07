环境配置 ubuntu18.04 或者 20.04
cuda 11.1

cudnn 8.05

torch==1.7.1+cu110

torchvision==0.8.2+cu110

torchaudio==0.7.2

opencv3.2.0

eigen 3.3.4或其他版本

先编译SLAM build.sh

建立conda环境yolact并配置yolact环境


本项目在剔除地图中动态物体的同时建立语义点云地图，在原来ORBSLAM2的基础上添加了分割线程和点云建图线程
