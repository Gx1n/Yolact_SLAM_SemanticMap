/**
* This file is part of ORB-SLAM2.
* 地图的可视化
*/

#ifndef MAPDRAWER_H
#define MAPDRAWER_H

#include"Map.h"
#include"MapPoint.h"
#include"KeyFrame.h"
#include<pangolin/pangolin.h>

#include<mutex>

namespace ORB_SLAM2
{

class MapDrawer
{
public:
    MapDrawer(Map* pMap, const string &strSettingPath);

    Map* mpMap;

// 显示点======普通点黑色===参考地图点红色===颜色可修改====
    void DrawMapPoints();
// 显示关键帧================蓝色====关键帧连线偏绿色
    void DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph);
// 显示当前帧 相机位姿========绿色==
    void DrawCurrentCamera(pangolin::OpenGlMatrix &Twc);
// 设置当前帧 相机姿==========
    void SetCurrentCameraPose(const cv::Mat &Tcw);
    void SetReferenceKeyFrame(KeyFrame *pKF);// 没看到
// 获取当前相机位姿，返回 OpenGlMatrix 类型=====
    void GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M);

private:

// 显示 大小尺寸参数
    float mKeyFrameSize;
    float mKeyFrameLineWidth;
    float mGraphLineWidth;
    float mPointSize;
    float mCameraSize;
    float mCameraLineWidth;

    cv::Mat mCameraPose;

    std::mutex mMutexCamera;
};

} //namespace ORB_SLAM

#endif // MAPDRAWER_H
