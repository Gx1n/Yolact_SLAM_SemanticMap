/**
* This file is part of ORB-SLAM2.
* 工程入口函数 系统类 头文件===================
*/


#ifndef SYSTEM_H
#define SYSTEM_H

#include<string>//字符串
#include<thread>// 线程
#include<opencv2/core/core.hpp>// opencv

// user 
#include "Tracking.h"    // 跟踪类头文件
#include "FrameDrawer.h" // 显示帧类 
#include "MapDrawer.h"   // 显示地图类
#include "Map.h"         // 地图类
#include "LocalMapping.h"// 局部建图类
#include "LoopClosing.h" // 闭环检测类
#include "KeyFrameDatabase.h"// 关键帧数据库类
#include "ORBVocabulary.h"   // ORB字典类
#include "Viewer.h"          // 可视化类
#include "MaskNet.h"
// for point cloud viewing
#include "pointcloudmapping.h"// 外部 点云建图类
class PointCloudMapping; // 申明 点云可视化类

// 命名空间========================
namespace ORB_SLAM2
{

// 声明上述头文件 定义的 需要使用的类=============
class Viewer;      // 可视化类
class FrameDrawer; // 显示帧类 
class Map;         // 地图类
class Tracking;    // 跟踪类
class LocalMapping;// 局部建图类
class LoopClosing; // 闭环检测类
class SegmentDynObject;//语义分割

// 本 System 类 的 声明
class System
{
    public:
	// Input sensor 枚举  输入 传感器类型
	enum eSensor
        {
	    MONOCULAR=0,// 单目0
	    STEREO=1,   // 双目1
	    RGBD=2	// 深度2
	};

public:

    // Initialize the SLAM system. It launches the Local Mapping, Loop Closing and Viewer threads.
     // 初始化系统  启动 建图 闭环检测  可视化 线程 
     System(const string &strVocFile, 
                    const string &strSettingsFile, 
                    const eSensor sensor, const bool bUseViewer = true);

    // Proccess the given stereo frame. Images must be synchronized and rectified.
    // Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Returns the camera pose (empty if tracking fails).
// 双目跟踪  返回相机位姿 ===================================
    cv::Mat TrackStereo(const cv::Mat &imLeft, 
                                             const cv::Mat &imRight, 
                                             const cv::Mat &maskLeft, // 考虑左右图的 语义分割mask
                                             const cv::Mat &maskRight, 
                                             const double &timestamp);

    // Process the given rgbd frame. Depthmap must be registered to the RGB frame.
    // Input image: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Input depthmap: Float (CV_32F).
    // Returns the camera pose (empty if tracking fails).
// 深度 跟踪  返回相机位姿================================
    cv::Mat TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, 
                                           cv::Mat &mask, // 考虑 语义分割mask
                                           const double &timestamp, 
                                           cv::Mat &imRGBOut, // 分割输出
                                           cv::Mat &imDOut, 
                                           cv::Mat &maskOut);

    // Process the given rgbd frame. Depthmap must be registered to the RGB frame.
    // Input image: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Input depthmap: Float (CV_32F).
    // Returns the camera pose (empty if tracking fails).
// 深度 跟踪  返回相机位姿================================
    cv::Mat TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, 
                                           cv::Mat &mask, const double &timestamp,cv::Mat kernel);

    //=========================ros============================
    std::pair<cv::Mat,cv::Mat> TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap,
                      cv::Mat &mask, const double &timestamp,cv::Mat kernel,bool& isKeyframe);
    // Proccess the given monocular frame
    // Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Returns the camera pose (empty if tracking fails).
// 单目 跟踪  返回相机位姿===========================================
    cv::Mat TrackMonocular(const cv::Mat &im, 
                                                     const cv::Mat &mask, // 考虑 语义分割mask
                                                     const double &timestamp);

    // This stops local mapping thread (map building) and performs only camera tracking.
    void ActivateLocalizationMode();// 定位 + 跟踪 模式
    // This resumes local mapping thread and performs SLAM again.
    void DeactivateLocalizationMode();// 定位  + 跟踪 + 建图模式

    // Returns true if there have been a big map change (loop closure, global BA)
    // since last call to this function
    bool MapChanged();// 地图发生重大变化，发生在全局优化时

    // Reset the system (clear map)
    void Reset();// 重置====

    // All threads will be requested to finish.
    // It waits until all threads have finished.
    // This function must be called before saving the trajectory.
    void Shutdown();// 退出=====

    // Save camera trajectory in the TUM RGB-D dataset format.
    // Only for stereo and RGB-D. This method does not work for monocular.
    // Call first Shutdown()
    // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
    void SaveTrajectoryTUM(const string &filename);

    // Save keyframe poses in the TUM RGB-D dataset format.
    // This method works for all sensor input.
    // Call first Shutdown()
    // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
    void SaveKeyFrameTrajectoryTUM(const string &filename);

    // Save camera trajectory in the KITTI dataset format.
    // Only for stereo and RGB-D. This method does not work for monocular.
    // Call first Shutdown()
    // See format details at: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
    void SaveTrajectoryKITTI(const string &filename);

    // TODO: Save/Load functions  保存和载入地图 考虑 八叉树octomap地图
    // SaveMap(const string &filename);
    // LoadMap(const string &filename);

    // Information from most recent processed frame
    // You can call this right after TrackMonocular (or stereo or RGBD)
	int GetTrackingState();// 获取 跟踪线程状态=======
	std::vector<MapPoint*> GetTrackedMapPoints();     // 当前帧 地图点
	std::vector<cv::KeyPoint> GetTrackedKeyPointsUn();// 当前帧 关键点
    //===========================new===========================
private:

    // Input sensor
    eSensor mSensor;// enum 枚举变量  输入相机类型 单目 双目 深度

    // ORB vocabulary used for place recognition and feature matching.
    ORBVocabulary* mpVocabulary;// 词典对象指针 用于 地点识别  特征匹配 orb特征

    // KeyFrame database for place recognition (relocalization and loop detection).
// 关键帧 数据库 对象指针  用于 地点识别 定位 回环检测
    KeyFrameDatabase* mpKeyFrameDatabase;

    // Map structure that stores the pointers to all KeyFrames and MapPoints.
    Map* mpMap;// 地图对象指针  存储 关键帧 和 地图点

    // Tracker. It receives a frame and computes the associated camera pose.
    // It also decides when to insert a new keyframe, create some new MapPoints and
    // performs relocalization if tracking fails.
    Tracking* mpTracker;// 跟踪对象 指针 ========

    // Local Mapper. It manages the local map and performs local bundle adjustment.
    LocalMapping* mpLocalMapper;// 建图对象 指针 =====

    // Loop Closer. It searches loops with every new keyframe. If there is a loop it performs
    // a pose graph optimization and full bundle adjustment (in a new thread) afterwards.
    LoopClosing* mpLoopCloser;// 回环检测对象指针 ======

    // The viewer draws the map and the current camera pose. It uses Pangolin.
    Viewer* mpViewer;// 可视化对象指针 ===========

    FrameDrawer* mpFrameDrawer;// 画关键帧
    MapDrawer* mpMapDrawer;       // 画地图
    FrameDrawer* mpOutputFrameDrawer;//=========new===新添加显示帧==========

    //语义分割
    SegmentDynObject* mpSegment;
    std::thread* mptSegment;

    // System threads: Local Mapping, Loop Closing, Viewer.
    // The Tracking thread "lives" in the main execution thread that creates the System object.
    std::thread* mptLocalMapping;// 建图线程         指针
    std::thread* mptLoopClosing;   // 闭环检测线程  指针
    std::thread* mptViewer;               // 可视化线程      指针

// Reset flag 线程重启标志 ===========================================
    std::mutex mMutexReset;// 互斥量   保护 mbReset 变量
    bool mbReset;

	// Change mode flags   系统模式=======================================
    std::mutex mMutexMode;
    bool mbActivateLocalizationMode;// 跟踪 + 定位
    bool mbDeactivateLocalizationMode;// 再加 建图

	// Tracking state 跟踪线程 状态 ======================================
    int mTrackingState;
    std::vector<MapPoint*> mTrackedMapPoints;// 当前帧跟踪到的地图点 3d
    std::vector<cv::KeyPoint> mTrackedKeyPointsUn;// 2d 关键点


    std::mutex mMutexState; // ========new 新添加 数据锁================

    // point cloud mapping
    shared_ptr<PointCloudMapping> mpPointCloudMapping;
};

}// namespace ORB_SLAM

#endif // SYSTEM_H
