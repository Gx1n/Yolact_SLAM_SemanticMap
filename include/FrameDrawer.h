/**
* This file is part of ORB-SLAM2.
* 关键帧显示 内容生成 图像+关键点+文字
*/

#ifndef FRAMEDRAWER_H
#define FRAMEDRAWER_H

#include "Tracking.h"
#include "MapPoint.h"
#include "Map.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include<mutex>
namespace ORB_SLAM2
{

class Tracking;
class Viewer;

class FrameDrawer
{
public:
    FrameDrawer(Map* pMap);// 类构造函数

    // Update info from the last processed frame.
    void Update(Tracking *pTracker);// 从 track 对象更新数据到 本类内

    // Draw last processed frame.
    cv::Mat DrawFrame();// 显示关键帧

protected:

    void DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText);//显示 文本信息

    // Info of the frame to be drawn
    cv::Mat mIm;
    int N;
    vector<cv::KeyPoint> mvCurrentKeys;// 当前帧关键点
    vector<bool> mvbMap, mvbVO;           // 是否有对应的地图点
    bool mbOnlyTracking;                              //  模式
    int mnTracked, mnTrackedVO;
    vector<cv::KeyPoint> mvIniKeys;// 初始化参考帧关键点
    vector<int> mvIniMatches;// 匹配点
    int mState;//tracker状态

    Map* mpMap;// 地图

    std::mutex mMutex;// 访问 tracker 类 的数据线程锁
};

} //namespace ORB_SLAM

#endif // FRAMEDRAWER_H
