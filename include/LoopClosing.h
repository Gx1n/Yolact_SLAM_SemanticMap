/**
* This file is part of ORB-SLAM2.
* 回环检测线程
*/

#ifndef LOOPCLOSING_H
#define LOOPCLOSING_H

#include "KeyFrame.h"
#include "LocalMapping.h"
#include "Map.h"
#include "ORBVocabulary.h"
#include "Tracking.h"
#include "pointcloudmapping.h"
#include "KeyFrameDatabase.h"

#include <thread>
#include <mutex>
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"
class PointCloudMapping;

namespace ORB_SLAM2
{

class Tracking;
class LocalMapping;
class KeyFrameDatabase;


class LoopClosing
{
public:

    typedef pair<set<KeyFrame*>,int> ConsistentGroup;    // 具有关联性的候选帧
    typedef map<KeyFrame*,g2o::Sim3,std::less<KeyFrame*>,
    Eigen::aligned_allocator<std::pair<const KeyFrame*, g2o::Sim3> > > KeyFrameAndPose;// 帧 对应的 sim3位姿

public:

    LoopClosing(Map* pMap, KeyFrameDatabase* pDB, ORBVocabulary* pVoc,const bool bFixScale, shared_ptr<PointCloudMapping> pPointCloud);
    //LoopClosing(Map* pMap, KeyFrameDatabase* pDB, ORBVocabulary* pVoc,const bool bFixScale);

    void SetTracker(Tracking* pTracker);

    void SetLocalMapper(LocalMapping* pLocalMapper);

    // Main function  线程 主函数===============
    void Run();

    void InsertKeyFrame(KeyFrame *pKF);

    void RequestReset();

    // This function will run in a separate thread 全局优化
    void RunGlobalBundleAdjustment(unsigned long nLoopKF);
    //============================new===========================
    shared_ptr<PointCloudMapping>  mpPointCloudMapping;

    bool isRunningGBA(){
        unique_lock<std::mutex> lock(mMutexGBA);
        return mbRunningGBA;
    }
    bool isFinishedGBA(){
        unique_lock<std::mutex> lock(mMutexGBA);
        return mbFinishedGBA;
    }   

    void RequestFinish();

    bool isFinished();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    //==================new===============
    int loopcount = 0;
protected:

    bool CheckNewKeyFrames();

    bool DetectLoop();// 关键帧数据库 闭环检测====

    bool ComputeSim3();// 计算闭环处的 相似变换===

    void SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap);

    void CorrectLoop();

    void ResetIfRequested();
    bool mbResetRequested;
    std::mutex mMutexReset;

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    Map* mpMap;
    Tracking* mpTracker;

    KeyFrameDatabase* mpKeyFrameDB;
    ORBVocabulary* mpORBVocabulary;

    LocalMapping *mpLocalMapper;

    std::list<KeyFrame*> mlpLoopKeyFrameQueue;

    std::mutex mMutexLoopQueue;

    // Loop detector parameters  闭环检测参数===
    float mnCovisibilityConsistencyTh;

    // Loop detector variables  闭环检测变量====
    KeyFrame* mpCurrentKF;// 当前帧
    KeyFrame* mpMatchedKF;// 匹配的关键帧
    std::vector<ConsistentGroup> mvConsistentGroups;// 具有连续性的候选帧 群组
    std::vector<KeyFrame*> mvpEnoughConsistentCandidates;
    std::vector<KeyFrame*> mvpCurrentConnectedKFs;
    std::vector<MapPoint*> mvpCurrentMatchedPoints;//当前帧 闭环检测得到的 匹配地图点
    std::vector<MapPoint*> mvpLoopMapPoints;// 当前帧 相邻关键帧上的 地图点 闭环处的地图点
    cv::Mat mScw;
    g2o::Sim3 mg2oScw;//相似变换

    long unsigned int mLastLoopKFid;

    // Variables related to Global Bundle Adjustment
    bool mbRunningGBA;
    bool mbFinishedGBA;
    bool mbStopGBA;
    std::mutex mMutexGBA;
    std::thread* mpThreadGBA;

    // Fix scale in the stereo/RGB-D case
    bool mbFixScale;// 双目/深度 固定的 空间尺度


    bool mnFullBAIdx;
};

} //namespace ORB_SLAM

#endif // LOOPCLOSING_H
