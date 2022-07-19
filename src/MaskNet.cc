/**
* This file is part of DynaSLAM.
*
* Copyright (C) 2018 Berta Bescos <bbescos at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/bertabescos/DynaSLAM>.
* c++ 调用 python 实现的 mask-rcnn 获取mask分割结果
*/

#include "MaskNet.h"
#include <iostream>
#define SKIP_NUMBER 1
//#include"ConverTool.h"

namespace ORB_SLAM2
{

    SegmentDynObject::SegmentDynObject():mbFinishRequested(false),mbNewImgFlag(false),mSkipIndex(SKIP_NUMBER),mSegmentTime(0),imgIndex(0){}

    SegmentDynObject::~SegmentDynObject() {}

    void SegmentDynObject::SetTracker(Tracking *pTracker)
    {
        mpTracker=pTracker;
    }

    bool SegmentDynObject::isNewImgArrived()
    {
        unique_lock<mutex> lock(mMutexGetNewImg);
        if(mbNewImgFlag)
        {
            mbNewImgFlag=false;
            return true;
        }
        else
            return false;
    }

    void SegmentDynObject::Run()
    {
        yolact = new Yolact();
        while (1)
        {
            usleep(1);
            if(!isNewImgArrived())
                continue;
            cout << "Wait for new RGB img time =" << endl;
            if(mSkipIndex==SKIP_NUMBER)
            {
                std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
                mImgSegment = yolact->GetSegmentation(mImg);
//                cv::Mat mask_ = mImgSegment.first;
//                cv::Mat img_ = mImgSegment.second;
                //cvtColor(mImgSegment,mImgSegment,CV_RGB2GRAY);
                std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
                mSegmentTime+=std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
                mSkipIndex=0;
                imgIndex++;
            }
            mSkipIndex++;
            ProduceImgSegment();
            if(CheckFinish())
            {
                break;
            }
        }

    }

    bool SegmentDynObject::CheckFinish()
    {
        unique_lock<mutex> lock(mMutexFinish);
        return mbFinishRequested;
    }

    void SegmentDynObject::RequestFinish()
    {
        unique_lock<mutex> lock(mMutexFinish);
        mbFinishRequested=true;
    }

    void SegmentDynObject::ProduceImgSegment()
    {
        std::unique_lock <std::mutex> lock(mMutexNewImgSegment);
        //mImgTemp=mImgSegmentLatest;
        mImgSegment.first.copyTo(mImgSegmentmask);
        mImgSegment.second.copyTo(mImgSegmented);
        //mImgSegment=mImgTemp;
        mpTracker->mbNewSegImgFlag=true;
    }

}
