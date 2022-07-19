/**
* This file is part of DynaSLAM.
* Copyright (C) 2018 Berta Bescos <bbescos at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/bertabescos/DynaSLAM>.
* mask-rcnn 实现的语义分割= 
主要是 python接口=
需要做一些数据转换====
*/

#ifndef __MASKNET_H
#define __MASKNET_H


#ifndef NULL
#define NULL   ((void *) 0)
#endif

//#include <python2.7/Python.h>
//#include </home/x1/miniconda3/envs/yolact/include/python3.6m/Python.h>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <cstdio>
#include <boost/thread.hpp>
#include "include/Conversion.h"
#include <mutex>
#include "Tracking.h"
#include "System.h"
#include "Yolact.h"

class Tracking;

using namespace std;
namespace ORB_SLAM2
{
    class Yolact;
    class SegmentDynObject
    {
    public:
        void Run();
        void SetTracker(Tracking *pTracker);
        mutex mMutexGetNewImg;
        mutex mMutexNewImgSegment;
        mutex mMutexFinish;
        cv::Mat mImg;
        //cv::Mat mImgTemp;
        //cv::Mat mImgSegment_color;
        //cv::Mat mImgSegment_color_final;
        std::pair<cv::Mat,cv::Mat> mImgSegment;
        cv::Mat mImgSegmentmask;
        cv::Mat mImgSegmented;
        Tracking* mpTracker;
        Yolact* yolact;
        bool mbNewImgFlag;
        int mSkipIndex;
        double mSegmentTime;
        int imgIndex;
        bool isNewImgArrived();
        bool CheckFinish();
        bool mbFinishRequested;
        void RequestFinish();
        //void Initialize(const cv::Mat& img);
        void ProduceImgSegment();
        SegmentDynObject();
        ~SegmentDynObject();
    };
}

#endif
