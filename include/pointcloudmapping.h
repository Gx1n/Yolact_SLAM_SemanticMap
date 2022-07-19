/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#ifndef POINTCLOUDMAPPING_H
#define POINTCLOUDMAPPING_H

#include "System.h"
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <condition_variable>

#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>




using namespace ORB_SLAM2;

class PointCloudMapping
{
public:
    typedef pcl::PointXYZRGBA PointT;
    typedef pcl::PointCloud<PointT> PointCloud;
    pcl::PointCloud<pcl::PointXYZRGBA> pcl_cloud_kf;
    pcl::PointCloud<pcl::PointXYZRGBA> pcl_filter;
    PointCloudMapping( double resolution_ );
    void Cloud_transform(pcl::PointCloud<pcl::PointXYZRGBA>& source, pcl::PointCloud<pcl::PointXYZRGBA>& out);
    void save();
    // 插入一个keyframe，会更新一次地图
    void insertKeyFrame( KeyFrame* kf, cv::Mat& color, cv::Mat& depth );
    void shutdown();
    void viewer();
    void public_cloud( pcl::PointCloud<pcl::PointXYZRGBA> &cloud_kf);

protected:
    PointCloud::Ptr generatePointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth);
    
    PointCloud::Ptr globalMap;
    //===================new==================
    PointCloud::Ptr KfMap;
    shared_ptr<thread>  viewerThread;   
    
    bool shutDownFlag =false;
    mutex shutDownMutex;
    
    condition_variable keyFrameUpdated;
    mutex keyFrameUpdateMutex;

    // data to generate point clouds
    vector<KeyFrame*> keyframes;
    std::vector<cv::Mat> semanticImgs;
    vector<cv::Mat> colorImgs;
    vector<cv::Mat> depthImgs;
    mutex keyframeMutex;
    uint16_t lastKeyframeSize =0;

    double resolution = 0.05;
    pcl::VoxelGrid<PointT>  voxel;
    //===================new==================
    pcl::StatisticalOutlierRemoval<PointT> sor;
};

#endif // POINTCLOUDMAPPING_H
