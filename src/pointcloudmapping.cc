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

#include "pointcloudmapping.h"
#include <KeyFrame.h>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/projection_matrix.h>
#include "Converter.h"
#include <boost/make_shared.hpp>

#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include <tf/transform_broadcaster.h>


PointCloudMapping::PointCloudMapping(double resolution_)
{
    this->resolution = resolution_;
    voxel.setLeafSize( resolution, resolution, resolution);
    globalMap = boost::make_shared< PointCloud >( );
    //=================new===================
    KfMap = boost::make_shared< PointCloud >( );
    viewerThread = make_shared<thread>( bind(&PointCloudMapping::viewer, this ) );
}

void PointCloudMapping::shutdown()
{
    {
        unique_lock<mutex> lck(shutDownMutex);
        shutDownFlag = true;
        keyFrameUpdated.notify_one();
    }
    viewerThread->join();
}

void PointCloudMapping::insertKeyFrame(KeyFrame* kf, cv::Mat& color, cv::Mat& depth)
{
    cout<<"receive a keyframe, id = "<<kf->mnId<<endl;
    unique_lock<mutex> lck(keyframeMutex);
    keyframes.push_back( kf );
    colorImgs.push_back( color.clone() );
    depthImgs.push_back( depth.clone() );

    keyFrameUpdated.notify_one();
}

pcl::PointCloud< PointCloudMapping::PointT >::Ptr PointCloudMapping::generatePointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth)
{
    PointCloud::Ptr tmp( new PointCloud() );
    //cv::Mat img_bgra;
    //转为4通道  多一个透明度
    //cv::cvtColor(color,img_bgra,CV_BGR2BGRA);
    // point cloud is null ptr
    //delete black point
    for ( int m=0; m<depth.rows; m+=1 )
    {
        for ( int n=0; n<depth.cols; n+=1 )
        {
            float d = depth.ptr<float>(m)[n];
            if (d < 0.01 || d > 8)
                continue;

            cv::Vec3b &pixel = color.at<cv::Vec3b>(m, n);
            if (pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0)
            {
                continue;
            }
            PointT p;
            p.z = d;
            p.x = ( n - kf->cx) * p.z / kf->fx;
            p.y = ( m - kf->cy) * p.z / kf->fy;

            p.b = color.ptr<uchar>(m)[n*3];
            p.g = color.ptr<uchar>(m)[n*3+1];
            p.r = color.ptr<uchar>(m)[n*3+2];

            tmp->points.push_back(p);
        }
    }

    Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat( kf->GetPose() );
    PointCloud::Ptr cloud(new PointCloud);
    pcl::transformPointCloud( *tmp, *cloud, T.inverse().matrix());
    cloud->is_dense = false;

    cout<<"generate point cloud for kf "<<kf->mnId<<", size="<<cloud->points.size()<<endl;
    return cloud;
}


void PointCloudMapping::viewer()
{
    pcl::visualization::CloudViewer viewer("viewer");
    while(1)
    {
        {
            unique_lock<mutex> lck_shutdown( shutDownMutex );
            if (shutDownFlag)
            {
                break;
            }
        }
        {
            unique_lock<mutex> lck_keyframeUpdated( keyFrameUpdateMutex );
            keyFrameUpdated.wait( lck_keyframeUpdated );
        }

        // keyframe is updated
        size_t N=0;
        {
            unique_lock<mutex> lck( keyframeMutex );
            N = keyframes.size();
        }

        if(N==0)
        {
            cout<<"Keyframes miss!"<<endl;
            usleep(1000);
            continue;
        }
        KfMap->clear();

        for ( size_t i=lastKeyframeSize; i<N ; i++ )
        {
            PointCloud::Ptr p = generatePointCloud( keyframes[i], colorImgs[i], depthImgs[i] );
            *KfMap += *p;
            *globalMap += *p;
        }
        PointCloud::Ptr tmp(new PointCloud());
        //voxel.setInputCloud( globalMap );
        voxel.setInputCloud( KfMap );
        voxel.filter( *tmp );
        //globalMap->swap( *tmp );
        KfMap->swap( *tmp );
        //pcl_cloud_kf = *KfMap;

        //Cloud_transform(pcl_cloud_kf,pcl_filter);
        //viewer.showCloud( globalMap );
        viewer.showCloud( KfMap );
        //cout << "show global map, size=" << globalMap->points.size() << endl;
        cout << "show global map, size=" << KfMap->points.size() << endl;
        lastKeyframeSize = N;
    }
}

void PointCloudMapping::public_cloud( pcl::PointCloud< pcl::PointXYZRGBA >& cloud_kf )
{
    cloud_kf =pcl_cloud_kf;
}

void PointCloudMapping::Cloud_transform(pcl::PointCloud<pcl::PointXYZRGBA>& source, pcl::PointCloud<pcl::PointXYZRGBA>& out)
{
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered;
    Eigen::Matrix4f m;

    m<< 0,0,1,0,
        -1,0,0,0,
        0,-1,0,0;
    Eigen::Affine3f transform(m);
    pcl::transformPointCloud (source, out, transform);
}
