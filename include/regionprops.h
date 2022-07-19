/*
 *  Regionprops
 *  Copyright 2015 Andrea Pennisi
 * 联通域====区域属性 面积 周长 矩 矩形框 多边形框 凸包 旋转矩阵 等
 *
 */

#ifndef REGIONPROPS_H
#define REGIONPROPS_H

#include <opencv2/opencv.hpp>
#include <iostream>

#include "region.h"

class RegionProps {
    public:
        /**
         * @brief RegionProps new object
         */
        RegionProps(const std::vector<cv::Point> &_contour, const cv::Mat &_img);
        /**
         * @brief getRegion return the object region with all the information
         * @return return the object region with all the information
         */
        inline Region getRegion() const
        {
            return region;
        }
    private:
        /**
         * @brief compute compute regionprops
         * @param contours
         */
        void compute();
        /**
         * @brief area compute the area bounded by the contour
         * @return the area bounded by the contour
         */
        double area();
        /**
         * @brief perimeter compute the perimeter
         * @return perimeter
         */
        double perimeter();
        /**
         * @brief moments compute the moments
         * @return the moments
         */
        cv::Moments moments();
        /**
         * @brief centroid compute the centroid
         * @return the centroid
         */
        cv::Point centroid();
        /**
         * @brief boundingbox compute the bounding box
         * @return the boundingbox
         */
        cv::Rect boundingbox();
        /**
         * @brief aspectratio compute the aspectratio
         * @return the aspectratio
         */
        double aspectratio();
        /**
         * @brief equivalentdiameter compute the equivalent diameter
         * @return equivalentdiameter
         */
        double equivalentdiameter();
        /**
         * @brief extent compute the extent
         * @return extent()
         */
        double extent();
        /**
         * @brief convex_hull compute the convex hull
         * @return the convex hull
         */
        std::vector< cv::Point > convex_hull();
        /**
         * @brief convexarea
         * @return the convex area
         */
        double convexarea();
        /**
         * @brief solidity
         * @return the solidity
         */
        double solidity();
        /**
         * @brief ellipse
         * @return the fitted ellipse
         */
        cv::RotatedRect ellipse();
        /**
         * @brief majoraxislength
         * @return the major axis of the ellipse
         */
        double majoraxislength();
        /**
         * @brief minoraxislength
         * @return the minor axis of the ellipse
         */
        double minoraxislength();
        /**
         * @brief orientation
         * @return the ellipse orientation
         */
        double orientation();
        /**
         * @brief eccentricity
         * @return the eccentricity of the ellipse
         */
        double eccentricity();
        /**
         * @brief approx
         * @return the approximate hull of the contour
         */
        std::vector<cv::Point> approx();
        /**
         * @brief filledimage
         * @return compute the filled image
         */
        cv::Mat filledimage();
        /**
         * @brief filledArea
         * @return the area of the filled image
         */
        double filledarea();
        /**
         * @brief pixellist
         * @return the array of indices of contour region
         */
        cv::Mat pixellist();
        /**
         * @brief conveimage
         * @return the convex image
         */
        cv::Mat conveximage();
        /**
         * @brief pixelparameters compute the pixel parameters: mean value, minvalue, maxvalue, maxloc, minloc
         */
        void pixelparameters();
        /**
         * @brief extrema compute extrame points
         */
        std::vector<cv::Point> extrema();
        Region region;
        std::vector<cv::Point> contour;
        cv::Mat img;

};


#endif
