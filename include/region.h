/*
 *  Regionprops
 *  Copyright 2015 Andrea Pennisi
 * 联通域====区域属性 面积 周长 矩 矩形框 多边形框 凸包 旋转矩阵 等
 * matlab 的 Regionprops  https://blog.csdn.net/langb2014/article/details/49886787
   c++ 实现 
 */

#ifndef REGION_H
#define REGION_H

#include <opencv2/opencv.hpp>
#include <iostream>

class Region {
    public:
        /**
         * @brief Region new Object
         */
        Region() {}
        /**
         * @brief Area
         * @return return area  返回 区域面积
         */
        inline double Area() const
        {
            return area;
        }
        /**
         * @brief setArea set area to _area
         * @param _area
         */
        inline void setArea(const double &_area)
        {
            area = _area;// 设置 区域面积
        }
        /**
         * @brief Perimeter
         * @return return the perimeter
         */
        inline double Perimeter() const
        {
            return perimeter; // 返回 区域周长
        }
        /**
         * @brief setPerimeter set perimeter to _perimeter
         * @param _perimeter
         */
        inline void setPerimeter(const double &_perimeter)
        {
            perimeter = _perimeter;// 设置 区域周长
        }
        /**
         * @brief Moments return moments;
         * @param _moments
         * @return moments
         */
        inline cv::Moments Moments() const
        {
            return moments;//返回 区域  图像矩 具有平移、灰度、尺度、旋转不变性 
        }
        /**
         * @brief setMoments set moments to _moments
         * @param _moments
         */
        inline void setMoments(const cv::Moments &_moments)
        {
            moments = _moments;// 设置 区域  图像矩
        }
        /**
         * @brief BoundingBox
         * @return the boundingBox
         */
        inline cv::Rect BoundingBox() const
        {
            return boundingBox; // 返回区域 包围框 ===矩形
        }
        /**
         * @brief setBoundingBox set boundingBox to _boundingBox
         * @param _boundingBox
         */
        inline void setBoundingBox(const cv::Rect &_boundingBox)
        {
            boundingBox = _boundingBox;// 设置区域包围框===
        }
        /**
         * @brief ConvexHull
         * @return return the convex hull
         */
        inline std::vector<cv::Point> ConvexHull() const
        {
            return convex_hull; // 返回图像 凸包，最小包围多边形，多边形包围框
        }
        /**
         * @brief setConvexHull set convex_hull to _convex_hull
         * @param _convex_hull
         */
        inline void setConvexHull(const std::vector<cv::Point> &_convex_hull)
        {
            convex_hull = _convex_hull;// 设置多边形包围框
        }
        /**
         * @brief ConvexArea
         * @return the convex area
         */
        inline double ConvexArea() const
        {
            return convex_area; // 返回区域 多边形包围框面积
        }
        /**
         * @brief setConvexArea set convex_area to _convex_area
         * @param _convex_area
         */
        inline void setConvexArea(const double &_convex_area)
        {
            convex_area = _convex_area;// 设置 区域 多边形包围框面积
        }
        /**
         * @brief Ellipse
         * @return the ellipse
         */
        inline cv::RotatedRect Ellipse() const
        {
            return ellipse;// 返回  区域 旋转矩形， 中心点 长和宽 旋转角度
        }
        /**
         * @brief setEllipse set the ellipse to _ellipse
         * @param _ellipse
         */
        inline void setEllipse(const cv::RotatedRect &_ellipse)
        {
            ellipse = _ellipse;//  设置 区域 旋转矩形， 中心点 长和宽 旋转角度
        }
        /**
         * @brief Orientation
         * @return the orientation of the ellipse
         */
        inline double Orientation() const
        {
            return orientation;// 返回 区域 旋转矩形 旋转角度
        }
        /**
         * @brief setOrientation set orientation to _orientation
         * @param _orientation
         */
        inline void setOrientation(const double &_orientation)
        {
            orientation = _orientation;// 设置 区域 旋转矩形 旋转角度
        }
        /**
         * @brief MinorAxis
         * @return the ellipse minor axis
         */
        inline double MinorAxis() const
        {
            return minoraxis_length; // 返回 短轴长
        }
        /**
         * @brief setMinorAxis set minoraxis_length to minor_axis
         * @param minor_axis
         */
        inline void setMinorAxis(const double &minor_axis)
        {
            minoraxis_length = minor_axis;
        }
        /**
         * @brief MinorAxis
         * @return the ellipse minor axis
         */
        inline double MajorAxis() const
        {
            return majoraxis_length; // 返回长轴长
        }
        /**
         * @brief setMinorAxis set minoraxis_length to minor_axis
         * @param minor_axis
         */
        inline void setMajorAxis(const double &major_axis)
        {
            majoraxis_length = major_axis;
        }
        /**
         * @brief Approx
         * @return the approximate hull of the contour
         */
        inline std::vector<cv::Point> Approx() const
        {
            return approx;// 轮廓 的 近似 多边形
        }
        /**
         * @brief setApprox set approx to _approx
         * @param _approx
         */
        inline void setApprox(const std::vector<cv::Point> &_approx)
        {
            approx = _approx;
        }
        /**
         * @brief FilledImage
         * @return the image where region is white and others are black
         */
        inline cv::Mat FilledImage() const
        {
            return filledImage;// 返回填充的区域========
        }
        /**
         * @brief setFilledImage set filledImage to _filledImage
         * @param _filledImage
         */
        inline void setFilledImage(const cv::Mat &_filledImage)
        {
            filledImage = _filledImage;
        }
        /**
         * @brief Centroid
         * @return the centroid of the hull
         */
        inline cv::Point Centroid() const
        {
            return centroid;// 区域中心
        }
        /**
         * @brief setCentroid set the centroid to _centroid
         * @param _centroid
         */
        inline void setCentroid(const cv::Point &_centroid)
        {
            centroid = _centroid;
        }
        /**
         * @brief AspectRatio
         * @return the aspect ratio of the hull
         */
        inline double AspectRatio() const
        {
            return aspect_ratio;// 区域 横纵比=====
        }
        /**
         * @brief setAspectRatio set aspect_ratio to _aspect_ratio
         * @param _aspect_ratio
         */
        inline void setAspectRatio(const double &_aspect_ratio)
        {
            aspect_ratio = _aspect_ratio;
        }
        /**
         * @brief EquivalentDiameter
         * @return the equivalent diameter of the circle with same as area as that of region
         */
        inline double EquivalentDiameter() const
        {
            return equi_diameter; // 相同面积的 等效圆的直径
        }
        /**
         * @brief setEquivalentDiameter set equi_diameter to _equi_diameter
         * @param _equi_diameter
         */
        inline void setEquivalentDiameter(const double &_equi_diameter)
        {
            equi_diameter = _equi_diameter;
        }
        /**
         * @brief Eccentricity
         * @return the eccentricity of the ellipse
         */
        inline double Eccentricity() const
        {
            return eccentricity;//椭圆偏心率 
        }
        /**
         * @brief setEccentricity set the eccentricity to _eccentricity
         * @param _eccentricity
         */
        inline void setEccentricity(const double &_eccentricity)
        {
            eccentricity = _eccentricity;
        }
        /**
         * @brief FilledArea
         * @return the number of white pixels in filledImage
         */
        inline double FilledArea() const
        {
            return filledArea;// 像素数量====
        }
        /**
         * @brief setFilledArea set filledArea to _filledArea
         * @param _filledArea
         */
        inline void setFilledArea(const double &_filledArea)
        {
            filledArea = _filledArea;
        }
        /**
         * @brief PixelList
         * @return the array of indices of on-pixels in filledImage
         */
        inline cv::Mat PixelList() const
        {
            return pixelList;
        }
        /**
         * @brief setPixelList set pixelList to _pixelList
         * @param _pixelList
         */
        inline void setPixelList(const cv::Mat &_pixelList)
        {
            pixelList = _pixelList;
        }
        /**
         * @brief ConvexImage
         * @return the image where convex hull region is white and others are black
         */
        inline cv::Mat ConvexImage() const
        {
            return convexImage;
        }
        /**
         * @brief setConvexImage set convexImage to _convexImage
         * @param _convexImage
         */
        inline void setConvexImage(const cv::Mat &_convexImage)
        {
            convexImage = _convexImage;
        }
        /**
         * @brief MaxVal
         * @return the max intensity in the contour region
         */
        inline double MaxVal() const
        {
            return maxval;
        }
        /**
         * @brief setMaxVal set maxval to _maxval
         * @param _maxval
         */
        inline void setMaxVal(const double &_maxval)
        {
            maxval = _maxval;
        }
        /**
         * @brief MinVal
         * @return the min intensity in the contour region
         */
        inline double MinVal() const
        {
            return minval;
        }
        /**
         * @brief setMinVal set minval to _minval
         * @param _minval
         */
        inline void setMinVal(const double &_minval)
        {
            minval = _minval;
        }
        /**
         * @brief MaxLoc
         * @return the max.intensity pixel location
         */
        inline cv::Point MaxLoc() const
        {
            return maxloc;
        }
        /**
         * @brief setMaxLoc set maxloc to _maxLoc
         * @param _maxloc
         */
        inline void setMaxLoc(const cv::Point &_maxloc)
        {
            maxloc = _maxloc;
        }
        /**
         * @brief MinLoc
         * @return the min.intensity pixel location
         */
        inline cv::Point MinLoc() const
        {
            return minloc;
        }
        /**
         * @brief setMinLoc set minloc to _minloc
         * @param _minloc
         */
        inline void setMinLoc(const cv::Point &_minloc)
        {
            minloc = _minloc;
        }
        /**
         * @brief MeanVal
         * @return the mean intensity in the contour region
         */
        inline cv::Scalar MeanVal() const
        {
            return meanval;
        }
        /**
         * @brief setMeanVal set meanval to _meanval
         * @param _meanval
         */
        inline void setMeanVal(const cv::Scalar &_meanval)
        {
            meanval = _meanval;
        }
        /**
         * @brief Extreme
         * @return the extremal points in the region, respectively: rightMost, leftMost, topMost, bottomMost
         */
        inline std::vector<cv::Point> Extrema() const
        {
            return extrema;
        }
        /**
         * @brief setExtrema set extrame to _extrema
         * @param _extrema
         */
        inline void setExtrema(const std::vector<cv::Point> &_extrema)
        {
            extrema = _extrema;
        }
        /**
         * @brief Solidity
         * @return solidity = contour area / convex hull area
         */
        inline double Solidity() const
        {
            return solidity;
        }
        /**
         * @brief setSolidity set solidity to _solidity
         * @param _solidity
         */
        inline void setSolidity(const double &_solidity)
        {
            solidity = _solidity;
        }

    private:
        double area, perimeter;
        cv::Moments moments;
        cv::Point centroid;
        cv::Rect boundingBox;
        double aspect_ratio, equi_diameter, extent;
        std::vector< cv::Point> convex_hull;
        double convex_area, solidity;
        cv::Point center;
        double majoraxis_length, minoraxis_length;
        double orientation, eccentricity;
        cv::Mat filledImage, pixelList;
        double filledArea;
        cv::Mat convexImage;
        cv::RotatedRect ellipse;
        std::vector<cv::Point> approx;
        double maxval, minval;
        cv::Point maxloc, minloc;
        cv::Scalar meanval;
        std::vector<cv::Point> extrema;
};

#endif
