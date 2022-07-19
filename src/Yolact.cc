//
// Created by x1 on 2022/7/4.
//

#include "Yolact.h"
#include <iostream>


namespace ORB_SLAM2
{
    Yolact::Yolact()
    {
        std::cout << "Importing Yolact Settings..." << std::endl;
        Py_SetPythonHome(L"/home/x1/miniconda3/envs/yolact");
        Py_Initialize();
        PyRun_SimpleString("import sys");
        PyRun_SimpleString("sys.path.append('/home/x1/catkin_ws/src/Yolact_SLAM_SemanticMap/src/yolact')");

        cvt = new NDArrayConverter();
        py_module = PyImport_ImportModule("YOLACT1");
        if (!py_module) {
            std::cout << "YOLACT.py文件没找到" << std::endl;
            assert(py_module != NULL);
        }
        py_dict = PyModule_GetDict(py_module); //获取模块字典属性
        if (!py_dict) {
            std::cout << "Can't find py_module!" << std::endl;
            assert(py_dict != NULL);
        }
        //获取Yolact类
        py_class = PyDict_GetItemString(py_dict, "YOLACT"); //通过字典属性获取模块中的类
        if (!py_class) {
            std::cout << "Can't find Yolact class!" << std::endl;
            assert(py_class != NULL);
        }
        //python2.7版本命令 PyObject *pInstance = PyInstance_New(pClass, NULL, NULL);
        //这个命令不能实例化python类 net = PyInstanceMethod_New(py_class); //实例化获取的类
        net = PyObject_CallObject(py_class, nullptr); //实例化获取的类
        if (!net) {
            std::cout << "Can't find Yolact instance!" << std::endl;
            assert(net != NULL);
        }
        std::cout << "Created net instance" << std::endl;
        //cv::Mat image = cv::Mat::zeros(480, 640, CV_8UC3); //Be careful with size!!
        //std::cout << "Loading net parameters..." << std::endl;
    }

    std::pair<cv::Mat,cv::Mat> Yolact::GetSegmentation(cv::Mat &image)
    {
        std::pair<cv::Mat, cv::Mat> result;
        PyObject* mask;
        PyObject* img;
        py_image = cvt->toNDArray(image.clone());
        assert(py_image != NULL);
        //args = Py_BuildValue("(O)",py_image);
        //py_mask_image = PyObject_CallMethod(net, "GetDynSeg","OO",net,args);
        py_mask_image = PyObject_CallMethod(net, "GetDynSeg","O",py_image);
        PyArg_ParseTuple(py_mask_image,"O|O",&mask,&img);
        cv::Mat mask_ = cvt->toMat(mask).clone();
        cv::Mat img_ = cvt->toMat(img).clone();
        cvtColor(mask_,mask_,CV_RGB2GRAY);
        mask_.cv::Mat::convertTo(mask_,CV_8U);//0 background y 1 foreground

        //std::cout<<img_.type()<<img_.channels()<<std::endl;
        //img_.cv::Mat::convertTo(img_,CV_8U);
        result = std::make_pair(mask_,img_);
        /*if (!py_mask_image) {
            std::cout << "py_mask_image == NULL!" << std::endl;
            assert(py_mask_image != NULL);
        }*/
        //cv::Mat seg = cvt->toMat(py_mask_image).clone();
        //seg.cv::Mat::convertTo(seg,CV_8U);//0 background y 1 foreground
        //seg.cv::Mat::convertTo(seg,CV_8UC1);
        //seg.convertTo(seg,CV_8UC1);
        //seg.cv::Mat::convertTo(seg,CV_8UC1);//0 background y 1 foreground

        /*if (dir.compare("no_save") != 0) {
            mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            cv::imwrite(dir + "/" + name, seg);*/
        return result;
    }
    //ros
    cv::Mat Yolact::ros_GetSegmentation(cv::Mat &image)
    {
        py_image = cvt->toNDArray(image.clone());
        assert(py_image != NULL);
        py_mask_image = PyObject_CallMethod(net, "GetDynSeg","O",py_image);
        cv::Mat seg = cvt->toMat(py_mask_image).clone();
        return seg;
    }

    Yolact::~Yolact()
    {
        delete this->cvt;
        delete this->py_module;
        delete this->py_dict;
        delete this->py_class;
        delete this->net;
    }
}