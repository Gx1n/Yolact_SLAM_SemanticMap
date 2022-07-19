cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

rosbuild_init()

IF(NOT ROS_BUILD_TYPE)
  SET(ROS_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${ROS_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3 -march=native")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../../cmake_modules)

message("PROJECT_SOURCE_DIR: " ${OpenCV_DIR})
find_package(OpenCV 3.2.0 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

find_package(Qt5Widgets REQUIRED)
find_package(Qt5Concurrent REQUIRED)
find_package(Qt5OpenGL REQUIRED)
find_package(Qt5Test REQUIRED)
#set(PYTHON_LIBRARY "/home/x1/miniconda3/envs/segment/lib/libpython3.6m.so")
find_package(PythonLibs REQUIRED)
if (NOT PythonLibs_FOUND)
    message(FATAL_ERROR "PYTHON LIBS not found.")
else()
    message("PYTHON LIBS were found!")
    message("PYTHON LIBS DIRECTORY: " ${PYTHON_LIBRARY})
endif()

find_package(Boost REQUIRED COMPONENTS thread)
if(Boost_FOUND)
    message("Boost was found!")
    message("Boost Headers DIRECTORY: " ${Boost_INCLUDE_DIRS})
    message("Boost LIBS DIRECTORY: " ${Boost_LIBRARY_DIRS})
    message("Found Libraries: " ${Boost_LIBRARIES})
endif()

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)

set(PYTHON_INCLUDE_DIRS ${PYTHON_INCLUDE_DIRS}
        #/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy
        /home/x1/miniconda3/envs/yolact/lib/python3.6/site-packages/numpy/core/include/numpy
    )

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/../../../
${PROJECT_SOURCE_DIR}/../../../include
${Pangolin_INCLUDE_DIRS}
#/usr/include/python2.7/
/home/x1/miniconda3/envs/yolact/include/python3.6m
#/usr/lib/python2.7/dist-packages/numpy/core/include/numpy/
/home/x1/miniconda3/envs/yolact/lib/python3.6/site-packages/numpy/core/include/numpy
${Boost_INCLUDE_DIRS}
)

set(LIBS 
${OpenCV_LIBS} 
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/../../../Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/../../../Thirdparty/g2o/lib/libg2o.so
${PROJECT_SOURCE_DIR}/../../../lib/libORB_SLAM2.so
#/home/x1/catkin_ws/src/Yolact_SLAM_SemanticMap/lib/libDynaSLAM.so
#/usr/lib/x86_64-linux-gnu/libpython2.7.so
/home/x1/miniconda3/envs/yolact/lib/libpython3.6m.so
${Boost_LIBRARIES}
)

# Node for RGB-D camera
#rosbuild_add_executable(RGBD
#        src/ros_yolact.cc
#        )
rosbuild_add_executable(RGBD
        src/ros_yolact.cc
        )

target_link_libraries(RGBD
        ${LIBS}
        #ros和miniconda3冲突
        /usr/lib/x86_64-linux-gnu/libtiff.so.5
        )
## Node for monocular camera
#rosbuild_add_executable(Mono
#src/ros_mono.cc
#)
#
#target_link_libraries(Mono
#${LIBS}
#)
#
## Node for monocular camera (Augmented Reality Demo)
#rosbuild_add_executable(MonoAR
#src/AR/ros_mono_ar.cc
#src/AR/ViewerAR.h
#src/AR/ViewerAR.cc
#)
#
#target_link_libraries(MonoAR
#${LIBS}
#)
#
## Node for stereo camera
#rosbuild_add_executable(Stereo
#src/ros_stereo.cc
#)
#
#target_link_libraries(Stereo
#${LIBS}
#)

# Node for RGB-D camera
#rosbuild_add_executable(RGBD
#src/ros_rgbd.cc
#)

#target_link_libraries(RGBD
#${LIBS}
#)



