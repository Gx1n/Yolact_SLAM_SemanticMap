echo "Building ROS nodes"

cd Examples/ROS/Yolact_SLAM_SemanticMap
mkdir build
cd build
cmake .. -DROS_BUILD_TYPE=Release
make -j6
