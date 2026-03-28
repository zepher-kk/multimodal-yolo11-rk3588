cmake_minimum_required(VERSION 3.10)
project(RknnMultiModal)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)

# 🌟 开启最高级别优化，释放 CPU 算力
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3 -ffast-math -pthread")

# 寻找 OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 包含头文件目录
include_directories(${CMAKE_SOURCE_DIR}/include)

# 编译流水线引擎和主程序
add_executable(rknn_infer src/main.cc src/yolo_pipeline.cc)

# 链接库
target_link_libraries(rknn_infer ${OpenCV_LIBS} rknnrt pthread)