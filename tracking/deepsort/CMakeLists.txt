cmake_minimum_required(VERSION 2.6)
project(deepsort)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE Release)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)


include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
)
link_directories(
    /usr/local/cuda/lib64
)

# Eigen3
include_directories(/usr/include/eigen3)

aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_DIR)

# ===== deepsort =====
add_library(deepsort SHARED ${SRC_DIR})
target_link_libraries(deepsort 
    ${CUDA_LIBS} ${OpenCV_LIBS} 
    cudart nvinfer nvonnxparser
)


# ===== yolov8_deepsort =====
add_executable(yolov8 ${PROJECT_SOURCE_DIR}/main.cpp)
target_link_libraries(yolov8 
    ${CUDA_LIBS} ${OpenCV_LIBS} 
    cudart nvinfer nvonnxparser deepsort
)

target_link_libraries(yolov8 /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so)