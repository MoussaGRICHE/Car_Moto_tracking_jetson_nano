cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(yolov8_deepsort LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -g")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# CUDA
find_package(CUDA REQUIRED)
message(STATUS "CUDA Libs: \n${CUDA_LIBRARIES}\n")
message(STATUS "CUDA Headers: \n${CUDA_INCLUDE_DIRS}\n")

# OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV Libs: \n${OpenCV_LIBS}\n")
message(STATUS "OpenCV Libraries: \n${OpenCV_LIBRARIES}\n")
message(STATUS "OpenCV Headers: \n${OpenCV_INCLUDE_DIRS}\n")

# OnnxRuntime
set(ONNXRUNTIME_INCLUDE_DIR /usr/lib/onnxruntime-linux-x64-1.14.1/include/)
set(ONNXRUNTIME_LIBRARY /usr/lib/onnxruntime-linux-x64-1.14.1/lib/libonnxruntime.so)

# Verify that the files exist
if (NOT EXISTS ${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_c_api.h)
    message(FATAL_ERROR "OnnxRuntime include file not found: ${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_c_api.h")
endif()
if (NOT EXISTS ${ONNXRUNTIME_LIBRARY})
    message(FATAL_ERROR "OnnxRuntime library file not found: ${ONNXRUNTIME_LIBRARY}")
endif()

message(STATUS "OnnxRuntime Lib: \n${ONNXRUNTIME_LIBRARY}\n")
message(STATUS "OnnxRuntime Headers: \n${ONNXRUNTIME_INCLUDE_DIR}\n")

# TensorRT
set(TensorRT_INCLUDE_DIRS /usr/src/TensorRT)
set(TensorRT_LIBRARIES /usr/src/TensorRT)

# Eigen3
include_directories(/usr/include/eigen3)

message(STATUS "TensorRT Libs: \n${TensorRT_LIBRARIES}\n")
message(STATUS "TensorRT Headers: \n${TensorRT_INCLUDE_DIRS}\n")

list(APPEND INCLUDE_DIRS
        ${CUDA_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${TensorRT_INCLUDE_DIRS}
        ${ONNXRUNTIME_INCLUDE_DIR}
        ./include
        )

list(APPEND ALL_LIBS
        ${CUDA_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${TensorRT_LIBRARIES}
        ${ONNXRUNTIME_LIBRARY}
        )

include_directories(${INCLUDE_DIRS})

file(GLOB My_Source_Files ${PROJECT_SOURCE_DIR}/src/*.cpp)
add_executable(${PROJECT_NAME} ${My_Source_Files})

link_directories(${ALL_LIBS})
target_link_libraries(${PROJECT_NAME} PRIVATE nvinfer nvinfer_plugin ${CUDA_LIBRARIES} ${OpenCV_LIBS} ${ONNXRUNTIME_LIBRARY})
