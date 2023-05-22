cmake_minimum_required(VERSION 3.5)

project(DeepSORT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package (Eigen3 3.3 REQUIRED NO_MODULE)


set(ONNXRUNTIME_DIR "/lib/onnxruntime-linux-x64-1.14.1")

include_directories("${ONNXRUNTIME_DIR}/include"
    )
find_package(OpenCV 4 REQUIRED )


include_directories(
    ${OpenCV_INCLUDE_DIRS}/include
    ${CMAKE_SOURCE_DIR}/include

    )


add_executable(yolov8_DeepSORT
    ./src/FeatureTensor.cpp
    ./src/model.cpp
    ./src/kalmanfilter.cpp
    ./src/linear_assignment.cpp
    ./src/nn_matching.cpp
    ./src/track.cpp
    ./src/tracker.cpp
    ./src/munkres.cpp
    ./src/hungarianoper.cpp

    main.cpp)
target_link_libraries(yolov8_DeepSORT PRIVATE "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so" ${OpenCV_LIBS} Eigen3::Eigen)

