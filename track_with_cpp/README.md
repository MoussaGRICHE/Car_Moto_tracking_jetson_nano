# Yolov8_TensorRT_JetsonNano

# clone the repo
    git clone https://github.com/MoussaGRICHE/Car_Moto_tracking_jetson_nano.git

    cd Car_Moto_tracking_jetson_nano

    $ export root=${PWD}

    $ model_name=$(ls $root/yolo_model | head -n1)
    $ base_name=${model_name%%.*}

    cd track_with_cpp

    conda activate {env_name}

    python3 export.py --weights ${root}/yolo_model/$model_name --sim


To export the engine with python:



To export the engine without python:

    /usr/src/tensorrt/bin/trtexec \
    --onnx=${root}/yolo_model/"${base_name}.onnx" \
    --saveEngine=${root}/yolo_model/"${base_name}.engine"

# Inference with c++

It is highly recommended to use C++ inference on Jetson. Here is a demo: csrc/jetson/detect .
Build:

Please modify CLASS_NAMES and COLORS in main.cpp for yourself.

And build:

    cd inference
    mkdir build
    cmake ./CMakeLists.txt
    make
    mv yolov8 ${root}/track_with_cpp
    cd ${root}/track_with_cpp
   
# test
    ./yolov8 ${root}/yolo_model/"${base_name}.engine" ./data/sample_1080p_h265.mp4

