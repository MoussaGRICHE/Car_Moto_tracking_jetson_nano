# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/tts/Car_Moto_tracking_jetson_nano/track_with_cpp/inference

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/tts/Car_Moto_tracking_jetson_nano/track_with_cpp/inference

# Include any dependencies generated for this target.
include CMakeFiles/yolov8.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/yolov8.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/yolov8.dir/flags.make

CMakeFiles/yolov8.dir/main.cpp.o: CMakeFiles/yolov8.dir/flags.make
CMakeFiles/yolov8.dir/main.cpp.o: main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tts/Car_Moto_tracking_jetson_nano/track_with_cpp/inference/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/yolov8.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/yolov8.dir/main.cpp.o -c /home/tts/Car_Moto_tracking_jetson_nano/track_with_cpp/inference/main.cpp

CMakeFiles/yolov8.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/yolov8.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tts/Car_Moto_tracking_jetson_nano/track_with_cpp/inference/main.cpp > CMakeFiles/yolov8.dir/main.cpp.i

CMakeFiles/yolov8.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/yolov8.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tts/Car_Moto_tracking_jetson_nano/track_with_cpp/inference/main.cpp -o CMakeFiles/yolov8.dir/main.cpp.s

CMakeFiles/yolov8.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/yolov8.dir/main.cpp.o.requires

CMakeFiles/yolov8.dir/main.cpp.o.provides: CMakeFiles/yolov8.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/yolov8.dir/build.make CMakeFiles/yolov8.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/yolov8.dir/main.cpp.o.provides

CMakeFiles/yolov8.dir/main.cpp.o.provides.build: CMakeFiles/yolov8.dir/main.cpp.o


# Object files for target yolov8
yolov8_OBJECTS = \
"CMakeFiles/yolov8.dir/main.cpp.o"

# External object files for target yolov8
yolov8_EXTERNAL_OBJECTS =

yolov8: CMakeFiles/yolov8.dir/main.cpp.o
yolov8: CMakeFiles/yolov8.dir/build.make
yolov8: /usr/local/cuda/lib64/libcudart.so
yolov8: /usr/lib/aarch64-linux-gnu/libopencv_dnn.so.4.1.1
yolov8: /usr/lib/aarch64-linux-gnu/libopencv_gapi.so.4.1.1
yolov8: /usr/lib/aarch64-linux-gnu/libopencv_highgui.so.4.1.1
yolov8: /usr/lib/aarch64-linux-gnu/libopencv_ml.so.4.1.1
yolov8: /usr/lib/aarch64-linux-gnu/libopencv_objdetect.so.4.1.1
yolov8: /usr/lib/aarch64-linux-gnu/libopencv_photo.so.4.1.1
yolov8: /usr/lib/aarch64-linux-gnu/libopencv_stitching.so.4.1.1
yolov8: /usr/lib/aarch64-linux-gnu/libopencv_video.so.4.1.1
yolov8: /usr/lib/aarch64-linux-gnu/libopencv_videoio.so.4.1.1
yolov8: /usr/lib/aarch64-linux-gnu/libopencv_imgcodecs.so.4.1.1
yolov8: /usr/lib/aarch64-linux-gnu/libopencv_calib3d.so.4.1.1
yolov8: /usr/lib/aarch64-linux-gnu/libopencv_features2d.so.4.1.1
yolov8: /usr/lib/aarch64-linux-gnu/libopencv_flann.so.4.1.1
yolov8: /usr/lib/aarch64-linux-gnu/libopencv_imgproc.so.4.1.1
yolov8: /usr/lib/aarch64-linux-gnu/libopencv_core.so.4.1.1
yolov8: CMakeFiles/yolov8.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/tts/Car_Moto_tracking_jetson_nano/track_with_cpp/inference/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable yolov8"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/yolov8.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/yolov8.dir/build: yolov8

.PHONY : CMakeFiles/yolov8.dir/build

CMakeFiles/yolov8.dir/requires: CMakeFiles/yolov8.dir/main.cpp.o.requires

.PHONY : CMakeFiles/yolov8.dir/requires

CMakeFiles/yolov8.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/yolov8.dir/cmake_clean.cmake
.PHONY : CMakeFiles/yolov8.dir/clean

CMakeFiles/yolov8.dir/depend:
	cd /home/tts/Car_Moto_tracking_jetson_nano/track_with_cpp/inference && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/tts/Car_Moto_tracking_jetson_nano/track_with_cpp/inference /home/tts/Car_Moto_tracking_jetson_nano/track_with_cpp/inference /home/tts/Car_Moto_tracking_jetson_nano/track_with_cpp/inference /home/tts/Car_Moto_tracking_jetson_nano/track_with_cpp/inference /home/tts/Car_Moto_tracking_jetson_nano/track_with_cpp/inference/CMakeFiles/yolov8.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/yolov8.dir/depend

