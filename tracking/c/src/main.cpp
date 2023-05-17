//
// Created by ubuntu on 3/16/23.
//
#include "yolov8.hpp"
#include "opencv2/opencv.hpp"
#include <ctime>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <iostream>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "BYTETracker.h"

const std::vector<std::string> CLASS_NAMES = {
	"person", "bicycle", "car", "motorcycle", "airplane", "bus",
	"train", "truck", "boat", "traffic light", "fire hydrant",
	"stop sign", "parking meter", "bench", "bird", "cat",
	"dog", "horse", "sheep", "cow", "elephant",
	"bear", "zebra", "giraffe", "backpack", "umbrella",
	"handbag", "tie", "suitcase", "frisbee", "skis",
	"snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
	"skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
	"cup", "fork", "knife", "spoon", "bowl",
	"banana", "apple", "sandwich", "orange", "broccoli",
	"carrot", "hot dog", "pizza", "donut", "cake",
	"chair", "couch", "potted plant", "bed", "dining table",
	"toilet", "tv", "laptop", "mouse", "remote",
	"keyboard", "cell phone", "microwave", "oven",
	"toaster", "sink", "refrigerator", "book", "clock", "vase",
	"scissors", "teddy bear", "hair drier", "toothbrush" };

const std::vector<std::vector<unsigned int>> COLORS = {
	{ 0, 114, 189 }, { 217, 83, 25 }, { 237, 177, 32 },
	{ 126, 47, 142 }, { 119, 172, 48 }, { 77, 190, 238 },
	{ 162, 20, 47 }, { 76, 76, 76 }, { 153, 153, 153 },
	{ 255, 0, 0 }, { 255, 128, 0 }, { 191, 191, 0 },
	{ 0, 255, 0 }, { 0, 0, 255 }, { 170, 0, 255 },
	{ 85, 85, 0 }, { 85, 170, 0 }, { 85, 255, 0 },
	{ 170, 85, 0 }, { 170, 170, 0 }, { 170, 255, 0 },
	{ 255, 85, 0 }, { 255, 170, 0 }, { 255, 255, 0 },
	{ 0, 85, 128 }, { 0, 170, 128 }, { 0, 255, 128 },
	{ 85, 0, 128 }, { 85, 85, 128 }, { 85, 170, 128 },
	{ 85, 255, 128 }, { 170, 0, 128 }, { 170, 85, 128 },
	{ 170, 170, 128 }, { 170, 255, 128 }, { 255, 0, 128 },
	{ 255, 85, 128 }, { 255, 170, 128 }, { 255, 255, 128 },
	{ 0, 85, 255 }, { 0, 170, 255 }, { 0, 255, 255 },
	{ 85, 0, 255 }, { 85, 85, 255 }, { 85, 170, 255 },
	{ 85, 255, 255 }, { 170, 0, 255 }, { 170, 85, 255 },
	{ 170, 170, 255 }, { 170, 255, 255 }, { 255, 0, 255 },
	{ 255, 85, 255 }, { 255, 170, 255 }, { 85, 0, 0 },
	{ 128, 0, 0 }, { 170, 0, 0 }, { 212, 0, 0 },
	{ 255, 0, 0 }, { 0, 43, 0 }, { 0, 85, 0 },
	{ 0, 128, 0 }, { 0, 170, 0 }, { 0, 212, 0 },
	{ 0, 255, 0 }, { 0, 0, 43 }, { 0, 0, 85 },
	{ 0, 0, 128 }, { 0, 0, 170 }, { 0, 0, 212 },
	{ 0, 0, 255 }, { 0, 0, 0 }, { 36, 36, 36 },
	{ 73, 73, 73 }, { 109, 109, 109 }, { 146, 146, 146 },
	{ 182, 182, 182 }, { 219, 219, 219 }, { 0, 114, 189 },
	{ 80, 183, 189 }, { 128, 128, 0 }
};

const std::vector<std::string> DISPALYED_CLASS_NAMES = {
	"car", "motorcycle" };

std::string gstreamer_pipeline (int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           std::to_string(capture_height) + ", framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}


int main(int argc, char** argv)
{
    const std::string engine_file_path{ argv[1] };
    const std::string input_type{ argv[2] };
    std::string input_value;
    int infer_rate;
    std::string output_type;

    std::vector<std::string> imagePathList;
    bool isVideo{ false };
    bool isCamera{ false };

    auto yolov8 = new YOLOv8(engine_file_path);
    yolov8->make_pipe(true);

    if (input_type == "video")
		{
			assert(argc == 6);
			input_value = argv[3];
			infer_rate = std::stoi(argv[4]);
			output_type = argv[5];
			if (IsFile(input_value))
				{
					std::string suffix = input_value.substr(input_value.find_last_of('.') + 1);
					if (suffix == "mp4" || suffix == "avi" || suffix == "m4v" || suffix == "mpeg" || suffix == "mov" || suffix == "mkv")
						{
							isVideo = true;
						}
					else
						{
							printf("suffix %s is wrong !!!\n", suffix.c_str());
							std::abort();
						}
				}
		}

    else if (input_type == "camera")
		{
			assert(argc == 5);
			infer_rate = std::stoi(argv[3]);
			output_type = argv[4];
			isCamera = true;
		}

    cv::VideoCapture cap;
    cv::VideoWriter writer;
    if (isVideo)
		{
			cap.open(input_value);
			if (!cap.isOpened())
				{
					printf("can not open %s\n", input_value.c_str());
					return -1;
				}
			
			cv::Size size = cv::Size((int)cap.get(cv::CAP_PROP_FRAME_WIDTH), (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT));
			if (output_type == "save") 
				{
					// Get current time
					auto t = std::time(nullptr);
					auto tm = *std::localtime(&t);

					// Format date and time
					std::ostringstream oss;
					oss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
					auto str = oss.str();

					// Get filename without extension
					size_t lastindex = input_value.find_last_of("."); 
					size_t lastSlash = input_value.find_last_of('/');
					size_t lastDot = input_value.find_last_of('.');
					std::string rawname = input_value.substr(lastSlash + 1, lastDot - lastSlash - 1);

					// Construct new filename
					std::string new_filename = rawname + "_detection_" + str + ".mp4";

					writer.open(new_filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30, size);  
				}
		}
    else
		{
			int capture_width = 1280;
			int capture_height = 720;
			int display_width = 1280;
			int display_height = 720;
			int framerate = 30;
			int flip_method = 2;

			std::string pipeline = gstreamer_pipeline(capture_width,
				capture_height,
				display_width,
				display_height,
				framerate,
				flip_method);
			std::cout << "Using pipeline: \n\t" << pipeline << "\n";

			cap.open(pipeline, cv::CAP_GSTREAMER);
			if (!cap.isOpened()) 
				{
					std::cout << "Failed to open camera." << std::endl;
					return (-1);
				}

			cv::Size size = cv::Size((int)cap.get(cv::CAP_PROP_FRAME_WIDTH), (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT));
			if (output_type == "save") 
				{
					// Get current time
					auto t = std::time(nullptr);
					auto tm = *std::localtime(&t);

					// Format date and time
					std::ostringstream oss;
					oss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
					auto str = oss.str();

					// Get filename without extension
					size_t lastindex = input_value.find_last_of("."); 
					std::string rawname = input_value.substr(0, lastindex);

					// Construct new filename
					std::string new_filename = "Camera_detection_" + str + ".mp4";

					writer.open(new_filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30, size);
					
				}
		}

	cv::Mat res, image;
	cv::Size size = cv::Size{ 640, 640 };
	std::vector<Object> objs;

	BYTETracker tracker(fps, 30);

	int frame_count = 0;

	while (cap.read(image))
	{
		objs.clear();
		yolov8->copy_from_Mat(image, size);
		auto start = std::chrono::system_clock::now();

		if (frame_count % infer_rate == 0)
			{
				yolov8->infer();
				yolov8->postprocess(objs);
				vector<STrack> output_stracks = tracker.update(objs);
			}
		auto end = std::chrono::system_clock::now();
		for (int i = 0; i < output_stracks.size(); i++)
		{
			vector<float> tlwh = output_stracks[i].tlwh;
			bool vertical = tlwh[2] / tlwh[3] > 1.6;
			if (tlwh[2] * tlwh[3] > 20 && !vertical)
			{
				Scalar s = tracker.get_color(output_stracks[i].track_id);
				putText(res, format("%d", output_stracks[i].track_id), Point(tlwh[0], tlwh[1] - 5), 
						0, 0.6, Scalar(0, 0, 255), 2, LINE_AA);
				rectangle(res, Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
			}
		}
		putText(img, format("frame: %d fps: %d num: %d", num_frames, num_frames * 1000000 / total_ms, output_stracks.size()), 
				Point(0, 30), 0, 0.6, Scalar(0, 0, 255), 2, LINE_AA);


		if (output_type == "save") 
			{
				writer.write(res);
			}

		auto tc = (double)
		std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
		double fps = 1000 / tc;

		printf("cost %2.4lf ms (%0.0lf fps)\n", tc, std::round(fps));


		if(output_type=="show")
		{
		cv::namedWindow("result", cv::WINDOW_NORMAL | cv::WINDOW_GUI_EXPANDED);
		cv::setWindowProperty("result", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
		cv::imshow("result", res);
		if (cv::waitKey(10) == 'q')
			{
			break;
			}
		}

		frame_count++;
	}

	cv::destroyAllWindows();
	delete yolov8;
	return 0;
}





