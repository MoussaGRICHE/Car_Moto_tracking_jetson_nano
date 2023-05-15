//
// Created by ubuntu on 3/16/23.
//
#include "chrono"
#include "yolov8.hpp"
#include "opencv2/opencv.hpp"

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
 int infer_rate = std::stoi(argv[4]);
 std::string output_type{ argv[5] };

 std::vector<std::string> imagePathList;
 bool isVideo{ false };
 bool isCamera{ false };

 auto yolov8 = new YOLOv8(engine_file_path);
 yolov8->make_pipe(true);

 if (input_type == "video")
 {
 assert(argc == 6);
 input_value = argv[3];
 if (IsFile(input_value))
 {
 std::string suffix = input_value.substr(input_value.find_last_of('.') + 1);
 if (
 suffix == "mp4" ||
 suffix == "avi" ||
 suffix == "m4v" ||
 suffix == "mpeg" ||
 suffix == "mov" ||
 suffix == "mkv"
 )
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
 double fps = cap.get(cv::CAP_PROP_FPS);
 cv::Size size = cv::Size((int)cap.get(cv::CAP_PROP_FRAME_WIDTH), (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT));
 if (output_type == "save") {
 writer.open("output.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, size);
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
 if (!cap.isOpened()) {
 std::cout << "Failed to open camera." << std::endl;
 return (-1);
 }
 }

 cv::Mat res, image;
 cv::Size size = cv::Size{ 640, 640 };
 std::vector<Object> objs;

 cv::namedWindow("result", cv::WINDOW_AUTOSIZE);

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
 }

 auto end = std::chrono::system_clock::now();
 yolov8->draw_objects(image, res, objs, CLASS_NAMES, COLORS, DISPALYED_CLASS_NAMES);

 if (output_type == "save") {
 writer.write(res);
 }

 auto tc = (double)
 std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;

 printf("cost %2.4lf ms\n", tc);


 if(output_type=="show"){
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




