//
// Created by ubuntu on 3/16/23.
//
#include "chrono"
#include "yolov8.hpp"
#include "opencv2/opencv.hpp"
#include "FeatureTensor.h"
#include "tracker.h"

const std::vector<std::string> CLASS_NAMES = {
	"car", "motorcycle", "person", "bicycle", "airplane", "bus",
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


int main(int argc, char** argv) {
    // Read command line arguments
    const std::string engine_file_path{argv[1]};
    const std::string input_type{argv[2]};
    std::string input_value;
    int infer_rate;
    std::string output_type;

    // Variables for video processing
    std::vector<std::string> imagePathList;
    bool isVideo{false};
    bool isCamera{false};


	// Initialize DeepSort tracker
    const int nn_budget = 100;
    const float max_cosine_distance = 0.2;
    tracker mytracker(max_cosine_distance, nn_budget);

    // Create an instance of the YOLOv8 object detector
    auto yolov8 = new YOLOv8(engine_file_path);
    yolov8->make_pipe(true);

    // Process based on input type
    if (input_type == "video") {
        assert(argc == 6);
        input_value = argv[3];
        infer_rate = std::stoi(argv[4]);
        output_type = argv[5];
        if (IsFile(input_value)) {
            std::string suffix = input_value.substr(input_value.find_last_of('.') + 1);
            // Check if the input video file has a supported format
            if (suffix == "mp4" || suffix == "avi" || suffix == "m4v" || suffix == "mpeg" || suffix == "mov" || suffix == "mkv") {
                isVideo = true;
            } else {
                printf("suffix %s is wrong !!!\n", suffix.c_str());
                std::abort();
            }
        }
    } else if (input_type == "camera") {
        assert(argc == 5);
        infer_rate = std::stoi(argv[3]);
        output_type = argv[4];
        isCamera = true;
    }

    // Initialize OpenCV video capture and video writer
    cv::VideoCapture cap;
    cv::VideoWriter writer;
    if (isVideo) {
        cap.open(input_value);
        if (!cap.isOpened()) {
            printf("can not open %s\n", input_value.c_str());
            return -1;
        }

        // Get video frame size
        cv::Size size = cv::Size((int)cap.get(cv::CAP_PROP_FRAME_WIDTH), (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        if (output_type == "save") {
            // Generate a new filename for the output video
            auto t = std::time(nullptr);
            auto tm = *std::localtime(&t);
            std::ostringstream oss;
            oss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
            auto str = oss.str();
            size_t lastindex = input_value.find_last_of(".");
            size_t lastSlash = input_value.find_last_of('/');
            size_t lastDot = input_value.find_last_of('.');
            std::string rawname = input_value.substr(lastSlash + 1, lastDot - lastSlash - 1);
            std::string new_filename = rawname + "_detection_" + str + ".mp4";
            writer.open(new_filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30, size);
        }
    } else {
        // Settings for camera input
        int capture_width = 1280;
        int capture_height = 720;
        int display_width = 1280;
        int display_height = 720;
        int framerate = 30;
        int flip_method = 2;

        // Generate the GStreamer pipeline string
        std::string pipeline = gstreamer_pipeline(capture_width, capture_height, display_width, display_height, framerate, flip_method);
        std::cout << "Using pipeline: \n\t" << pipeline << "\n";

        // Open the camera using the GStreamer pipeline
        cap.open(pipeline, cv::CAP_GSTREAMER);
        if (!cap.isOpened()) {
            std::cout << "Failed to open camera." << std::endl;
            return -1;
        }

        // Get camera frame size
        cv::Size size = cv::Size((int)cap.get(cv::CAP_PROP_FRAME_WIDTH), (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        if (output_type == "save") {
            // Generate a new filename for the output video
            auto t = std::time(nullptr);
            auto tm = *std::localtime(&t);
            std::ostringstream oss;
            oss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
            auto str = oss.str();
            size_t lastindex = input_value.find_last_of(".");
            std::string rawname = input_value.substr(0, lastindex);
            std::string new_filename = "Camera_detection_" + str + ".mp4";
            writer.open(new_filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30, size);
        }
    }

    // Get the frame rate of the video
    int fps = cap.get(cv::CAP_PROP_FPS);

    cv::Mat res, image;
    cv::Size size = cv::Size{ 640, 640 };
    std::vector<Object> objs;
    std::vector<DETECTION_ROW> detections;
    std::vector<RESULT_DATA> result;
    int frame_count = 0;

    while (cap.read(image))
    {
        objs.clear();

        result.clear();

        auto start = std::chrono::system_clock::now();

        yolov8->copy_from_Mat(image, size);
        yolov8->infer();
        yolov8->postprocess(objs);

        detections.clear();
        
        // Convert objects to detections in the right format
        for (const Object& obj : objs)
        {
            DETECTION_ROW detection;
            detection.tlwh = DETECTBOX(obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);
            detection.confidence = obj.prob;
            detections.push_back(detection);
        }

        //if(FeatureTensor::getInstance()->getRectsFeature(image, detections))
        //{   
            mytracker.predict();

            mytracker.update(detections);


            
            for (Track& track : mytracker.tracks)
            {
                if (!track.is_confirmed() || track.time_since_update > 1)
                    continue;
                result.push_back(std::make_pair(track.track_id, track.to_tlwh()));
            }

            for(unsigned int k = 0; k < detections.size(); k++)
                {
                    DETECTBOX tmpbox = detections[k].tlwh;
                    cv::Rect rect(tmpbox(0), tmpbox(1), tmpbox(2), tmpbox(3));
                    //cv::rectangle(image, rect, cv::Scalar(0,0,255), 4);

                    for(unsigned int k = 0; k < result.size(); k++)
                    {
                        DETECTBOX tmp = result[k].second;
                        cv::Rect rect = cv::Rect(tmp(0), tmp(1), tmp(2), tmp(3));
                        rectangle(image, rect, cv::Scalar(255, 255, 0), 2);

                        std::string label = cv::format("%d", result[k].first);
                        cv::putText(image, label, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
                    }
                }
        //}

        auto end = std::chrono::system_clock::now();
        double tc = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
        double infer_fps = (1000.0 / tc) * infer_rate;
        printf("cost %2.4lf ms (%0.0lf fps, 1/ %d frame traited)\n", tc, std::round(infer_fps), infer_rate);

        if (output_type == "save")
        {
            writer.write(res);
        }

        if (output_type == "show")
        {
            // Show the result image
            cv::namedWindow("result", cv::WINDOW_NORMAL | cv::WINDOW_GUI_EXPANDED);
            cv::setWindowProperty("result", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
            cv::imshow("result", image);
            if (cv::waitKey(10) == 'q')
            {
                break;
            }
        }

        frame_count++;
    }

    // Clean up resources
    cv::destroyAllWindows();
    delete yolov8;
    return 0;
}