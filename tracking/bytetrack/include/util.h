
#ifndef UTIL_H
#define UTIL_H

#include <string>
#include <vector>
#include <cmath>
#include <list>


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

// Define a vector of displayed class names
const std::vector<std::string> DISPALYED_CLASS_NAMES = {
	"car", "motorcycle" };

// Function to generate the GStreamer pipeline string
std::string gstreamer_pipeline (int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           std::to_string(capture_height) + ", framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}


cv::Point crossingLine[4];
int clickCount = 0;


void onMouse(int event, int x, int y, int flags, void* userdata) {
    int count_line = *(static_cast<int*>(userdata));
    if (event == cv::EVENT_LBUTTONUP) {
        if (clickCount < count_line * 2) {
            crossingLine[clickCount].x = x;
            crossingLine[clickCount].y = y;
            std::cout << "Click " << clickCount + 1 << ": (" << crossingLine[clickCount].x << ", " << crossingLine[clickCount].y << ")\n";
            clickCount++;
        }
    }
}

std::list<double> calculateAngles(const cv::Point* crossingLine, int count_line)
{
    std::list<double> angles;

    double deltaX1 = static_cast<double>(crossingLine[1].x - crossingLine[0].x);
    double deltaY1 = static_cast<double>(crossingLine[1].y - crossingLine[0].y);

    // Check if the line is horizontal
    if (deltaY1 == 0)
    {
        if (deltaX1 >= 0)
        {
            angles.push_back(0.0); // Angle is 0 degrees
        }
        else
        {
            angles.push_back(180.0); // Angle is 180 degrees
        }
    }
    else
    {
        // Calculate the first angle using atan2
        double angle1 = std::atan2(deltaY1, deltaX1) * 180.0 / M_PI;

        // Adjust the first angle to be between 0 and 360 degrees
        if (angle1 < 0)
        {
            angle1 += 360.0;
        }

        angles.push_back(angle1);
    }

    if (count_line == 2)
    {
        double deltaX2 = static_cast<double>(crossingLine[3].x - crossingLine[2].x);
        double deltaY2 = static_cast<double>(crossingLine[3].y - crossingLine[2].y);

        // Check if the line is horizontal
        if (deltaY2 == 0)
        {
            if (deltaX2 >= 0)
            {
                angles.push_back(0.0); // Angle is 0 degrees
            }
            else
            {
                angles.push_back(180.0); // Angle is 180 degrees
            }
        }
        else
        {
            // Calculate the second angle using atan2
            double angle2 = std::atan2(deltaY2, deltaX2) * 180.0 / M_PI;

            // Adjust the second angle to be between 0 and 360 degrees
            if (angle2 < 0)
            {
                angle2 += 360.0;
            }

            angles.push_back(angle2);
        }
    }

    return angles;
}

bool hasPassedLine(const cv::Point& lineStart, const cv::Point& lineEnd, const cv::Point& point)
{
    cv::Point lineVec(lineEnd.x - lineStart.x, lineEnd.y - lineStart.y);
    cv::Point pointVec(point.x - lineStart.x, point.y - lineStart.y);

    int crossProduct = lineVec.x * pointVec.y - lineVec.y * pointVec.x;

    // If the cross product is positive and the point is within the line segment, it has passed the line
    return crossProduct > 0 && point.x >= std::min(lineStart.x, lineEnd.x) && point.x <= std::max(lineStart.x, lineEnd.x) && point.y >= std::min(lineStart.y, lineEnd.y) && point.y <= std::max(lineStart.y, lineEnd.y);
}




#endif // UTIL_H