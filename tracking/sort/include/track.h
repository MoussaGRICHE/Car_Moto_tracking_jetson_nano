#pragma once

#include <opencv2/core.hpp>
#include "kalman_filter.h"

class Track {
public:
 // Constructor
 Track();

 // Destructor
 ~Track() = default;

 void Init(const cv::Rect_<float>& bbox);
 void Predict();
 void Update(const cv::Rect_<float>& bbox);
 cv::Rect_<float> GetStateAsBbox() const;
 float GetNIS() const;

 int coast_cycles_ = 0, hit_streak_ = 0;
 int label; // added label member

private:
 Eigen::VectorXd ConvertBboxToObservation(const cv::Rect_<float>& bbox) const;
 cv::Rect_<float> ConvertStateToBbox(const Eigen::VectorXd &state) const;

 KalmanFilter kf_;
};

