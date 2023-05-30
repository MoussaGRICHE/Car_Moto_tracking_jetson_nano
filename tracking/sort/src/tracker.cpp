#include "tracker.h"
#include "common.hpp"

Tracker::Tracker() {
    id = 0;
}

float Tracker::CalculateIou(const cv::Rect& det, const Track& track) {
    auto trk = track.GetStateAsBbox();
    // get min/max points
    auto xx1 = std::max(static_cast<float>(det.tl().x), trk.tl().x);
    auto yy1 = std::max(static_cast<float>(det.tl().y), trk.tl().y);
    auto xx2 = std::min(static_cast<float>(det.br().x), trk.br().x);
    auto yy2 = std::min(static_cast<float>(det.br().y), trk.br().y);
   auto w = std::max(0.0f, static_cast<float>(xx2 - xx1));
    auto h = std::max(0.0f, static_cast<float>(yy2 - yy1));



    // calculate area of intersection and union
    float det_area = det.area();
    float trk_area = trk.area();
    auto intersection_area = w * h;
    float union_area = det_area + trk_area - intersection_area;
    auto iou = intersection_area / union_area;
    return iou;
}


void Tracker::HungarianMatching(const std::vector<std::vector<float>>& iou_matrix,
                                size_t nrows, size_t ncols,
                                std::vector<std::vector<float>>& association) {
    Matrix<float> matrix(nrows, ncols);
    // Initialize matrix with IOU values
    for (size_t i = 0 ; i < nrows ; i++) {
        for (size_t j = 0 ; j < ncols ; j++) {
            // Multiply by -1 to find max cost
            if (iou_matrix[i][j] != 0) {
                matrix(i, j) = -iou_matrix[i][j];
            }
            else {
                // TODO: figure out why we have to assign value to get correct result
                matrix(i, j) = 1.0f;
            }
        }
    }

//    // Display begin matrix state.
//    for (size_t row = 0 ; row < nrows ; row++) {
//        for (size_t col = 0 ; col < ncols ; col++) {
//            std::cout.width(10);
//            std::cout << matrix(row,col) << ",";
//        }
//        std::cout << std::endl;
//    }
//    std::cout << std::endl;


    // Apply Kuhn-Munkres algorithm to matrix.
    Munkres<float> m;
    m.solve(matrix);

//    // Display solved matrix.
//    for (size_t row = 0 ; row < nrows ; row++) {
//        for (size_t col = 0 ; col < ncols ; col++) {
//            std::cout.width(2);
//            std::cout << matrix(row,col) << ",";
//        }
//        std::cout << std::endl;
//    }
//    std::cout << std::endl;

    for (size_t i = 0 ; i < nrows ; i++) {
        for (size_t j = 0 ; j < ncols ; j++) {
            association[i][j] = matrix(i, j);
        }
    }
}


void Tracker::AssociateDetectionsToTrackers(const std::vector<cv::Rect>& detection,
                                            std::map<int, Track>& tracks,
                                            std::map<int, cv::Rect>& matched,
                                            std::vector<cv::Rect>& unmatched_det,
                                            float iou_threshold) {

    // Set all detection as unmatched if no tracks existing
    if (tracks.empty()) {
        for (const auto& det : detection) {
            unmatched_det.push_back(det);
        }
        return;
    }

    std::vector<std::vector<float>> iou_matrix;
    // resize IOU matrix based on number of detection and tracks
    iou_matrix.resize(detection.size(), std::vector<float>(tracks.size()));

    std::vector<std::vector<float>> association;
    // resize association matrix based on number of detection and tracks
    association.resize(detection.size(), std::vector<float>(tracks.size()));


    // row - detection, column - tracks
    for (size_t i = 0; i < detection.size(); i++) {
        size_t j = 0;
        for (const auto& trk : tracks) {
            iou_matrix[i][j] = CalculateIou(detection[i], trk.second);
            j++;
        }
    }

    // Find association
    HungarianMatching(iou_matrix, detection.size(), tracks.size(), association);

    for (size_t i = 0; i < detection.size(); i++) {
        bool matched_flag = false;
        size_t j = 0;
        for (const auto& trk : tracks) {
            if (0 == association[i][j]) {
                // Filter out matched with low IOU
                if (iou_matrix[i][j] >= iou_threshold) {
                    matched[trk.first] = detection[i];
                    matched_flag = true;
                }
                // It builds 1 to 1 association, so we can break from here
                break;
            }
            j++;
        }
        // if detection cannot match with any tracks
        if (!matched_flag) {
            unmatched_det.push_back(detection[i]);
        }
    }
}


void Tracker::Run(const std::vector<det::Object>& objs) {
    /*** Predict internal tracks from previous frame ***/
    for (auto &track : tracks_) {
        track.second.Predict();
    }
    

    // Extract bounding boxes from detections
    std::vector<cv::Rect> bboxes;
    for (const auto& obj : objs) {
        bboxes.push_back(obj.rect);
    }

    // Hash-map between track ID and associated detection bounding box
    std::map<int, cv::Rect> matched;
    // vector of unassociated detections
    std::vector<cv::Rect> unmatched_det;

    // return values - matched, unmatched_det
    if (!bboxes.empty()) {
        AssociateDetectionsToTrackers(bboxes, tracks_, matched, unmatched_det);
    }

    /*** Update tracks with associated bbox ***/
    for (const auto& match : matched) {
        const auto& ID = match.first;
        // Find corresponding detection
        const auto& detection = *std::find_if(objs.begin(), objs.end(), [&match](const det::Object& obj) {
            // Convert cv::Rect_<int> to cv::Rect_<float> for comparison
            cv::Rect_<float> rectFloat(static_cast<float>(match.second.x), static_cast<float>(match.second.y), static_cast<float>(match.second.width), static_cast<float>(match.second.height));
            return obj.rect == rectFloat;
        });
        tracks_[ID].Update(detection.rect);
        tracks_[ID].label = detection.label;  // Assign the label to the track
    }

    /*** Create new tracks for unmatched detections ***/
    for (const auto& det : unmatched_det) {
        Track tracker;
        // Find corresponding detection
        const auto& detection = *std::find_if(objs.begin(), objs.end(), [&det](const det::Object& obj) {
            // Convert cv::Rect_<int> to cv::Rect_<float> for comparison
            cv::Rect_<float> rectFloat(static_cast<float>(det.x), static_cast<float>(det.y), static_cast<float>(det.width), static_cast<float>(det.height));
            return obj.rect == rectFloat;
        });
        tracker.Init(detection.rect);
        tracker.label = detection.label;  // Assign the label to the track
        // Create new track and generate new ID
        tracks_[id++] = tracker;
    }

    /*** Delete lose tracked tracks ***/
    for (auto it = tracks_.begin(); it != tracks_.end();) {
        if (it->second.coast_cycles_ > kMaxCoastCycles) {
            it = tracks_.erase(it);
        } else {
            it++;
        }
    }
}


std::vector<det::Object> Tracker::GetTracks() {
    std::vector<det::Object> objects;
    for (const auto& track : tracks_) {
        det::Object obj;
        obj.rect = track.second.GetStateAsBbox();
        obj.label = track.second.label;  // Access the label member
        obj.id = track.first;
        objects.push_back(obj);
    }
    return objects;
}


