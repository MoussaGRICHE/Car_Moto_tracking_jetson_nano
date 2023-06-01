#include "linear_assignment.h"
#include "hungarianoper.h"
#include <map>

linear_assignment *linear_assignment::instance = NULL;
linear_assignment::linear_assignment()
{
}

linear_assignment *linear_assignment::getInstance()
{
    if(instance == NULL) instance = new linear_assignment();
    return instance;
}

TRACHER_MATCHD
linear_assignment::matching_cascade(
        tracker *distance_metric,
        tracker::GATED_METRIC_FUNC distance_metric_func,
        float max_distance,
        int cascade_depth,
        std::vector<Track> &tracks,
        const DETECTIONS &detections,
        std::vector<int>& track_indices,
        std::vector<int> detection_indices)
{
    TRACHER_MATCHD res;

    for(size_t i = 0; i < detections.size(); i++) {
        detection_indices.push_back(int(i));
    }

    std::vector<int> unmatched_detections;
    unmatched_detections.assign(detection_indices.begin(), detection_indices.end());


    res.matches.clear();
    std::vector<int> track_indices_l;

    std::map<int, int> matches_trackid;
    for(int level = 0; level < cascade_depth; level++) {

        if(unmatched_detections.size() == 0) break; //No detections left;

        track_indices_l.clear();
        for(int k:track_indices) {
            if(tracks[k].time_since_update == 1+level)
                track_indices_l.push_back(k);
        }

        if(track_indices_l.size() == 0) continue; //Nothing to match at this level.

        //std::cout << "distance_metric " << distance_metric << std::endl;

        //std::cout << "distance_metric_func " << distance_metric_func<< std::endl;

        //std::cout << "max_distance" << max_distance << std::endl;


        //for (int i = 0; i < track_indices_l.size(); i++) {
            //std::cout << "track_indices_l[i]" <<  track_indices_l[i] << std::endl;;
        //}

        TRACHER_MATCHD tmp = min_cost_matching(
                    distance_metric, distance_metric_func,
                    max_distance, tracks, detections, track_indices_l,
                    unmatched_detections);
        
        //std::cout << "min_cost_matching OK " << std::endl;

        unmatched_detections.assign(tmp.unmatched_detections.begin(), tmp.unmatched_detections.end());

        //std::cout << "unmatched_detections.assign OK " << std::endl;

        for(size_t i = 0; i < tmp.matches.size(); i++) {
            MATCH_DATA pa = tmp.matches[i];
            res.matches.push_back(pa);
            matches_trackid.insert(pa);
        }
    }

    //std::cout << "res.matches OK " << std::endl;

    res.unmatched_detections.assign(unmatched_detections.begin(), unmatched_detections.end());
    for(size_t i = 0; i < track_indices.size(); i++) {
        int tid = track_indices[i];
        if(matches_trackid.find(tid) == matches_trackid.end())
            res.unmatched_tracks.push_back(tid);
    }

    //std::cout << "res.unmatched OK " << std::endl;

    return res;
}

TRACHER_MATCHD 
linear_assignment::min_cost_matching(tracker *distance_metric,
                                                  tracker::GATED_METRIC_FUNC distance_metric_func,
                                                  float max_distance,
                                                  std::vector<Track> &tracks,
                                                  const DETECTIONS &detections,
                                                  std::vector<int> &track_indices,
                                                  std::vector<int> &detection_indices)
{

    // Print input values
    std::cout <<  "" << std::endl;
    std::cout << "distance_metric: " << distance_metric << std::endl;
    std::cout << "distance_metric_func: " << distance_metric_func << std::endl;
    std::cout << "max_distance: " << max_distance << std::endl;
    std::cout << "tracks size: " << tracks.size() << std::endl;
    std::cout << "detections size: " << detections.size() << std::endl;
    std::cout << "track_indices size: " << track_indices.size() << std::endl;
    std::cout << "detection_indices size: " << detection_indices.size() << std::endl;
    std::cout <<  "" << std::endl;

    TRACHER_MATCHD res;

    if ((detection_indices.size() == 0) || (track_indices.size() == 0)) {
        res.matches.clear();
        res.unmatched_tracks.assign(track_indices.begin(), track_indices.end());
        res.unmatched_detections.assign(detection_indices.begin(), detection_indices.end());
        return res;
    }

    DYNAMICM cost_matrix = (distance_metric->*(distance_metric_func))(tracks, detections, track_indices, detection_indices);
    
    std::cout << "matrix with distance_metric_func" << std::endl;
    std::cout << "Rows: " << cost_matrix.rows() << ", Columns: " << cost_matrix.cols() << std::endl;

    for (int i = 0; i < cost_matrix.rows(); i++) {
        for (int j = 0; j < cost_matrix.cols(); j++) {
            float tmp = cost_matrix(i, j);
            if (tmp > max_distance) {
                cost_matrix(i, j) = max_distance + 1e-5;
            }
        }
    }

    //std::cout << "" << std::endl;
    //std::cout << "cost_matrix 2" << cost_matrix << std::endl;

    Eigen::Matrix<float, -1, 2, Eigen::RowMajor> indices = HungarianOper::Solve(cost_matrix);

    std::cout << "matrix with HungarianOper" << std::endl;
    std::cout << "Rows: " << cost_matrix.rows() << ", Columns: " << cost_matrix.cols() << std::endl;

    res.matches.clear();
    res.unmatched_tracks.clear();
    res.unmatched_detections.clear();

    for (size_t col = 0; col < detection_indices.size(); col++) {
        bool flag = false;
        for (int i = 0; i < indices.rows(); i++) {
            if (indices(i, 1) == col) {
                flag = true;
                break;
            }
        }
        if (flag == false) {
            res.unmatched_detections.push_back(detection_indices[col]);
        }
    }
    //std::cout << "Number of unmatched detections: " << res.unmatched_detections.size() << std::endl;

    for (size_t row = 0; row < track_indices.size(); row++) {
        bool flag = false;
        for (int i = 0; i < indices.rows(); i++) {
            if (indices(i, 0) == row) {
                flag = true;
                break;
            }
        }
        if (flag == false) {
            res.unmatched_tracks.push_back(track_indices[row]);
        }
    }
    //std::cout << "Number of unmatched tracks: " << res.unmatched_tracks.size() << std::endl;

    for (int i = 0; i < indices.rows(); i++) {
        int row = indices(i, 0);
        int col = indices(i, 1);

        int track_idx = track_indices[row];
        int detection_idx = detection_indices[col];

        if (cost_matrix(row, col) > max_distance) {
            res.unmatched_tracks.push_back(track_idx);
            res.unmatched_detections.push_back(detection_idx);
        } else {
            res.matches.push_back(std::make_pair(track_idx, detection_idx));
        }
    }

    // Print statements to help identify where the function encounters issues
    //std::cout << "Number of matches: " << res.matches.size() << std::endl;
    
    

    return res;
}

DYNAMICM
linear_assignment::gate_cost_matrix(
        KalmanFilter *kf,
        DYNAMICM &cost_matrix,
        std::vector<Track> &tracks,
        const DETECTIONS &detections,
        const std::vector<int> &track_indices,
        const std::vector<int> &detection_indices,
        float gated_cost, bool only_position)
{
    int gating_dim = (only_position == true?2:4);
    double gating_threshold = KalmanFilter::chi2inv95[gating_dim];
    std::vector<DETECTBOX> measurements;
    for(int i:detection_indices) {
        DETECTION_ROW t = detections[i];
        measurements.push_back(t.to_xyah());
    }
    for(size_t i  = 0; i < track_indices.size(); i++) {
        Track& track = tracks[track_indices[i]];
        Eigen::Matrix<float, 1, -1> gating_distance = kf->gating_distance(
                    track.mean, track.covariance, measurements, only_position);
        for (int j = 0; j < gating_distance.cols(); j++) {
            if (gating_distance(0, j) > gating_threshold)  cost_matrix(i, j) = gated_cost;
        }
    }
    return cost_matrix;
}

