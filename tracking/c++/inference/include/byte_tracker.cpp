#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include "matching.hpp"
#include "kalman_filter_xyah.hpp"
#include "basetrack.hpp"

class STrack : public BaseTrack {
private:
    static KalmanFilterXYAH shared_kalman;

    cv::Mat _tlwh;
    KalmanFilterXYAH* kalman_filter;
    cv::Mat mean, covariance;
    bool is_activated;

public:
    float score;
    int tracklet_len;
    int cls;
    int idx;

    STrack(const std::vector<float>& tlwh, float score_, int cls_) : 
            _tlwh(4, 1, CV_32FC1), kalman_filter(nullptr), mean(8, 1, CV_32FC1), covariance(8, 8, CV_32FC1),
            is_activated(false), score(score_), tracklet_len(0), cls(cls_), idx(0)
    {
        // Copy the values from tlwh
        for (int i = 0; i < 4; i++)
        {
            _tlwh.at<float>(i) = tlwh[i];
        }
        idx = static_cast<int>(_tlwh.at<float>(3));
    }

    ~STrack() override {}

    void predict() override {
        cv::Mat mean_state = mean.clone();
        if (state != TrackState::Tracked) {
            mean_state.at<float>(7) = 0.f;
        }
        shared_kalman.predict(mean_state, covariance, mean, covariance);
    }

    static void multi_predict(std::vector<STrack*>& stracks) {
        if (stracks.empty()) {
            return;
        }

        std::vector<cv::Mat> multi_mean(stracks.size());
        std::vector<cv::Mat> multi_covariance(stracks.size());

        for (size_t i = 0; i < stracks.size(); i++) {
            multi_mean[i] = stracks[i]->mean.clone();
            multi_covariance[i] = stracks[i]->covariance.clone();
            if (stracks[i]->state != TrackState::Tracked) {
                multi_mean[i].at<float>(7) = 0.f;
            }
        }

        shared_kalman.multi_predict(multi_mean, multi_covariance, multi_mean, multi_covariance);

        for (size_t i = 0; i < stracks.size(); i++) {
            stracks[i]->mean = multi_mean[i];
            stracks[i]->covariance = multi_covariance[i];
        }
    }

    static void multi_gmc(std::vector<STrack*>& stracks, const cv::Mat& H = cv::Mat::eye(2, 3, CV_32FC1)) {
        if (stracks.empty()) {
            return;
        }

        cv::Mat R = H(cv::Rect(0, 0, 2, 2)).clone();
        cv::Mat t = H.col(2).clone();
        cv::Mat R8x8 = cv::Mat::zeros(8, 8, CV_32FC1);

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                R8x8.at<float>(i, j) = R.at<float>(i / 2, j / 2);
            }
        }

        for (size_t i = 0; i < stracks.size(); i++) {
	            stracks[i]->mean.at<float>(0) += t.at<float>(0);
            stracks[i]->mean.at<float>(1) += t.at<float>(1);
            stracks[i]->mean.at<float>(2) = R8x8.at<float>(0, 0) * stracks[i]->mean.at<float>(2) +
                R8x8.at<float>(0, 1) * stracks[i]->mean.at<float>(3);
            stracks[i]->mean.at<float>(3) = R8x8.at<float>(1, 0) * stracks[i]->mean.at<float>(2) +
                R8x8.at<float>(1, 1) * stracks[i]->mean.at<float>(3);
            stracks[i]->mean.at<float>(4) = R8x8.at<float>(2, 2) * stracks[i]->mean.at<float>(4) +
                R8x8.at<float>(2, 3) * stracks[i]->mean.at<float>(5);
            stracks[i]->mean.at<float>(5) = R8x8.at<float>(3, 2) * stracks[i]->mean.at<float>(4) +
                R8x8.at<float>(3, 3) * stracks[i]->mean.at<float>(5);
            stracks[i]->mean.at<float>(6) = stracks[i]->mean.at<float>(6);
            stracks[i]->mean.at<float>(7) = stracks[i]->mean.at<float>(7);
            shared_kalman.update(stracks[i]->mean, stracks[i]->covariance, stracks[i]->mean, stracks[i]->covariance);
        }
    }

    void activate(const cv::Mat& img, int new_id) override {
        kalman_filter = &shared_kalman;
        mean.at<float>(0) = _tlwh.at<float>(0);
        mean.at<float>(1) = _tlwh.at<float>(1);
        mean.at<float>(2) = 0;
        mean.at<float>(3) = 0;
        mean.at<float>(4) = _tlwh.at<float>(2) / _tlwh.at<float>(3);
        mean.at<float>(5) = 0;
        mean.at<float>(6) = _tlwh.at<float>(3);
        mean.at<float>(7) = 0;

        covariance.at<float>(0, 0) = 4 * pow(_tlwh.at<float>(3), 2);
        covariance.at<float>(1, 1) = 4 * pow(_tlwh.at<float>(3), 2);
        covariance.at<float>(2, 2) = 4 * pow(_tlwh.at<float>(3), 2);
        covariance.at<float>(3, 3) = 4 * pow(_tlwh.at<float>(3), 2);
        covariance.at<float>(4, 4) = 0.25;
        covariance.at<float>(5, 5) = 0.25;
        covariance.at<float>(6, 6) = 16 * pow(_tlwh.at<float>(3), 2);
        covariance.at<float>(7, 7) = 1;

        idx = new_id;
        is_activated = true;
        tracklet_len = 1;
        update(img, _tlwh);
    }

    void update(const cv::Mat& img, const std::vector<float>& tlwh) override {
        _tlwh = cv::Mat(4, 1, CV_32FC1);
        for (int i
	
	        _tlwh.at<float>(0) = tlwh[0];
        _tlwh.at<float>(1) = tlwh[1];
        _tlwh.at<float>(2) = tlwh[2];
        _tlwh.at<float>(3) = tlwh[3];

        if (!is_activated) {
            return;
        }

        tracklet_len++;
        prediction();
        _tlwh.at<float>(3) *= 1.05;
        update_mean_covariance();
        if (time_since_update > 0) {
            hit_streak = 0;
            state = TrackState::Tentative;
        }
        time_since_update++;

        if (state == TrackState::Tentative && hit_streak >= min_hits) {
            state = TrackState::Confirmed;
        }
    }

    void mark_missed() override {
        if (time_since_update > 0) {
            hit_streak = 0;
            state = TrackState::Deleted;
        }
        time_since_update++;
    }

    bool is_confirmed() const override {
        return state == TrackState::Confirmed;
    }

    bool is_deleted() const override {
        return state == TrackState::Deleted;
    }

    bool is_tentative() const override {
        return state == TrackState::Tentative;
    }

    bool is_tracked() const override {
        return state == TrackState::Tracked || state == TrackState::Tentative || state == TrackState::Confirmed;
    }

    cv::Rect_<float> to_tlwh() const override {
        cv::Rect_<float> ret;
        ret.width = mean.at<float>(6) * mean.at<float>(4);
        ret.height = mean.at<float>(6);
        ret.x = mean.at<float>(0) - ret.width / 2;
        ret.y = mean.at<float>(1) - ret.height / 2;
        return ret;
    }

    cv::Point2f to_tlbr() const override {
        return cv::Point2f(mean.at<float>(0) + mean.at<float>(4) * mean.at<float>(6) / 2,
            mean.at<float>(1) + mean.at<float>(6) / 2);
    }

    float tracking_score() const override {
        return state == TrackState::Deleted ? 0.f : 1.f;
    }
};


