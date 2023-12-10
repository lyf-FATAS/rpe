#pragma once

#include <string>
#include <opencv2/core/core.hpp>
#include <glog/logging.h>
#include "gurobi_c++.h"
#include "temp_variables.hpp"

using namespace std;

namespace RPE
{
    class Matcher
    {
    public:
        Matcher(string settings_path);

        void matchStereo(const vector<cv::KeyPoint> &kps_l, const vector<cv::KeyPoint> &kps_r,
                         const vector<cv::Mat> &desc_l, const vector<cv::Mat> &desc_r,
                         vector<int> &match);

        void matchByDesc(const vector<cv::KeyPoint> &kps1, const vector<cv::KeyPoint> &kps2,
                         const vector<cv::Mat> &desc1, const vector<cv::Mat> &desc2,
                         vector<int> &match);

        void matchByDescCV(const vector<cv::KeyPoint> &kps1, const vector<cv::KeyPoint> &kps2,
                           const vector<cv::Mat> &desc1, const vector<cv::Mat> &desc2,
                           vector<int> &match);

        void matchByDescGNN(const vector<cv::KeyPoint> &kps1, const vector<cv::KeyPoint> &kps2,
                            const vector<cv::Mat> &desc1, const vector<cv::Mat> &desc2,
                            vector<int> &match);

        double fx;

        enum MatcherType
        {
            CV_NN, // OpenCV's Nearest Neighbour
            GNN    // Global Nearest Neighbour
        } matcher_type;

        cv::NormTypes norm_type;

        cv::Ptr<cv::BFMatcher> matcher_cv;
        bool enable_ratio_test;
        double lowe_ratio;

        double stereo_desc_dist_th, desc_dist_th;
        bool enable_consistency;
        double weight_desc_vs_consis, weight_len_vs_ang;
        double diff_len_th, diff_ang_th;
        int GRB_log_to_console;
        int GRB_mip_focus;
        double GRB_heuristics;
        int GRB_presolve;
        int GRB_presparsify;
        double GRB_time_limit;
    };
}