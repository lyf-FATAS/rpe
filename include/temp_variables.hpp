#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace Eigen;

namespace RPE
{
    extern cv::Mat img1_l, img1_r, img2_l, img2_r;

    extern vector<cv::KeyPoint> kps1_l_ori, kps1_r_ori, kps2_l_ori, kps2_r_ori;
    extern vector<cv::KeyPoint> kps1_l_stereo, kps1_r_stereo, kps2_l_stereo, kps2_r_stereo;
    extern vector<cv::KeyPoint> kps1_l_matched, kps2_l_matched;
    extern vector<cv::KeyPoint> kps1_l_refined, kps2_l_refined;
    extern vector<cv::KeyPoint> kps1_l_inliers_mono, kps2_l_inliers_mono;
    extern vector<cv::KeyPoint> kps1_l_inliers_stereo, kps2_l_inliers_stereo;

    extern vector<Vector3d> kps3d1_stereo, kps3d2_stereo;
    extern vector<Vector3d> kps3d1_matched, kps3d2_matched;

    // "gv" stands for "geometric verification"
    extern Matrix3d R12_gv_mono;
    extern Matrix3d R12_gv;
    extern Vector3d t12_gv;
    extern Matrix3d R12_refined;
    extern Vector3d t12_refined;
}