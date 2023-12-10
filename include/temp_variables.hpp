#pragma once

#include <vector>
#include <opencv2/core/core.hpp>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

namespace RPE
{
    extern cv::Mat img1_l, img1_r, img2_l, img2_r;

    extern vector<cv::KeyPoint> kps1_l, kps1_r, kps2_l, kps2_r;
    extern vector<cv::KeyPoint> kps1_l_ori, kps1_r_ori, kps2_l_ori, kps2_r_ori;
    extern vector<cv::KeyPoint> kps1_l_stereo, kps1_r_stereo, kps2_l_stereo, kps2_r_stereo;
    extern vector<cv::KeyPoint> kps1_l_desc, kps2_l_desc;

    extern vector<cv::Mat> desc1_l, desc1_r, desc2_l, desc2_r;

    extern vector<Vector3d> kps3d1, kps3d2;
    extern vector<Vector3d> kps3d1_debug, kps3d2_debug;

    extern Matrix3d R12;
    extern Vector3d t12;
}