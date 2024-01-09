#include "temp_variables.hpp"

namespace RPE
{
    cv::Mat img1_l, img1_r, img2_l, img2_r;

    vector<cv::KeyPoint> kps1_l, kps1_r, kps2_l, kps2_r;
    vector<cv::KeyPoint> kps1_l_ori, kps1_r_ori, kps2_l_ori, kps2_r_ori;
    vector<cv::KeyPoint> kps1_l_stereo, kps1_r_stereo, kps2_l_stereo, kps2_r_stereo;
    vector<cv::KeyPoint> kps1_l_matched, kps2_l_matched;
    vector<cv::KeyPoint> kps1_l_refined, kps2_l_refined;

    vector<cv::Mat> desc1_l, desc1_r, desc2_l, desc2_r;

    vector<Vector3d> kps3d1, kps3d2;
    vector<Vector3d> kps3d1_stereo, kps3d2_stereo;

    Matrix3d R12_mono, R12_stereo;
    Vector3d t12_stereo;
}