#include "temp_variables.hpp"

namespace RPE
{
    cv::Mat img1_l, img1_r, img2_l, img2_r;

    vector<cv::KeyPoint> kps1_l_ori, kps1_r_ori, kps2_l_ori, kps2_r_ori;
    vector<cv::KeyPoint> kps1_l_stereo, kps1_r_stereo, kps2_l_stereo, kps2_r_stereo;
    vector<cv::KeyPoint> kps1_l_matched, kps2_l_matched;
    vector<cv::KeyPoint> kps1_l_inliers_mono, kps2_l_inliers_mono;
    vector<cv::KeyPoint> kps1_l_outliers_stereo, kps2_l_outliers_stereo;
    vector<cv::KeyPoint> kps1_l_outliers_gnc, kps2_l_outliers_gnc;
    vector<cv::KeyPoint> kps1_l_refined, kps2_l_refined;
    vector<cv::KeyPoint> kps1_l_outliers_refined_gnc, kps2_l_outliers_refined_gnc;

    vector<Vector3d> kps3d1_stereo, kps3d2_stereo;
    vector<Vector3d> kps3d1_matched, kps3d2_matched;
    vector<Vector3d> kps3d1_refined, kps3d2_refined;

    Matrix3d R12_ransac_mono;
    Matrix3d R12_ransac;
    Vector3d t12_ransac;
    Matrix3d R12_gnc;
    Vector3d t12_gnc;
    Matrix3d R12_refined;
    Vector3d t12_refined;

    vector<Matrix3d> R12_debug;
    vector<Vector3d> t12_debug;
}