#pragma once

#include <string>
#include <chrono>
#include <ros/ros.h>
#include <rpe/FeatureDetection.h>
#include <rpe/Matching.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <eigen3/Eigen/Dense>
#include <opengv/point_cloud/PointCloudAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/point_cloud/PointCloudSacProblem.hpp>
#include <glog/logging.h>
#include "Matcher.hpp"
#include "temp_variables.hpp"

using namespace std;
using namespace Eigen;
using namespace sensor_msgs;

namespace RPE
{
    class RPE
    {
    public:
        RPE(string settings_path, const ros::NodeHandle &nh_);

        bool estimate(const cv::Mat &img1_l_, const cv::Mat &img1_r_,
                      const cv::Mat &img2_l_, const cv::Mat &img2_r_,
                      Matrix3d &R12_, Vector3d &t12_);

        void extractKpsDepth(vector<cv::KeyPoint> &kps_l, vector<cv::KeyPoint> &kps_r,
                             vector<cv::Mat> &desc_l, vector<cv::Mat> &desc_r,
                             vector<Vector3d> &kps3d);

        void extractKpsDepth(const vector<int> &match, vector<cv::KeyPoint> &kps_l, vector<cv::KeyPoint> &kps_r,
                             vector<Vector3d> &kps3d);

        void matchByDescriptor(vector<cv::KeyPoint> &kps1, vector<cv::KeyPoint> &kps2,
                               vector<cv::Mat> &desc1, vector<cv::Mat> &desc2,
                               vector<Vector3d> &kps3d1, vector<Vector3d> &kps3d2);

        template <typename T>
        inline void rearrangeMatchedVec(const vector<int> &match, vector<T> &vec1, vector<T> &vec2);

        bool recoverPoseArun(const vector<Vector3d> &kps3d1_, const vector<Vector3d> &kps3d2_,
                             Matrix3d &R12, Vector3d &t12); // Borrowed from Kimera-VIO

        ros::NodeHandle nh;

        // Camera params
        double baseline;
        double fx;
        double fy;
        double cx;
        double cy;

        enum class FeatureDetectorType
        {
            GFTT,
            SIFT,
            HLOC
        } feature_detector_type;

        cv::Ptr<cv::FeatureDetector> feature_detector;
        bool enable_subpix_corner_refinement;
        ros::ServiceClient hloc_feature_detector;

        enum class DescriptorExtractorType
        {
            SIFT,
            ORB
        } descriptor_extractor_type;

        cv::Ptr<cv::DescriptorExtractor> descriptor_extractor;

        bool enable_hloc_matcher;
        unique_ptr<Matcher> matcher;
        ros::ServiceClient hloc_matcher;

        using SacProblem = opengv::sac_problems::point_cloud::PointCloudSacProblem;
        unique_ptr<opengv::sac::Ransac<SacProblem>> ransacer;
        int ransac_min_inliers;
        int ransac_verbosity_level;
    };
}