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
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/point_cloud/PointCloudAdapter.hpp>
#include <opengv/point_cloud/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>
#include <opengv/sac_problems/point_cloud/PointCloudSacProblem.hpp>
#include <glog/logging.h>
#include "Matcher.hpp"
#include "Solver.hpp"
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
                      Matrix3d &R12, Vector3d &t12);

        void extractKpsDepth(vector<cv::KeyPoint> &kps_l, vector<cv::KeyPoint> &kps_r,
                             vector<cv::Mat> &desc_l, vector<cv::Mat> &desc_r,
                             vector<Vector3d> &kps3d);

        void extractKpsDepth(vector<int> &match, vector<cv::KeyPoint> &kps_l, vector<cv::KeyPoint> &kps_r,
                             vector<Vector3d> &kps3d);

        void matchByDescriptor(vector<cv::KeyPoint> &kps1, vector<cv::KeyPoint> &kps2,
                               vector<cv::Mat> &desc1, vector<cv::Mat> &desc2,
                               vector<Vector3d> &kps3d1, vector<Vector3d> &kps3d2);

        template <typename T>
        inline void rearrangeMatchedVec(const vector<int> &match12, vector<T> &vec1, vector<T> &vec2);

        bool geometricVerificationNister(const vector<cv::KeyPoint> &kps1, const vector<cv::KeyPoint> &kps2, Matrix3d &R12_mono, vector<int> &inliers_mono); // Borrowed from Kimera-VIO

        bool recoverPose(const vector<Vector3d> &kps3d1, const vector<Vector3d> &kps3d2, const Matrix3d &R12_mono, const vector<int> &inliers_mono,
                         Matrix3d &R12_stereo, Vector3d &t12_stereo, vector<int> &inliers_stereo); // Borrowed from Kimera-VIO

        void alternateOpt(const vector<Vector3d> &kps3d1, const vector<Vector3d> &kps3d2,
                          vector<int> &match12, Matrix3d &R12, Vector3d &t12);

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

        double far_pt_thr;

        int ransac_verbosity_level;
        int ransac_min_inliers;

        using SacProblemMono = opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem;
        unique_ptr<opengv::sac::Ransac<SacProblemMono>> mono_ransacer;

        using SacProblemStereo = opengv::sac_problems::point_cloud::PointCloudSacProblem;
        unique_ptr<opengv::sac::Ransac<SacProblemStereo>> stereo_ransacer;

        bool enable_nlopt;

        bool enable_outdoor_mode;

        unique_ptr<ASolver> A_solver;
        unique_ptr<TSolver> T_solver;
    };
}