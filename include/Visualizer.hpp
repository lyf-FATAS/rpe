#pragma once

#include <memory>
#include <thread>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <eigen3/Eigen/Dense>
#include "temp_variables.hpp"

using namespace Eigen;

namespace RPE
{
    class Visualizer
    {
    public:
        Visualizer(const ros::NodeHandle &nh_);

        enum DrawType
        {
            FEATURE_EXTRACTION,
            STEREO_MATCHES,
            DESCRIPTOR_MATCHES
        };

        void draw(DrawType draw_type);

        void drawKps();

        void drawStereoMatches();

        void drawDescriptorMatches();

        void pubPose(nav_msgs::Odometry &pose, const string &topic);

        void pubPose(const Matrix3d &R, const Vector3d &t, const string &topic);

        void pubKps3d(const vector<Vector3d> &kps3d, const Matrix3d &R, const Vector3d &t, const string &topic);

        void pubKps3d(const vector<Vector3d> &kps3d, const nav_msgs::Odometry &pose, const string &topic);

        ros::NodeHandle nh;
        map<string, ros::Publisher> pose_publisher_set;
        map<string, ros::Publisher> kps3d_publisher_set;
    };
}