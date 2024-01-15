#include "Visualizer.hpp"

namespace RPE
{
    Visualizer::Visualizer(const ros::NodeHandle &nh_) : nh(nh_) {}

    void Visualizer::draw(DrawType draw_type)
    {
        switch (draw_type)
        {
        case DrawType::FEATURE_EXTRACTION:
            drawKps();
            break;
        case DrawType::STEREO_MATCHES:
            drawStereoMatches();
            break;
        case DrawType::DESCRIPTOR_MATCHES:
            drawDescriptorMatches();
            break;
        default:
            break;
        }
    }

    void Visualizer::drawKps()
    {
        cv::Mat img1_l_show = img1_l.clone();
        cv::Mat img1_r_show = img1_r.clone();
        cv::Mat img2_l_show = img2_l.clone();
        cv::Mat img2_r_show = img2_r.clone();

        cv::cvtColor(img1_l_show, img1_l_show, cv::COLOR_GRAY2BGR);
        cv::cvtColor(img1_r_show, img1_r_show, cv::COLOR_GRAY2BGR);
        cv::cvtColor(img2_l_show, img2_l_show, cv::COLOR_GRAY2BGR);
        cv::cvtColor(img2_r_show, img2_r_show, cv::COLOR_GRAY2BGR);

        for (size_t i = 0; i < kps1_l_ori.size(); i++)
            cv::circle(img1_l_show, kps1_l_ori[i].pt, 2, cv::Scalar(0, 255, 0), -1);
        for (size_t i = 0; i < kps1_r_ori.size(); i++)
            cv::circle(img1_r_show, kps1_r_ori[i].pt, 2, cv::Scalar(0, 255, 0), -1);
        for (size_t i = 0; i < kps2_l_ori.size(); i++)
            cv::circle(img2_l_show, kps2_l_ori[i].pt, 2, cv::Scalar(0, 255, 0), -1);
        for (size_t i = 0; i < kps2_r_ori.size(); i++)
            cv::circle(img2_r_show, kps2_r_ori[i].pt, 2, cv::Scalar(0, 255, 0), -1);

        cv::Mat img1_show, img2_show, img12_show;
        cv::vconcat(img1_l_show, img1_r_show, img1_show);
        cv::vconcat(img2_l_show, img2_r_show, img2_show);
        cv::hconcat(img1_show, img2_show, img12_show);

        cv::resize(img12_show, img12_show, cv::Size(), 0.66, 0.66);

        cv::putText(img12_show, to_string(kps1_l_ori.size()) + ":" + to_string(kps1_r_ori.size()) + ":" + to_string(kps2_l_ori.size()) + ":" + to_string(kps2_r_ori.size()),
                    cv::Point(3, 17), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);

        cv::imshow("Feature Extraction", img12_show);
        cv::waitKey(50);
    };

    void Visualizer::drawStereoMatches()
    {
        cv::Mat img1_l_show = img1_l.clone();
        cv::Mat img1_r_show = img1_r.clone();
        cv::Mat img2_l_show = img2_l.clone();
        cv::Mat img2_r_show = img2_r.clone();

        cv::cvtColor(img1_l_show, img1_l_show, cv::COLOR_GRAY2BGR);
        cv::cvtColor(img1_r_show, img1_r_show, cv::COLOR_GRAY2BGR);
        cv::cvtColor(img2_l_show, img2_l_show, cv::COLOR_GRAY2BGR);
        cv::cvtColor(img2_r_show, img2_r_show, cv::COLOR_GRAY2BGR);

        for (size_t i = 0; i < kps1_l_ori.size(); i++)
            cv::circle(img1_l_show, kps1_l_ori[i].pt, 2, cv::Scalar(0, 0, 255), -1);
        for (size_t i = 0; i < kps1_r_ori.size(); i++)
            cv::circle(img1_r_show, kps1_r_ori[i].pt, 2, cv::Scalar(0, 0, 255), -1);
        for (size_t i = 0; i < kps2_l_ori.size(); i++)
            cv::circle(img2_l_show, kps2_l_ori[i].pt, 2, cv::Scalar(0, 0, 255), -1);
        for (size_t i = 0; i < kps2_r_ori.size(); i++)
            cv::circle(img2_r_show, kps2_r_ori[i].pt, 2, cv::Scalar(0, 0, 255), -1);

        cv::Mat img1_show, img2_show;
        cv::vconcat(img1_l_show, img1_r_show, img1_show);
        cv::vconcat(img2_l_show, img2_r_show, img2_show);

        for (size_t i = 0; i < kps1_l_stereo.size(); i++)
        {
            const cv::Point &kp_l = kps1_l_stereo[i].pt;
            const cv::Point &kp_r_ = kps1_r_stereo[i].pt;
            const cv::Point kp_r(kp_r_.x, kp_r_.y + img1_l_show.rows);

            cv::line(img1_show, kp_l, kp_r, cv::Scalar(0, 155, 0), 1);
            cv::circle(img1_show, kp_l, 2, cv::Scalar(0, 255, 0), -1);
            cv::circle(img1_show, kp_r, 2, cv::Scalar(0, 255, 0), -1);
        }
        for (size_t i = 0; i < kps2_l_stereo.size(); i++)
        {
            const cv::Point &kp_l = kps2_l_stereo[i].pt;
            const cv::Point &kp_r_ = kps2_r_stereo[i].pt;
            const cv::Point kp_r(kp_r_.x, kp_r_.y + img2_l_show.rows);

            double disparity = kp_l.x - kp_r_.x;
            if (disparity > 13)
            {
                cv::line(img2_show, kp_l, kp_r, cv::Scalar(0, 155, 0), 1);
                cv::circle(img2_show, kp_l, 2, cv::Scalar(0, 255, 0), -1);
                cv::circle(img2_show, kp_r, 2, cv::Scalar(0, 255, 0), -1);
            }
        }

        cv::Mat img12_show;
        cv::hconcat(img1_show, img2_show, img12_show);

        cv::resize(img12_show, img12_show, cv::Size(), 0.66, 0.66);

        cv::putText(img12_show, to_string(kps1_l_stereo.size()) + "|" + to_string(kps1_l_ori.size()) + ":" + to_string(kps1_r_ori.size()) + "  " + to_string(kps2_l_stereo.size()) + "|" + to_string(kps2_l_ori.size()) + ":" + to_string(kps2_r_ori.size()),
                    cv::Point(3, 17), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);

        cv::imshow("Stereo Matches", img12_show);
        cv::waitKey(50);
    }

    void Visualizer::drawDescriptorMatches()
    {
        cv::Mat img1_l_show = img1_l.clone();
        cv::Mat img2_l_show = img2_l.clone();

        cv::cvtColor(img1_l_show, img1_l_show, cv::COLOR_GRAY2BGR);
        cv::cvtColor(img2_l_show, img2_l_show, cv::COLOR_GRAY2BGR);

        for (size_t i = 0; i < kps1_l_stereo.size(); i++)
            cv::circle(img1_l_show, kps1_l_stereo[i].pt, 2, cv::Scalar(0, 0, 255), -1);
        for (size_t i = 0; i < kps2_l_stereo.size(); i++)
            cv::circle(img2_l_show, kps2_l_stereo[i].pt, 2, cv::Scalar(0, 0, 255), -1);

        cv::Mat img12_show;
        cv::vconcat(img1_l_show, img2_l_show, img12_show);

        for (size_t i = 0; i < kps1_l_matched.size(); i++)
        {
            const cv::Point &kp_1 = kps1_l_matched[i].pt;
            const cv::Point &kp_2_ = kps2_l_matched[i].pt;
            const cv::Point kp_2(kp_2_.x, kp_2_.y + img1_l_show.rows);

            cv::line(img12_show, kp_1, kp_2, cv::Scalar(0, 155, 0), 1);
            cv::circle(img12_show, kp_1, 2, cv::Scalar(0, 255, 0), -1);
            cv::circle(img12_show, kp_2, 2, cv::Scalar(0, 255, 0), -1);
        }

        cv::resize(img12_show, img12_show, cv::Size(), 0.66, 0.66);

        cv::putText(img12_show, to_string(kps1_l_matched.size()) + "|" + to_string(kps1_l_stereo.size()) + ":" + to_string(kps2_l_stereo.size()),
                    cv::Point(3, 17), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);

        cv::Mat img12_refined_show;
        cv::vconcat(img1_l_show, img2_l_show, img12_refined_show);

        for (size_t i = 0; i < kps1_l_refined.size(); i++)
        {
            const cv::Point &kp_1 = kps1_l_refined[i].pt;
            const cv::Point &kp_2_ = kps2_l_refined[i].pt;
            const cv::Point kp_2(kp_2_.x, kp_2_.y + img1_l_show.rows);

            cv::line(img12_refined_show, kp_1, kp_2, cv::Scalar(0, 155, 0), 1);
            cv::circle(img12_refined_show, kp_1, 2, cv::Scalar(0, 255, 0), -1);
            cv::circle(img12_refined_show, kp_2, 2, cv::Scalar(0, 255, 0), -1);
        }

        cv::resize(img12_refined_show, img12_refined_show, cv::Size(), 0.66, 0.66);

        cv::putText(img12_refined_show, to_string(kps2_l_refined.size()) + "|" + to_string(kps1_l_stereo.size()) + ":" + to_string(kps2_l_stereo.size()),
                    cv::Point(3, 17), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);

        cv::Mat img12_show_final;
        cv::hconcat(img12_show, img12_refined_show, img12_show_final);

        cv::imshow("Descriptor Matches", img12_show_final);
        cv::waitKey(50);
    }

    void Visualizer::pubPose(nav_msgs::Odometry &pose_msg, const string &topic)
    {
        pose_msg.header.frame_id = "map";

        if (pose_publisher_set.find(topic) == pose_publisher_set.end())
        {
            pose_publisher_set[topic] = nh.advertise<nav_msgs::Odometry>(topic, 10);
            std::this_thread::sleep_for(std::chrono::seconds(2));
            pose_publisher_set[topic].publish(pose_msg);
        }
        else
            pose_publisher_set[topic].publish(pose_msg);
    }

    void Visualizer::pubPose(const Matrix3d &R, const Vector3d &t, const string &topic)
    {
        nav_msgs::Odometry pose_msg;

        const Quaterniond q(R);
        pose_msg.pose.pose.orientation.w = q.w();
        pose_msg.pose.pose.orientation.x = q.x();
        pose_msg.pose.pose.orientation.y = q.y();
        pose_msg.pose.pose.orientation.z = q.z();
        pose_msg.pose.pose.position.x = t.x();
        pose_msg.pose.pose.position.y = t.y();
        pose_msg.pose.pose.position.z = t.z();

        pubPose(pose_msg, topic);
    }

    void Visualizer::pubKps3d(const vector<Vector3d> &kps3d, const Matrix3d &R, const Vector3d &t, const string &topic)
    {
        pcl::PointCloud<pcl::PointXYZ> pc;
        for (size_t i = 0; i < kps3d.size(); i++)
        {
            Vector3d pt = kps3d[i];
            pt = R * pt + t;
            pc.emplace_back(pt.x(), pt.y(), pt.z());
        }

        sensor_msgs::PointCloud2 kps3d_msg;
        pcl::toROSMsg(pc, kps3d_msg);
        kps3d_msg.header.frame_id = "map";

        if (kps3d_publisher_set.find(topic) == kps3d_publisher_set.end())
        {
            kps3d_publisher_set[topic] = nh.advertise<sensor_msgs::PointCloud2>(topic, 10);
            std::this_thread::sleep_for(std::chrono::seconds(2));
            kps3d_publisher_set[topic].publish(kps3d_msg);
        }
        else
            kps3d_publisher_set[topic].publish(kps3d_msg);
    }

    void Visualizer::pubKps3d(const vector<Vector3d> &kps3d, const nav_msgs::Odometry &pose, const string &topic)
    {
        const Matrix3d R(Quaterniond(pose.pose.pose.orientation.w,
                                     pose.pose.pose.orientation.x,
                                     pose.pose.pose.orientation.y,
                                     pose.pose.pose.orientation.z));
        const Vector3d t(pose.pose.pose.position.x,
                         pose.pose.pose.position.y,
                         pose.pose.pose.position.z);

        pubKps3d(kps3d, R, t, topic);
    }
}