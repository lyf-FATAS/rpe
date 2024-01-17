#include <termios.h>
#include <unistd.h>
#include <string>
#include <thread>
#include <chrono>
#include <mutex>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <nav_msgs/Odometry.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <glog/logging.h>

#include "RPE.hpp"
#include "Visualizer.hpp"
#include "temp_variables.hpp"

using namespace std;
using namespace sensor_msgs;
using namespace message_filters;

enum FsmState
{
    WAITING_FOR_IMG,
    INIT,
    SOLVING_BY_TRIGGER,
    CONTINUOUS_SOLVING
} state;
bool trigger = false;

mutex img_mutex, odom_mutex;
cv_bridge::CvImageConstPtr img1_l_ptr, img1_r_ptr;
cv_bridge::CvImageConstPtr img2_l_ptr, img2_r_ptr;
nav_msgs::Odometry odom1, odom2;
bool recv_odom = false;

unique_ptr<RPE::Visualizer> visualizer;

// https://stackoverflow.com/questions/421860/capture-characters-from-standard-input-without-waiting-for-enter-to-be-pressed
char getch()
{
    char buf = 0;
    struct termios old = {0};
    if (tcgetattr(0, &old) < 0)
        perror("tcsetattr()");
    old.c_lflag &= ~ICANON;
    old.c_lflag &= ~ECHO;
    old.c_cc[VMIN] = 1;
    old.c_cc[VTIME] = 0;
    if (tcsetattr(0, TCSANOW, &old) < 0)
        perror("tcsetattr ICANON");
    if (read(0, &buf, 1) < 0)
        perror("read()");
    old.c_lflag |= ICANON;
    old.c_lflag |= ECHO;
    if (tcsetattr(0, TCSADRAIN, &old) < 0)
        perror("tcsetattr ~ICANON");
    return (buf);
}

void keyboardInputThread()
{
    while (ros::ok())
    {
        char ch = getch();
        switch (ch)
        {
        case ' ':
        {
            // Capture current img as img1
            {
                lock_guard<mutex> lock(img_mutex);
                img1_r_ptr = img2_r_ptr;
                img1_l_ptr = img2_l_ptr;
            }

            if (state == FsmState::INIT)
            {
                state = FsmState::SOLVING_BY_TRIGGER;
                LOG(INFO) << "\033[42mINIT\033[0m --> \033[42mSOLVING_BY_TRIGGER\033[0m";
                cv::destroyWindow("img_l");
            }

            if (recv_odom)
            {
                lock_guard<mutex> lock(odom_mutex);
                odom1 = odom2;
                visualizer->pubPose(odom1, "odom1");
            }
            break;
        }
        case 'c': // Switch to CONTINUOUS_SOLVING mode
        {
            if (state == FsmState::SOLVING_BY_TRIGGER)
            {
                state = FsmState::CONTINUOUS_SOLVING;
                LOG(INFO) << "\033[42mSOLVING_BY_TRIGGER\033[0m --> \033[42mCONTINUOUS_SOLVING\033[0m";
                cv::destroyWindow("img1_l and img2_l");
            }
            break;
        }
        case 't': // Switch to SOLVING_BY_TRIGGER mode
        {
            if (state == FsmState::CONTINUOUS_SOLVING)
            {
                state = FsmState::SOLVING_BY_TRIGGER;
                LOG(INFO) << "\033[42mCONTINUOUS_SOLVING\033[0m --> \033[42mSOLVING_BY_TRIGGER\033[0m";
                trigger = false;
            }
            else if (state == FsmState::SOLVING_BY_TRIGGER)
                trigger = true;
            break;
        }
        default:
            break;
        }
    }
}

void imgCallback(const ImageConstPtr &img_l_msg, const ImageConstPtr &img_r_msg)
{
    try
    {
        lock_guard<mutex> lock(img_mutex);
        img2_l_ptr = cv_bridge::toCvCopy(img_l_msg, image_encodings::MONO8);
        img2_r_ptr = cv_bridge::toCvCopy(img_r_msg, image_encodings::MONO8);
    }
    catch (const cv_bridge::Exception &e)
    {
        LOG(ERROR) << "cv_bridge exception: " << string(e.what());
        return;
    }

    if (state == FsmState::WAITING_FOR_IMG)
    {
        state = FsmState::INIT;
        LOG(INFO) << "\033[42mWAITING_FOR_IMG\033[0m --> \033[42mINIT\033[0m";

        thread keyboard_input_thread(keyboardInputThread);
        keyboard_input_thread.detach();
    }
}

void odomCallback(const nav_msgs::OdometryConstPtr &odom_msg)
{
    lock_guard<mutex> lock(odom_mutex);
    odom2 = *odom_msg;
    recv_odom = true;
}

int main(int argc, char **argv)
{
    google::InitGoogleLogging(argv[0]);
    FLAGS_colorlogtostderr = true;

    ros::init(argc, argv, "online_rpe_node");
    ros::NodeHandle nh("~");

    string settings_path = argv[1];
    cv::FileStorage settings(settings_path, cv::FileStorage::READ);

    int glog_severity_level = settings["glog_severity_level"];
    switch (glog_severity_level)
    {
    case 0:
        FLAGS_stderrthreshold = google::INFO;
        break;
    case 1:
        FLAGS_stderrthreshold = google::WARNING;
        break;
    case 2:
        FLAGS_stderrthreshold = google::ERROR;
        break;
    case 3:
        FLAGS_stderrthreshold = google::FATAL;
        break;
    default:
        break;
    }

    RPE::RPE estimator(settings_path, nh);

    visualizer = make_unique<RPE::Visualizer>(nh);
    RPE::Visualizer::DrawType draw_type;
    bool enable_ransac, enable_gnc, enable_alternate_opt;
    settings["Visualizer.draw_type"] >> draw_type;
    settings["enable_ransac"] >> enable_ransac;
    settings["enable_gnc"] >> enable_gnc;
    settings["enable_alternate_opt"] >> enable_alternate_opt;

    string img_l_topic, img_r_topic, odom_topic;
    settings["img_l_topic"] >> img_l_topic;
    settings["img_r_topic"] >> img_r_topic;
    settings["odom_topic"] >> odom_topic;

    Subscriber<Image> img_l_sub(nh, img_l_topic, 1);
    Subscriber<Image> img_r_sub(nh, img_r_topic, 1);
    TimeSynchronizer<Image, Image> sync(img_l_sub, img_r_sub, 10);

    sync.registerCallback(boost::bind(&imgCallback, _1, _2));
    ros::Subscriber odom_sub = nh.subscribe(odom_topic, 10, odomCallback);

    ros::AsyncSpinner spinner(2);
    spinner.start();

    while (ros::ok())
    {
        switch (state)
        {
        case FsmState::WAITING_FOR_IMG:
        {
            LOG(INFO) << "Waiting for img to be published @_@";
            if (recv_odom)
            {
                lock_guard<mutex> lock(odom_mutex);
                visualizer->pubPose(odom2, "odom2");
            }
            this_thread::sleep_for(chrono::milliseconds(333));
            break;
        }
        case FsmState::INIT:
        {
            {
                lock_guard<mutex> lock(img_mutex);
                cv::imshow("img_l", img2_l_ptr->image);
            }
            cv::waitKey(50);
            if (recv_odom)
            {
                lock_guard<mutex> lock(odom_mutex);
                visualizer->pubPose(odom2, "odom2");
            }
            break;
        }
        case FsmState::SOLVING_BY_TRIGGER:
        {
            if (trigger)
            {
                nav_msgs::Odometry odom1_, odom2_;
                if (recv_odom)
                {
                    lock_guard<mutex> lock(odom_mutex);
                    visualizer->pubPose(odom2, "odom2");
                    odom1_ = odom1;
                    odom2_ = odom2;
                }

                cv::Mat img1_l, img1_r, img2_l, img2_r;
                {
                    lock_guard<mutex> lock(img_mutex);
                    img1_l = img1_l_ptr->image.clone();
                    img1_r = img1_r_ptr->image.clone();
                    img2_l = img2_l_ptr->image.clone();
                    img2_r = img2_r_ptr->image.clone();
                }
                Matrix3d R12;
                Vector3d t12;
                bool recover_pose_success = estimator.estimate(img1_l, img1_r, img2_l, img2_r, R12, t12);

                visualizer->draw(draw_type);
                // visualizer->pubKps3d(RPE::kps3d1_stereo, Matrix3d::Identity(), Vector3d::Zero(), "kps3d1_stereo");
                // visualizer->pubKps3d(RPE::kps3d2_stereo, Matrix3d::Identity(), Vector3d::Zero(), "kps3d2_stereo");

                if (recv_odom && recover_pose_success)
                {
                    const Matrix3d R1(Quaterniond(odom1_.pose.pose.orientation.w,
                                                  odom1_.pose.pose.orientation.x,
                                                  odom1_.pose.pose.orientation.y,
                                                  odom1_.pose.pose.orientation.z));
                    const Vector3d t1(odom1_.pose.pose.position.x,
                                      odom1_.pose.pose.position.y,
                                      odom1_.pose.pose.position.z);

                    visualizer->pubKps3d(RPE::kps3d1_matched, odom1_, "kps3d1_gt");
                    visualizer->pubKps3d(RPE::kps3d2_matched, odom2_, "kps3d2_gt");

                    // x = R1 * (R12 * x2 + t12) + t1 = (R1 * R12) * x2 + (R1 * t12 + t1)

                    if (enable_ransac)
                    {
                        const Matrix3d R2_ransan = R1 * RPE::R12_ransac;
                        const Vector3d t2_ransan = R1 * RPE::t12_ransac + t1;
                        visualizer->pubPose(R2_ransan, t2_ransan, "pose2_ransac");
                        visualizer->pubKps3d(RPE::kps3d2_matched, R2_ransan, t2_ransan, "kps3d2_ransac");
                    }

                    if (enable_gnc)
                    {
                        const Matrix3d R2_gnc = R1 * RPE::R12_gnc;
                        const Vector3d t2_gnc = R1 * RPE::t12_gnc + t1;
                        visualizer->pubPose(R2_gnc, t2_gnc, "pose2_gnc");
                        visualizer->pubKps3d(RPE::kps3d2_matched, R2_gnc, t2_gnc, "kps3d2_gnc");
                    }

                    if (enable_alternate_opt)
                    {
                        const Matrix3d R2 = R1 * R12;
                        const Vector3d t2 = R1 * t12 + t1;
                        visualizer->pubPose(R2, t2, "pose2_altopt");
                        visualizer->pubKps3d(RPE::kps3d2_matched, R2, t2, "kps3d2_altopt");
                    }
                }
                trigger = false;
            }
            else
            {
                {
                    cv::Mat img_concat;
                    lock_guard<mutex> lock(img_mutex);
                    cv::hconcat(img1_l_ptr->image, img2_l_ptr->image, img_concat);
                    cv::resize(img_concat, img_concat, cv::Size(), 0.66, 0.66);
                    cv::imshow("img1_l and img2_l", img_concat);
                }
                cv::waitKey(50);
            }
            break;
        }
        case FsmState::CONTINUOUS_SOLVING:
        {
            nav_msgs::Odometry odom1_, odom2_;
            if (recv_odom)
            {
                lock_guard<mutex> lock(odom_mutex);
                visualizer->pubPose(odom2, "odom2");
                odom1_ = odom1;
                odom2_ = odom2;
            }

            cv::Mat img1_l, img1_r, img2_l, img2_r;
            {
                lock_guard<mutex> lock(img_mutex);
                img1_l = img1_l_ptr->image.clone();
                img1_r = img1_r_ptr->image.clone();
                img2_l = img2_l_ptr->image.clone();
                img2_r = img2_r_ptr->image.clone();
            }
            Matrix3d R12;
            Vector3d t12;
            bool recover_pose_success = estimator.estimate(img1_l, img1_r, img2_l, img2_r, R12, t12);

            visualizer->draw(draw_type);
            // visualizer->pubKps3d(RPE::kps3d1_stereo, Matrix3d::Identity(), Vector3d::Zero(), "kps3d1_stereo");
            // visualizer->pubKps3d(RPE::kps3d2_stereo, Matrix3d::Identity(), Vector3d::Zero(), "kps3d2_stereo");

            if (recv_odom && recover_pose_success)
            {
                const Matrix3d R1(Quaterniond(odom1_.pose.pose.orientation.w,
                                              odom1_.pose.pose.orientation.x,
                                              odom1_.pose.pose.orientation.y,
                                              odom1_.pose.pose.orientation.z));
                const Vector3d t1(odom1_.pose.pose.position.x,
                                  odom1_.pose.pose.position.y,
                                  odom1_.pose.pose.position.z);

                visualizer->pubKps3d(RPE::kps3d1_matched, odom1_, "kps3d1_gt");
                visualizer->pubKps3d(RPE::kps3d2_matched, odom2_, "kps3d2_gt");

                // x = R1 * (R12 * x2 + t12) + t1 = (R1 * R12) * x2 + (R1 * t12 + t1)

                if (enable_ransac)
                {
                    const Matrix3d R2_ransan = R1 * RPE::R12_ransac;
                    const Vector3d t2_ransan = R1 * RPE::t12_ransac + t1;
                    visualizer->pubPose(R2_ransan, t2_ransan, "pose2_ransac");
                    visualizer->pubKps3d(RPE::kps3d2_matched, R2_ransan, t2_ransan, "kps3d2_ransac");
                }

                if (enable_gnc)
                {
                    const Matrix3d R2_gnc = R1 * RPE::R12_gnc;
                    const Vector3d t2_gnc = R1 * RPE::t12_gnc + t1;
                    visualizer->pubPose(R2_gnc, t2_gnc, "pose2_gnc");
                    visualizer->pubKps3d(RPE::kps3d2_matched, R2_gnc, t2_gnc, "kps3d2_gnc");
                }

                if (enable_alternate_opt)
                {
                    const Matrix3d R2 = R1 * R12;
                    const Vector3d t2 = R1 * t12 + t1;
                    visualizer->pubPose(R2, t2, "pose2_altopt");
                    visualizer->pubKps3d(RPE::kps3d2_matched, R2, t2, "kps3d2_altopt");
                }
            }
            break;
        }
        }
    }

    return 0;
}