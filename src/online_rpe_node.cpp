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

mutex img_mutex, odom_mutex; //设置互斥锁保证作用域内语句执行不受其他线程打扰，在作用域结束后自动解锁
cv_bridge::CvImageConstPtr img1_l_ptr, img1_r_ptr; //img_l_topic和img_r_topic的callback会直接传入、一直更新img2，按下空格键时，会保存为img1
cv_bridge::CvImageConstPtr img2_l_ptr, img2_r_ptr; //img_l_topic和img_r_topic的callback会直接传入、一直更新img2，按下空格键时，会保存为img1
nav_msgs::Odometry odom1, odom2; //odom_topic的callback会直接传入、一直更新odom2，按下空格键时，会保存为odom1
bool recv_odom = false; //odom_topic的第一次callback会将此标志激活，后始终保持

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

void keyboardInputThread() //img_l_topic和img_r_topic的第一次callback会启用此线程
{
    while (ros::ok())
    {
        char ch = getch();
        switch (ch)
        {
        case ' ':
        {
            // Capture current img as img1 获取参考帧及位姿
            {
                lock_guard<mutex> lock(img_mutex);
                img1_r_ptr = img2_r_ptr;
                img1_l_ptr = img2_l_ptr;
            }

            if (state == FsmState::INIT) //img_l_topic和img_r_topic的第一次callback会将state设置为INIT
            {
                state = FsmState::SOLVING_BY_TRIGGER;
                LOG(INFO) << "\033[42mINIT\033[0m --> \033[42mSOLVING_BY_TRIGGER\033[0m";
                cv::destroyWindow("img_l");
            }

            if (recv_odom)
            {
                lock_guard<mutex> lock(odom_mutex);
                odom1 = odom2;
                visualizer->pubPose(odom1, "odom1"); //可视化参考帧的位姿
            }
            break;
        }
        case 'c': // Switch to CONTINUOUS_SOLVING mode 持续解算帧间相对位姿 c模式
        {
            if (state == FsmState::SOLVING_BY_TRIGGER) //为检查是否已经获取参考帧及位姿
            {
                state = FsmState::CONTINUOUS_SOLVING;
                LOG(INFO) << "\033[42mSOLVING_BY_TRIGGER\033[0m --> \033[42mCONTINUOUS_SOLVING\033[0m";
                cv::destroyWindow("img1_l and img2_l");
            }
            break;
        }
        case 't': // Switch to SOLVING_BY_TRIGGER mode 按下t键时，解算该时刻的帧间相对位姿 t模式
        {
            if (state == FsmState::CONTINUOUS_SOLVING) 
            {
                state = FsmState::SOLVING_BY_TRIGGER;
                LOG(INFO) << "\033[42mCONTINUOUS_SOLVING\033[0m --> \033[42mSOLVING_BY_TRIGGER\033[0m";
                trigger = false;
            }
            else if (state == FsmState::SOLVING_BY_TRIGGER) //为检查是否已经获取参考帧及位姿
                trigger = true; //使用t模式二级标志（由于t模式一级标志FsmState::SOLVING_BY_TRIGGER无法用于判别是否按下t键）
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
    settings["Visualizer.draw_type"] >> draw_type; //0: FEATURE_EXTRACTION  1: STEREO_MATCHES  2: DESCRIPTOR_MATCHES

    // img_l_topic: /d435i/infra1/image_rect_raw
    // img_r_topic: /d435i/infra2/image_rect_raw
    // odom_topic: /vins_node/odometry
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
        case FsmState::WAITING_FOR_IMG://未收到图像时，显示相机位姿（相机坐标系）
        {
            LOG(INFO) << "Waiting for img to be published @_@";
            if (recv_odom)
            {
                lock_guard<mutex> lock(odom_mutex);
                visualizer->pubPose(odom2, "odom2");  //可视化目标帧的位姿（groundtruth）
            }
            this_thread::sleep_for(chrono::milliseconds(333));
            break;
        }
        case FsmState::INIT: //未获取参考帧及位姿时，显示图像与相机位姿（相机坐标系）
        {
            {
                lock_guard<mutex> lock(img_mutex);
                cv::imshow("img_l", img2_l_ptr->image);
            }
            cv::waitKey(50);
            if (recv_odom)
            {
                lock_guard<mutex> lock(odom_mutex);
                visualizer->pubPose(odom2, "odom2");  //可视化目标帧的位姿（groundtruth）
            }
            break;
        }
        case FsmState::SOLVING_BY_TRIGGER: //t模式一级标志
        {
            if (trigger) //已经获取参考帧及位姿且已进入t模式（t模式二级标志）
            {
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
                bool recover_pose_success = estimator.estimate(img1_l, img1_r, img2_l, img2_r, R12, t12); //恢复帧间相对位姿

                visualizer->draw(draw_type);
                // visualizer->pubKps3d(RPE::kps3d1_debug, Matrix3d::Identity(), Vector3d::Zero(), "kps3d1_debug");
                // visualizer->pubKps3d(RPE::kps3d2_debug, Matrix3d::Identity(), Vector3d::Zero(), "kps3d2_debug");

                if (recv_odom)
                {
                    if (recover_pose_success)
                    {
                        nav_msgs::Odometry odom1_, odom2_;
                        {
                            lock_guard<mutex> lock(odom_mutex);
                            odom1_ = odom1;
                            odom2_ = odom2;
                        }

                        visualizer->pubPose(odom2_, "odom2"); //可视化目标帧的位姿（groundtruth）

                        const Matrix3d R1(Quaterniond(odom1_.pose.pose.orientation.w,
                                                      odom1_.pose.pose.orientation.x,
                                                      odom1_.pose.pose.orientation.y,
                                                      odom1_.pose.pose.orientation.z));
                        const Vector3d t1(odom1_.pose.pose.position.x,
                                          odom1_.pose.pose.position.y,
                                          odom1_.pose.pose.position.z);

                        // x = R1 * (R12 * x2 + t12) + t1 = (R1 * R12) * x2 + (R1 * t12 + t1)
                        const Matrix3d R2 = R1 * R12;
                        const Vector3d t2 = R1 * t12 + t1;

                        visualizer->pubPose(R2, t2, "odom2_estimate"); //可视化目标帧的位姿（estimation）

                        //R1，t1，R2，t2是帧（相机坐标系）相对于启动时刻（全局坐标系）的位姿
                        //kps3d1与kps3d2表示三维点相对于帧（相机坐标系）的位姿
                        //R12是帧间的相对位姿，或者说是三维点间的相对位姿
                        visualizer->pubKps3d(RPE::kps3d1, odom1_, "kps3d1"); 
                        visualizer->pubKps3d(RPE::kps3d2, odom2_, "kps3d2");
                        visualizer->pubKps3d(RPE::kps3d2, R2, t2, "kps3d2_estimate");

                        const Matrix3d odom_R2(Quaterniond(odom2_.pose.pose.orientation.w,
                                                      odom2_.pose.pose.orientation.x,
                                                      odom2_.pose.pose.orientation.y,
                                                      odom2_.pose.pose.orientation.z));
                        const Vector3d odom_t2(odom2_.pose.pose.position.x,
                                          odom2_.pose.pose.position.y,
                                          odom2_.pose.pose.position.z);
                        const Matrix3d Re2g = R2.inverse() * odom_R2;
                        const Vector3d te2g = R2.inverse() * odom_t2 - R2.inverse() * t2;                        
                        visualizer->pubPose(Re2g, te2g, "estimation2groundtruth");

                        const Matrix3d R_parallex = R1.inverse() * odom_R2;
                        const Vector3d t_parallex = R1.inverse() * odom_t2 - R1.inverse() * t1;
                        visualizer->pubPose(R_parallex , t_parallex, "parallex");  
                    }
                    else
                    {
                        lock_guard<mutex> lock(odom_mutex);
                        visualizer->pubPose(odom2, "odom2"); //可视化目标帧的位姿（groundtruth）
                    }
                }
                trigger = false; //退出t模式，形成单次触发效果
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
            //LOG(INFO) << "cd CONTINUOUS_SOLVING @_@";
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
            //LOG(INFO) << "Successfully recover the position @_@";
            visualizer->draw(draw_type);
            // visualizer->pubKps3d(RPE::kps3d1_debug, Matrix3d::Identity(), Vector3d::Zero(), "kps3d1_debug");
            // visualizer->pubKps3d(RPE::kps3d2_debug, Matrix3d::Identity(), Vector3d::Zero(), "kps3d2_debug");
            //LOG(INFO) << "Build the visualization tool @_@";
            if (recv_odom && recover_pose_success)
            {
                nav_msgs::Odometry odom1_, odom2_;
                {
                    lock_guard<mutex> lock(odom_mutex);
                    odom1_ = odom1;
                    odom2_ = odom2;
                }

                visualizer->pubPose(odom2_, "odom2"); //可视化目标帧的位姿（groundtruth）
                //LOG(INFO) << "Pub the odom2 pose @_@";
                const Matrix3d R1(Quaterniond(odom1_.pose.pose.orientation.w,
                                              odom1_.pose.pose.orientation.x,
                                              odom1_.pose.pose.orientation.y,
                                              odom1_.pose.pose.orientation.z));
                const Vector3d t1(odom1_.pose.pose.position.x,
                                  odom1_.pose.pose.position.y,
                                  odom1_.pose.pose.position.z);

                // x1 = R1 * (R12 * x2 + t12) + t1 = (R1 * R12) * x2 + (R1 * t12 + t1)
                const Matrix3d R2 = R1 * R12;
                const Vector3d t2 = R1 * t12 + t1;

                visualizer->pubPose(R2, t2, "odom2_estimate"); //可视化目标帧的位姿（estimation）
                //R1，t1，R2，t2是帧（相机坐标系）相对于启动时刻（全局坐标系）的位姿
                //kps3d1与kps3d2表示三维点相对于帧（相机坐标系）的位姿
                //R12是帧间的相对位姿，或者说是三维点间的相对位姿
                //LOG(INFO) << "Pub the odom2_estimate pose @_@";
                visualizer->pubKps3d(RPE::kps3d1, odom1_, "kps3d1");
                visualizer->pubKps3d(RPE::kps3d2, odom2_, "kps3d2");
                visualizer->pubKps3d(RPE::kps3d2, R2, t2, "kps3d2_estimate");
                //LOG(INFO) << "Pub the 3d point @_@";
                const Matrix3d odom_R2(Quaterniond(odom2_.pose.pose.orientation.w,
                                                odom2_.pose.pose.orientation.x,
                                                odom2_.pose.pose.orientation.y,
                                                odom2_.pose.pose.orientation.z));
                const Vector3d odom_t2(odom2_.pose.pose.position.x,
                                    odom2_.pose.pose.position.y,
                                    odom2_.pose.pose.position.z);
                const Matrix3d Re2g = R2.inverse() * odom_R2;
                const Vector3d te2g = R2.inverse() * odom_t2 - R2.inverse() * t2;

                visualizer->pubPose(Re2g, te2g, "estimation2groundtruth");
                //LOG(INFO) << "Pub the e2g pose @_@";

                const Matrix3d R_parallex = R1.inverse() * odom_R2;
                const Vector3d t_parallex = R1.inverse() * odom_t2 - R1.inverse() * t1;
                visualizer->pubPose(R_parallex , t_parallex, "parallex");
                //LOG(INFO) << "Pub the parallex pose @_@";  
            }
            break;
        }
        }
    }

    return 0;
}