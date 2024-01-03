import argparse
import rospy
from nav_msgs.msg import Odometry
import numpy as np
from tf.transformations import euler_from_quaternion, quaternion_matrix
import matplotlib.pyplot as plt
from message_filters import TimeSynchronizer, Subscriber


def evaluation_callback(data):
    global num_messages
    global dis_num_below_threshold
    global dis_percentages
    global ori_num_below_threshold
    global ori_percentages
    #print("Successfully receive.")
    num_messages += 1

    # 获取位姿信息并进行处理
    position_x = data.pose.pose.position.x
    position_y = data.pose.pose.position.y
    position_z = data.pose.pose.position.z

    orientation_x = data.pose.pose.orientation.x
    orientation_y = data.pose.pose.orientation.y
    orientation_z = data.pose.pose.orientation.z
    orientation_w = data.pose.pose.orientation.w

    # Convert Quaternion to Euler angles
    quaternion = (orientation_x, orientation_y, orientation_z, orientation_w)
    euler = euler_from_quaternion(quaternion)

    roll = euler[0]
    pitch = euler[1]
    yaw = euler[2]

    #rospy.loginfo("E2G Position: (%f, %f, %f)", position_x, position_y, position_z)
    #rospy.loginfo("E2G Orientation (RPY): (%f, %f, %f)", roll, pitch, yaw)

    # Calculate distance from origin
    distance = np.sqrt(position_x**2 + position_y**2 + position_z**2)


    rospy.loginfo("E2G Distance: %f", distance)

    # Calculate rotation matrix from the quaternion
    rotation_matrix = quaternion_matrix(quaternion)[:3, :3]
    #rospy.loginfo("E2G Rotation Matrix: \n%s", rotation_matrix)

    rotation_angle_radians = np.arccos((np.trace(rotation_matrix) - 1.0) / 2.0) # Angle in radians
    rotation_angle_degrees = np.degrees(rotation_angle_radians)  # Angle in degrees
    rospy.loginfo("E2G Rotation Angle: %f", rotation_angle_degrees)



    if distance < distance_threshold:
        dis_num_below_threshold += 1
    if rotation_angle_degrees < orientation_threshold:
        ori_num_below_threshold += 1       

    dis_percentage = (dis_num_below_threshold / num_messages) * 100
    dis_percentages.append(dis_percentage)
    print("Num:",num_messages,"  Dis_Percentage:", dis_percentage)
    ori_percentage = (ori_num_below_threshold / num_messages) * 100
    ori_percentages.append(ori_percentage)
    print("Num:",num_messages,"  Ori_Percentage:", ori_percentage)



def evaluation_and_paralex_callback(data1,data2):
    global num_messages
    global dis_num_below_threshold
    global dis_percentages
    global ori_num_below_threshold
    global ori_percentages
    #print("Successfully receive.")
    num_messages += 1

    # 获取位姿信息并进行处理
    position_x = data1.pose.pose.position.x
    position_y = data1.pose.pose.position.y
    position_z = data1.pose.pose.position.z

    orientation_x = data1.pose.pose.orientation.x
    orientation_y = data1.pose.pose.orientation.y
    orientation_z = data1.pose.pose.orientation.z
    orientation_w = data1.pose.pose.orientation.w

    # Convert Quaternion to Euler angles
    quaternion = (orientation_x, orientation_y, orientation_z, orientation_w)
    euler = euler_from_quaternion(quaternion)

    roll = euler[0]
    pitch = euler[1]
    yaw = euler[2]

    #rospy.loginfo("E2G Position: (%f, %f, %f)", position_x, position_y, position_z)
    #rospy.loginfo("E2G Orientation (RPY): (%f, %f, %f)", roll, pitch, yaw)

    # Calculate distance from origin
    distance = np.sqrt(position_x**2 + position_y**2 + position_z**2)
    # Calculate rotation matrix from the quaternion
    rotation_matrix = quaternion_matrix(quaternion)[:3, :3]
    #rospy.loginfo("E2G Rotation Matrix: \n%s", rotation_matrix)
    rotation_angle_radians = np.arccos((np.trace(rotation_matrix) - 1.0) / 2.0) # Angle in radians
    rotation_angle_degrees = np.degrees(rotation_angle_radians)  # Angle in degrees


    # 获取位姿信息并进行处理
    par_position_x = data2.pose.pose.position.x
    par_position_y = data2.pose.pose.position.y
    par_position_z = data2.pose.pose.position.z

    par_orientation_x = data2.pose.pose.orientation.x
    par_orientation_y = data2.pose.pose.orientation.y
    par_orientation_z = data2.pose.pose.orientation.z
    par_orientation_w = data2.pose.pose.orientation.w

    # Convert Quaternion to Euler angles
    par_quaternion = (par_orientation_x, par_orientation_y, par_orientation_z, par_orientation_w)
    par_euler = euler_from_quaternion(par_quaternion)

    par_roll = par_euler[0]
    par_pitch = par_euler[1]
    par_yaw = euler[2]

    #rospy.loginfo("E2G Position: (%f, %f, %f)", position_x, position_y, position_z)
    #rospy.loginfo("E2G Orientation (RPY): (%f, %f, %f)", roll, pitch, yaw)

    # Calculate distance from origin
    par_distance = np.sqrt(par_position_x**2 + par_position_y**2 + par_position_z**2)
    # Calculate rotation matrix from the quaternion
    par_rotation_matrix = quaternion_matrix(par_quaternion)[:3, :3]
    #rospy.loginfo("E2G Rotation Matrix: \n%s", rotation_matrix)
    par_rotation_angle_radians = np.arccos((np.trace(par_rotation_matrix) - 1.0) / 2.0) # Angle in radians
    par_rotation_angle_degrees = np.degrees(par_rotation_angle_radians)  # Angle in degrees

    rospy.loginfo("E2G Distance: %f", distance)
    rospy.loginfo("E2G Rotation Angle: %f", rotation_angle_degrees)

    rospy.loginfo("PAR Distance: %f", par_distance)
    rospy.loginfo("PAR Rotation Angle: %f", par_rotation_angle_degrees)

    if distance < distance_threshold:
        dis_num_below_threshold += 1
    if rotation_angle_degrees < orientation_threshold:
        ori_num_below_threshold += 1       

    dis_percentage = (dis_num_below_threshold / num_messages) * 100
    dis_percentages.append(dis_percentage)
    print("Num:",num_messages,"  Dis_Percentage:", dis_percentage)
    ori_percentage = (ori_num_below_threshold / num_messages) * 100
    ori_percentages.append(ori_percentage)
    print("Num:",num_messages,"  Ori_Percentage:", ori_percentage)




if __name__ == '__main__':
    num_messages = 0
    dis_num_below_threshold = 0
    ori_num_below_threshold = 0
    dis_percentages = []
    ori_percentages = []
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dis_thres",
        type=float,
        default=0.2 #m
        
    )
    parser.add_argument(
        "--ori_thres",
        type=float,
        default=10 #deg
    )
    args = parser.parse_args()
    distance_threshold = args.dis_thres
    orientation_threshold = args.ori_thres
    rospy.init_node('evaluator')
    #rospy.Subscriber("/online_rpe_node/estimation2groundtruth", Odometry, evaluation_callback)
    e2g_sub = Subscriber("/online_rpe_node/estimation2groundtruth", Odometry)
    par_sub = Subscriber("/online_rpe_node/parallex", Odometry)

    ts = TimeSynchronizer([e2g_sub, par_sub], 10)
    ts.registerCallback(evaluation_and_paralex_callback)
    print("Start the evaluation.")
    rospy.spin()

