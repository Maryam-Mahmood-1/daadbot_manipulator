"""MoveIt 2 move_group + RViz for the urdf_table_vel_j6m90 model.

Planning layer only — expects Gazebo already running with velocity_arm_controller
and velocity_gripper_controller (JointTrajectoryController, velocity interface).

Full stack in one command:
    ros2 launch daadbot_bringup sim_robot_table_vel_j6m90_moveit.launch.py
Standalone (sim started elsewhere):
    ros2 launch daadbot_moveit moveit_table_vel_j6m90.launch.py
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    is_sim_arg = DeclareLaunchArgument("is_sim", default_value="True")
    is_sim = LaunchConfiguration("is_sim")

    pkg_desc = get_package_share_directory("daadbot_desc")
    pkg_moveit = get_package_share_directory("daadbot_moveit")

    moveit_config = (
        MoveItConfigsBuilder("daadbot", package_name="daadbot_moveit")
        .robot_description(
            file_path=os.path.join(
                pkg_desc, "urdf", "urdf_table_vel_j6m90", "daadbot.urdf"
            )
        )
        .robot_description_semantic(file_path="config/daadbot_table_vel_j6m90.srdf")
        .trajectory_execution(file_path="config/moveit_controllers_vel_j6m90.yaml")
        .robot_description_kinematics(
            file_path=os.path.join(pkg_moveit, "config", "kinematics.yaml")
        )
        .planning_pipelines("ompl")
        .to_moveit_configs()
    )

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"use_sim_time": is_sim},
            {"publish_robot_description_semantic": True},
            {"moveit_manage_controllers": True},
            os.path.join(pkg_moveit, "config", "moveit_controllers_vel_j6m90.yaml"),
        ],
        arguments=["--ros-args", "--log-level", "info"],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", os.path.join(pkg_moveit, "config", "moveit9.rviz")],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
            {"use_sim_time": is_sim},
        ],
    )

    return LaunchDescription([is_sim_arg, move_group_node, rviz_node])
