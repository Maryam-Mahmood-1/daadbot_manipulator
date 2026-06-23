"""Reachability/capability-map visualizer for urdf_table_pos_j6m90.

Loads the robot model + SRDF, runs the reachability_map node (FK sampling within
joint limits, voxelized, coloured by an orientation-diversity reachability
index), and opens RViz. No move_group needed.

    ros2 launch daadbot_moveit reachability_table_pos_j6m90.launch.py
    ros2 launch daadbot_moveit reachability_table_pos_j6m90.launch.py \
        n_samples:=500000 voxel_size:=0.04 check_collision:=true

In RViz: Fixed Frame -> base_link, Add -> MarkerArray -> /reachability_markers.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    n_samples = LaunchConfiguration("n_samples")
    voxel_size = LaunchConfiguration("voxel_size")
    check_collision = LaunchConfiguration("check_collision")
    alpha_min = LaunchConfiguration("alpha_min")
    alpha_max = LaunchConfiguration("alpha_max")
    alpha_by_reachability = LaunchConfiguration("alpha_by_reachability")

    args = [
        DeclareLaunchArgument("n_samples", default_value="300000"),
        DeclareLaunchArgument("voxel_size", default_value="0.05"),
        DeclareLaunchArgument("check_collision", default_value="false"),
        # transparency: alpha = alpha_min + (alpha_max-alpha_min)*reachability
        # set alpha_by_reachability:=false for a single uniform alpha (= alpha_max)
        DeclareLaunchArgument("alpha_min", default_value="0.35"),
        DeclareLaunchArgument("alpha_max", default_value="1.0"),
        DeclareLaunchArgument("alpha_by_reachability", default_value="true"),
    ]

    pkg_desc = get_package_share_directory("daadbot_desc")

    moveit_config = (
        MoveItConfigsBuilder("daadbot", package_name="daadbot_moveit")
        .robot_description(
            file_path=os.path.join(
                pkg_desc, "urdf", "urdf_table_pos_j6m90", "daadbot.urdf"
            )
        )
        .robot_description_semantic(file_path="config/daadbot_table_pos_j6m90.srdf")
        .robot_description_kinematics(
            file_path=os.path.join(
                get_package_share_directory("daadbot_moveit"), "config", "kinematics.yaml"
            )
        )
        .to_moveit_configs()
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[moveit_config.robot_description],
    )

    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
    )

    reachability = Node(
        package="some_examples_cpp",
        executable="reachability_map",
        name="reachability_map",
        output="screen",
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            {
                "group": "arm",
                "ee_link": "endeffector",
                "base_frame": "base_link",
                "n_samples": n_samples,
                "voxel_size": voxel_size,
                "check_collision": check_collision,
                "alpha_by_reachability": alpha_by_reachability,
                "alpha_min": alpha_min,
                "alpha_max": alpha_max,
            },
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        parameters=[moveit_config.robot_description],
    )

    return LaunchDescription(args + [
        robot_state_publisher,
        joint_state_publisher,
        reachability,
        rviz,
    ])
