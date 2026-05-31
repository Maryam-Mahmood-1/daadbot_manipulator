from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory 
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node 
import os

def generate_launch_description():
    is_sim_arg = DeclareLaunchArgument(
        "is_sim",
        default_value="True"
    )

    is_sim = LaunchConfiguration("is_sim")
    moveit_config = (MoveItConfigsBuilder("daadbot", package_name="daadbot_moveit")
                    .robot_description(file_path=os.path.join(get_package_share_directory("daadbot_desc"), "urdf/urdf_table_vel", "daadbot.urdf.xacro"))
                    .robot_description_semantic(file_path="config/daadbot_table.srdf")
                    .trajectory_execution(file_path="config/moveit_controllers_vel.yaml")  # This was missing
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
            os.path.join(get_package_share_directory("daadbot_moveit"), "config", "moveit_controllers_vel.yaml"),
            os.path.join(get_package_share_directory("daadbot_moveit"), "config", "kinematics.yaml")
        ],
        arguments=["--ros-args", "--log-level", "info"]
    )


    # rviz_config = os.path.join(get_package_share_directory("daadbot_moveit"), "config", "moveit7.rviz")
    rviz_config = os.path.join(get_package_share_directory("daadbot_moveit"), "config", "moveit8.rviz")

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits
        ]
    )
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('realsense2_camera'),
                'launch',
                'rs_launch.py'
            )
        ),
        launch_arguments={
            "align_depth.enable": "true",
            "pointcloud.enable": "true",
            "colorizer.enable": "true",
            "decimation_filter.enable": "true",
            "spatial_filter.enable": "true",
            "temporal_filter.enable": "true",
            "disparity_filter.enable": "true",
            "hole_filling_filter.enable": "true",
            "hdr_merge.enable": "true"
        }.items()
    )

    return LaunchDescription([
        is_sim_arg,
        move_group_node,
        rviz_node,
        realsense_launch
    ])