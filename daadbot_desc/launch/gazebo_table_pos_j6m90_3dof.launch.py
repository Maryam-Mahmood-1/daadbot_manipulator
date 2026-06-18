import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    daadbot_desc_dir = get_package_share_directory("daadbot_desc")
    urdf_path = os.path.join(
        daadbot_desc_dir, "urdf", "urdf_table_pos_j6m90_3dof", "daadbot.urdf",
    )

    with open(urdf_path, "r", encoding="utf-8") as urdf_file:
        robot_description = urdf_file.read().replace(
            "pos_trajectory_controller.yaml",
            "command_controller_table_pos_j6m90_3dof.yaml",
        )

    gazebo_resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=[str(Path(daadbot_desc_dir).parent.resolve())],
    )

    ros_distro = os.environ.get("ROS_DISTRO")
    physics_engine = (
        ""
        if ros_distro == "humble"
        else "--physics-engine gz-physics-bullet-featherstone-plugin"
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_description, "use_sim_time": True}],
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(get_package_share_directory("ros_gz_sim"), "launch"),
                "/gz_sim.launch.py",
            ]
        ),
        launch_arguments=[("gz_args", [" -v 4 -r empty.sdf ", physics_engine])],
    )

    gz_spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-topic", "robot_description",
            "-name", "daadbot",
            "-x", "0.0", "-y", "0.0", "-z", "0.0",
            "-R", "0", "-P", "0", "-Y", "0",
        ],
        output="screen",
    )

    gz_ros2_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
    )

    return LaunchDescription(
        [
            robot_state_publisher_node,
            gazebo_resource_path,
            gazebo,
            gz_spawn_entity,
            gz_ros2_bridge,
        ]
    )
