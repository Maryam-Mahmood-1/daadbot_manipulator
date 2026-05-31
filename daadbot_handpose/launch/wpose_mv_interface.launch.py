import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import UnlessCondition, IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    is_sim_arg = DeclareLaunchArgument(
        "is_sim",
        default_value="True"
    )

    use_python_arg = DeclareLaunchArgument(
        "use_python",
        default_value="False",
    )

    use_python = LaunchConfiguration("use_python")
    is_sim = LaunchConfiguration("is_sim")

    wrist_pose_node = Node(
        package="daadbot_handpose",
        executable="wrist_pose_node",
        condition=UnlessCondition(use_python),
        parameters=[{"use_sim_time": is_sim}]
    )

    moveit_config = (
        MoveItConfigsBuilder("daadbot", package_name="daadbot_moveit")
        .robot_description(file_path=os.path.join(
            get_package_share_directory("daadbot_desc"),
            "urdf/urdf_oct",
            "daadbot.urdf.xacro"
            )
        )
        .robot_description_semantic(file_path="config/daadbot.srdf")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .moveit_cpp(file_path="config/planning_python_api.yaml")
        .to_moveit_configs()
    )

    

    

    return LaunchDescription([
        use_python_arg,
        is_sim_arg,
        wrist_pose_node,
    ])