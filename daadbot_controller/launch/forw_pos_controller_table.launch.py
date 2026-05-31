from launch import LaunchDescription
from launch_ros.actions import Node
import os
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command, LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.conditions import UnlessCondition

def generate_launch_description():

    is_sim = LaunchConfiguration("is_sim")
    controller_manager_name = LaunchConfiguration("controller_manager_name")

    is_sim_arg = DeclareLaunchArgument("is_sim", default_value="True")
    controller_manager_name_arg = DeclareLaunchArgument(
        "controller_manager_name", default_value="controller_manager"
    )

    robot_description = ParameterValue(
        Command([
            "xacro ",
            os.path.join(get_package_share_directory('daadbot_desc'), 'urdf/urdf_inverted_pos_forw/daadbot.urdf.xacro'),
            " is_sim:=False"
        ]),
        value_type=str
    )

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            "robot_description": robot_description,
            "use_sim_time": False
        }],
        condition=UnlessCondition(is_sim),
    )

    controller_manager = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            {"robot_description": robot_description, "use_sim_time": is_sim},
            os.path.join(
                get_package_share_directory("daadbot_controller"),
                "config",
                "forw_pos_controller.yaml",
            ),
        ],
        condition=UnlessCondition(is_sim),
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager", controller_manager_name
        ]
    )

    arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "position_forw_arm_controller",
            "--controller-manager", controller_manager_name
        ]
    )

    gripper_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "position_forw_gripper_controller",
            "--controller-manager", controller_manager_name
        ]
    )

    return LaunchDescription([
        is_sim_arg,
        controller_manager_name_arg,
        robot_state_publisher_node,
        controller_manager,
        joint_state_broadcaster_spawner,
        arm_controller_spawner,
        gripper_controller_spawner
    ])
