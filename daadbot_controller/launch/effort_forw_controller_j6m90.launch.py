from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    controller_manager_name = LaunchConfiguration("controller_manager_name")

    controller_manager_name_arg = DeclareLaunchArgument(
        "controller_manager_name",
        default_value="controller_manager",
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            controller_manager_name,
        ],
    )

    arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "effort_arm_controller",
            "--controller-manager",
            controller_manager_name,
        ],
    )

    gripper_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "effort_gripper_controller",
            "--controller-manager",
            controller_manager_name,
        ],
    )

    return LaunchDescription(
        [
            controller_manager_name_arg,
            joint_state_broadcaster_spawner,
            arm_controller_spawner,
            gripper_controller_spawner,
        ]
    )
