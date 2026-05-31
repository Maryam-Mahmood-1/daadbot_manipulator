from launch import LaunchDescription
from launch_ros.actions import Node
import os
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command, LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    robot_description = ParameterValue(
        Command(
            [
                "xacro ",
                os.path.join(get_package_share_directory('daadbot_desc'), 'urdf/urdf_oct_effort/daadbot.urdf.xacro')
            ]
        ),
        value_type = str
    )

    robot_state_publisher_node = Node(
        package = 'robot_state_publisher', 
        executable = 'robot_state_publisher',
        parameters = [{"robot_description": robot_description}],
        )
    
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
        ]
    )

    effort_arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "effort_arm_controller",
        ]
    )

    effort_gripper_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "effort_gripper_controller",
        ]
    )

    return LaunchDescription([
        robot_state_publisher_node,
        joint_state_broadcaster_spawner,
        effort_arm_controller_spawner,
        effort_gripper_controller_spawner
    ])