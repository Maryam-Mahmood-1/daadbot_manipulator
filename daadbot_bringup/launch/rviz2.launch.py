from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    xacro_file = os.path.join(
        get_package_share_directory('daadbot_desc'),
        'urdf/urdf_inverted_torque/daadbot.urdf.xacro'
    )

    model_arg = DeclareLaunchArgument(
        name="model", 
        default_value=xacro_file,
        description='Absolute path to robot urdf file'
    )

    robot_description = ParameterValue(Command(['xacro ', LaunchConfiguration('model')]), value_type=str)

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}]
    )

    # NOTE: joint_state_publisher_gui is REMOVED here.
    # We rely on your CRCLF.py script to publish the joint states.

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', os.path.join(get_package_share_directory('daadbot_desc'), 'rviz/display.rviz')]
    )

    return LaunchDescription([
        model_arg,
        robot_state_publisher_node,
        rviz_node
    ])