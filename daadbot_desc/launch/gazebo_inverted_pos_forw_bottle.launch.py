from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, IncludeLaunchDescription, ExecuteProcess
from launch.substitutions import Command, LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from pathlib import Path
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    daadbot_desc_dir = get_package_share_directory('daadbot_desc')

    # Declare Xacro model path
    model_arg = DeclareLaunchArgument(
        name="model",
        default_value=os.path.join(daadbot_desc_dir, 'urdf/urdf_inverted_pos_forw_bottle/daadbot.urdf.xacro'),
        description='Absolute path to robot urdf.xacro file'
    )


    # Set Gazebo resource path
    gazebo_resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=[str(Path(daadbot_desc_dir).parent.resolve())]
    )

    # Determine ROS distro and physics engine
    ros_distro = os.environ.get('ROS_DISTRO')
    is_ignition = "True" if ros_distro == 'humble' else "False"
    physics_engine = "" if ros_distro == 'humble' else "--physics-engine gz-physics-bullet-featherstone-plugin"

   


    # Define robot_description parameter
    robot_description = ParameterValue(Command([
        "xacro ",
        LaunchConfiguration("model"),
        " is_ignition:=",
        is_ignition
    ]), value_type=str)

    # Robot State Publisher
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{"robot_description": robot_description, "use_sim_time": True}]
    )

    # Launch Gazebo Sim with `table_world`
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('ros_gz_sim'),
            'launch'
        ), '/gz_sim.launch.py']),
        launch_arguments=[("gz_args", [f" -v 4 -r /home/maryam-mahmood/udaadbot_ws/src/daadbot_desc/worlds/empty.sdf ", physics_engine])]
    )

    # Spawn DaadBot in the world
    gz_spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'daadbot',
            '-x', '0.0',  # X position
            '-y', '0.0',  # Y position
            '-z', '0.0',  # Z position
            '-R', '0',  # Roll
            '-P', '0',  # Pitch
            '-Y', '0'   # Yaw
        ],
        output='screen'
    )



    # Bridge for Gazebo <--> ROS 2 Communication
    gz_ros2_bridge = Node(
        package = 'ros_gz_bridge',
        executable = 'parameter_bridge',
        arguments= ['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'
                ]
    )

    return LaunchDescription([
        model_arg,
        robot_state_publisher_node,
        gazebo_resource_path,
        gazebo,
        gz_spawn_entity,
        gz_ros2_bridge
    ])
