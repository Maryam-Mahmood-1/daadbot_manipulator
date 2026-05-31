from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, IncludeLaunchDescription
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
        # UPDATE: Pointing to the new .xacro file
        default_value=os.path.join(daadbot_desc_dir, 'urdf/2_link_urdf/2link_robot_noisy_3.urdf'),
        description='Absolute path to robot urdf file'
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
    # The 'Command' automatically processes the Xacro file we passed above
    robot_description = ParameterValue(Command([
        "xacro ",
        LaunchConfiguration("model"),
        " is_ignition:=", is_ignition
    ]), value_type=str)

    # Robot State Publisher
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{"robot_description": robot_description, "use_sim_time": True}]
    )

    # Launch Gazebo Sim 
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('ros_gz_sim'),
            'launch'
        ), '/gz_sim.launch.py']),
        launch_arguments=[("gz_args", [" -v 4 -r empty.sdf ", physics_engine])]
    )

    # Spawn DaadBot in the world
    gz_spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'daadbot',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.05', # Pro-Tip: Raised slightly (0.05) so it doesn't spawn inside the floor
            '-R', '0',
            '-P', '0',
            '-Y', '0'
        ],
        output='screen'
    )

    # Bridge for Gazebo <--> ROS 2 Communication
    gz_ros2_bridge = Node(
        package = 'ros_gz_bridge',
        executable = 'parameter_bridge',
        arguments= ['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock']
    )

    return LaunchDescription([
        model_arg,
        robot_state_publisher_node,
        gazebo_resource_path,
        gazebo,
        gz_spawn_entity,
        gz_ros2_bridge
    ])