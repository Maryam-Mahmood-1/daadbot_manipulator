import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("daadbot_desc"),
                "launch",
                "gazebo_table_vel.launch.py"
            )
        )
    )

    controller = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("daadbot_controller"),
                "launch",
                "vel_controller_table.launch.py"
            )
        ),
        launch_arguments={"is_sim": "True"}.items()
    )

    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("daadbot_moveit"),
                "launch",
                "moveit_table_sim_ws.launch.py"
            )
        ),
        launch_arguments={"is_sim": "True"}.items()
    )

    z_torque_aggregator = Node(
        package="some_examples_py",
        executable="z_torque_aggregator",
        name="z_torque_aggregator",
        output="screen"
    )

    rviz_torque_text = Node(
        package="some_examples_py",
        executable="rviz_torque_text",
        name="rviz_torque_text",
        output="screen"
    )

    # Delay torque-related nodes by 3 seconds
    z_torque_aggregator_delayed = TimerAction(
        period=0.0,
        actions=[z_torque_aggregator]
    )

    rviz_torque_text_delayed = TimerAction(
        period=0.0,
        actions=[rviz_torque_text]
    )

    tf_visualizer = Node(
        package="some_examples_py",
        executable="tf_visualizer",
        name="tf_visualizer",
        output="screen"
    )

    tf_visualizer_delayed = TimerAction(
        period=3.0,
        actions=[tf_visualizer]
    )



    return LaunchDescription([
        gazebo,
        controller,
        moveit,
        z_torque_aggregator_delayed,
        rviz_torque_text_delayed,
        tf_visualizer_delayed,
    ])
