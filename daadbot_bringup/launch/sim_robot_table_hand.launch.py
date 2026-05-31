import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # --- Gazebo simulation ---
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("daadbot_desc"),
                "launch",
                "gazebo_table_vel.launch.py"
            )
        )
    )

    # --- Controllers ---
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

    # --- MoveIt ---
    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("daadbot_moveit"),
                "launch",
                "moveit_table_sim.launch.py"
            )
        ),
        launch_arguments={"is_sim": "True"}.items()
    )

    # --- Torque aggregator node ---
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

    # --- Hand pose node ---
    hand_pose_node = Node(
        package="moveit_pose",           # ⚠️ replace with your actual package name
        executable="hand_pose_node",     # ⚠️ must match your C++ target name
        name="hand_pose_node",
        output="screen",
        parameters=[{"use_sim_time": True}]
    )

    # Delay hand_pose_node startup until controllers + MoveIt are ready
    hand_pose_node_delayed = TimerAction(
        period=8.0,   # wait 8 seconds before starting hand_pose_node
        actions=[hand_pose_node]
    )

    return LaunchDescription([
        gazebo,
        controller,
        moveit,
        hand_pose_node_delayed,
        z_torque_aggregator,
        rviz_torque_text,
    ])
