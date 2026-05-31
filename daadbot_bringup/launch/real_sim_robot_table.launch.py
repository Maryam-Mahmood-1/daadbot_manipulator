import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import PushRosNamespace
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # --- SIM GROUP ---
    sim_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("daadbot_desc"),
                "launch",
                "gazebo_table_vel.launch.py"
            )
        )
    )

    sim_controller = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("daadbot_controller"),
                "launch",
                "vel_controller_table.launch.py"
            )
        ),
        launch_arguments={
            "is_sim": "True",
            "controller_manager_name": "/sim/controller_manager"
        }.items()
    )

    sim_moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("daadbot_moveit"),
                "launch",
                "moveit_table.launch.py"
            )
        ),
        launch_arguments={
            "is_sim": "True"
        }.items()
    )

    sim_group = GroupAction([
        PushRosNamespace("sim"),
        sim_gazebo,
        sim_controller,
        sim_moveit
    ])

    # --- REAL GROUP ---
    real_controller = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("daadbot_controller"),
                "launch",
                "vel_controller_table.launch.py"
            )
        ),
        launch_arguments={
            "is_sim": "False",
            "controller_manager_name": "/real/controller_manager"
        }.items()
    )

    real_moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("daadbot_moveit"),
                "launch",
                "moveit_table.launch.py"
            )
        ),
        launch_arguments={
            "is_sim": "False"
        }.items()
    )

    real_group = GroupAction([
        PushRosNamespace("real"),
        real_controller,
        real_moveit
    ])

    return LaunchDescription([
        sim_group,
        real_group
    ])
