import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    controller = IncludeLaunchDescription(
            os.path.join(
                get_package_share_directory("daadbot_controller"),
                "launch",
                "vel_controller.launch.py"
            ),
            launch_arguments={"is_sim": "False"}.items()
        )
    
    moveit = IncludeLaunchDescription(
            os.path.join(
                get_package_share_directory("daadbot_moveit"),
                "launch",
                "moveit_vel.launch.py"
            ),
            launch_arguments={"is_sim": "False"}.items()
        )
    
    
    return LaunchDescription([
        controller,
        moveit,
    ])