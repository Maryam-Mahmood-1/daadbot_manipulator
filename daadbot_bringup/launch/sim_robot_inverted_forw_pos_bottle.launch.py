import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    gazebo = IncludeLaunchDescription(
            os.path.join(
                get_package_share_directory("daadbot_desc"),
                "launch",
                "gazebo_inverted_pos_forw_bottle.launch.py"
            )
        )
    
    controller = IncludeLaunchDescription(
            os.path.join(
                get_package_share_directory("daadbot_controller"),
                "launch",
                "forw_pos_controller_table_bottle.launch.py"
            ),
            launch_arguments={"is_sim": "True"}.items()
        )
    
  
    
    return LaunchDescription([
        gazebo,
        controller,
    ])