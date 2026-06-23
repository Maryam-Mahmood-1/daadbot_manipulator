"""Full MoveIt 2 + Gazebo stack for urdf_table_pos_j6m90 in one command.

Composes:
  1. gazebo_table_pos_j6m90_traj.launch.py  (Gazebo, loads the
     JointTrajectoryControllers from pos_trajectory_controller.yaml)
  2. controller_table_pos_j6m90.launch.py   (spawns joint_state_broadcaster +
     position_arm_controller + position_gripper_controller by name)
  3. moveit_table_pos_j6m90.launch.py       (move_group + RViz MotionPlanning)

    ros2 launch daadbot_bringup sim_robot_table_pos_j6m90_moveit.launch.py
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    pkg_desc = get_package_share_directory("daadbot_desc")
    pkg_controller = get_package_share_directory("daadbot_controller")
    pkg_moveit = get_package_share_directory("daadbot_moveit")

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                pkg_desc, "launch", "gazebo_table_pos_j6m90_traj.launch.py"
            )
        )
    )

    controllers = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                pkg_controller, "launch", "controller_table_pos_j6m90.launch.py"
            )
        )
    )

    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                pkg_moveit, "launch", "moveit_table_pos_j6m90.launch.py"
            )
        ),
        launch_arguments={"is_sim": "True"}.items(),
    )

    return LaunchDescription([gazebo, controllers, moveit])
