#!/usr/bin/env python3
"""Generate lower-DOF versions of a 7-DOF daadbot arm with a full ros2_control
sim pipeline -- URDF + controller config + controller launch + gazebo launch +
bringup launch -- WITHOUT modifying any original files.

A "k-DOF version" keeps joint_1..joint_k actuated and locks every other
revolute joint (joint_{k+1}..joint_N, plus the gripper/claw joints) by turning
them into <joint type="fixed"> at their zero pose. The full geometry is kept;
only the first k joints move. The embedded <ros2_control> block, the controller
YAML and the joint_state_broadcaster are trimmed to the kept joints, and the
command-interface type (position/velocity/effort) is auto-detected so the right
forward controller is used.

It mirrors the existing j6m90 pipeline:
  daadbot_desc/urdf/<tag>_<k>dof/daadbot.urdf
  daadbot_controller/config/command_controller_<tag>_<k>dof.yaml
  daadbot_controller/launch/controller_<tag>_<k>dof.launch.py
  daadbot_desc/launch/gazebo_<tag>_<k>dof.launch.py
  daadbot_bringup/launch/sim_robot_<tag>_<k>dof.launch.py

Example:
  python3 generate_lower_dof.py \
    --urdf .../urdf/urdf_table_pos_j6m90/daadbot.urdf --dofs 2 3 4 5 6
"""
import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

# command-interface name -> (ros2_control controller type, short label)
CONTROLLER_BY_CMD = {
    'position': ('position_controllers/JointGroupPositionController', 'position'),
    'velocity': ('velocity_controllers/JointGroupVelocityController', 'velocity'),
    'effort':   ('effort_controllers/JointGroupEffortController',   'effort'),
}
ARM_JOINT_RE = re.compile(r'^joint_(\d+)$')
# children that don't belong on a fixed joint
MOVING_JOINT_CHILDREN = ('axis', 'limit', 'dynamics', 'mimic', 'safety_controller')


def find_packages(script_path, args):
    desc = Path(args.desc_pkg).resolve() if args.desc_pkg else \
        script_path.parent.parent                      # daadbot_desc
    manip = desc.parent                                # daadbot_manipulator
    controller = Path(args.controller_pkg).resolve() if args.controller_pkg else \
        manip / 'daadbot_controller'
    bringup = Path(args.bringup_pkg).resolve() if args.bringup_pkg else \
        manip / 'daadbot_bringup'
    for p in (desc, controller, bringup):
        if not p.is_dir():
            raise SystemExit(f'package dir not found: {p}')
    return desc, controller, bringup


def detect_arm_joints(root):
    """Return arm joints ordered by index: [(idx, name), ...]."""
    arm = []
    for j in root.findall('joint'):
        m = ARM_JOINT_RE.match(j.get('name', ''))
        if m and j.get('type') in ('revolute', 'continuous'):
            arm.append((int(m.group(1)), j.get('name')))
    arm.sort()
    return arm


def detect_cmd_type(root, arm_names):
    rc = root.find('ros2_control')
    if rc is None:
        raise SystemExit('URDF has no <ros2_control> block.')
    for j in rc.findall('joint'):
        if j.get('name') in arm_names:
            ci = j.find('command_interface')
            if ci is not None:
                return ci.get('name')
    raise SystemExit('Could not find a command_interface on any arm joint.')


def detect_controller_yaml(urdf_text):
    """The yaml filename referenced by the gazebo plugin <parameters>."""
    m = re.search(r'([A-Za-z0-9_./-]+\.yaml)', urdf_text)
    return Path(m.group(1)).name if m else None


def lock_joint(joint_el):
    joint_el.set('type', 'fixed')
    for tag in MOVING_JOINT_CHILDREN:
        for child in joint_el.findall(tag):
            joint_el.remove(child)


def make_kdof_urdf(template_root, arm, k):
    """Return a deep-copied root locked to k actuated arm joints."""
    import copy
    root = copy.deepcopy(template_root)
    keep = {name for idx, name in arm if idx <= k}

    # 1) lock every revolute/continuous joint that is not a kept arm joint
    for j in root.findall('joint'):
        if j.get('type') in ('revolute', 'continuous') and j.get('name') not in keep:
            lock_joint(j)

    # 2) trim the ros2_control block to the kept joints only
    rc = root.find('ros2_control')
    for j in list(rc.findall('joint')):
        if j.get('name') not in keep:
            rc.remove(j)
    return root, [name for idx, name in arm if idx <= k]


def write_urdf(root, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ET.indent(root)          # py3.9+
    except AttributeError:
        pass
    ET.ElementTree(root).write(path, xml_declaration=True, encoding='utf-8')


# ----------------------------- text templates -----------------------------
def controller_yaml(arm_ctrl, ctrl_type, joints):
    jl = '\n'.join(f'      - {j}' for j in joints)
    return f"""controller_manager:
  ros__parameters:
    update_rate: 1000

    {arm_ctrl}:
      type: {ctrl_type}

    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

{arm_ctrl}:
  ros__parameters:
    joints:
{jl}

joint_state_broadcaster:
  ros__parameters:
    update_rate: 1000
    joints:
{jl}
    interfaces:
      - position
      - velocity
      - effort
"""


def controller_launch(arm_ctrl):
    return f'''from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    controller_manager_name = LaunchConfiguration("controller_manager_name")

    controller_manager_name_arg = DeclareLaunchArgument(
        "controller_manager_name",
        default_value="controller_manager",
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            controller_manager_name,
        ],
    )

    arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "{arm_ctrl}",
            "--controller-manager",
            controller_manager_name,
        ],
    )

    return LaunchDescription(
        [
            controller_manager_name_arg,
            joint_state_broadcaster_spawner,
            arm_controller_spawner,
        ]
    )
'''


def gazebo_launch(urdf_folder, old_yaml, new_yaml):
    return f'''import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    daadbot_desc_dir = get_package_share_directory("daadbot_desc")
    urdf_path = os.path.join(
        daadbot_desc_dir, "urdf", "{urdf_folder}", "daadbot.urdf",
    )

    with open(urdf_path, "r", encoding="utf-8") as urdf_file:
        robot_description = urdf_file.read().replace(
            "{old_yaml}",
            "{new_yaml}",
        )

    gazebo_resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=[str(Path(daadbot_desc_dir).parent.resolve())],
    )

    ros_distro = os.environ.get("ROS_DISTRO")
    physics_engine = (
        ""
        if ros_distro == "humble"
        else "--physics-engine gz-physics-bullet-featherstone-plugin"
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{{"robot_description": robot_description, "use_sim_time": True}}],
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(get_package_share_directory("ros_gz_sim"), "launch"),
                "/gz_sim.launch.py",
            ]
        ),
        launch_arguments=[("gz_args", [" -v 4 -r empty.sdf ", physics_engine])],
    )

    gz_spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-topic", "robot_description",
            "-name", "daadbot",
            "-x", "0.0", "-y", "0.0", "-z", "0.0",
            "-R", "0", "-P", "0", "-Y", "0",
        ],
        output="screen",
    )

    gz_ros2_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
    )

    return LaunchDescription(
        [
            robot_state_publisher_node,
            gazebo_resource_path,
            gazebo,
            gz_spawn_entity,
            gz_ros2_bridge,
        ]
    )
'''


def bringup_launch(gazebo_file, controller_file, rviz='traj_safety2.rviz'):
    return f'''import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg_desc = get_package_share_directory("daadbot_desc")
    pkg_controller = get_package_share_directory("daadbot_controller")

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_desc, "launch", "{gazebo_file}")
        )
    )

    controller = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_controller, "launch", "{controller_file}")
        )
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", os.path.join(pkg_desc, "rviz", "{rviz}")],
    )

    return LaunchDescription([gazebo, controller, rviz_node])
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--urdf', required=True, help='flat URDF with embedded ros2_control + gazebo')
    ap.add_argument('--dofs', type=int, nargs='+', default=[2, 3, 4, 5, 6])
    ap.add_argument('--name', default=None,
                    help='tag for output names (default: urdf folder name minus leading "urdf_")')
    ap.add_argument('--desc-pkg', default=None)
    ap.add_argument('--controller-pkg', default=None)
    ap.add_argument('--bringup-pkg', default=None)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    script_path = Path(__file__).resolve()
    desc, controller, bringup = find_packages(script_path, args)

    urdf_path = Path(args.urdf).resolve()
    urdf_text = urdf_path.read_text()
    template_root = ET.fromstring(urdf_text)

    arm = detect_arm_joints(template_root)
    if not arm:
        raise SystemExit('No arm joints (joint_<n>) found.')
    full_dof = arm[-1][0]
    cmd = detect_cmd_type(template_root, {n for _, n in arm})
    if cmd not in CONTROLLER_BY_CMD:
        raise SystemExit(f'Unsupported command interface "{cmd}".')
    ctrl_type, label = CONTROLLER_BY_CMD[cmd]
    arm_ctrl = f'{label}_arm_controller'
    old_yaml = detect_controller_yaml(urdf_text) or 'controller.yaml'

    tag = args.name or re.sub(r'^urdf_', '', urdf_path.parent.name)
    print(f'input: {urdf_path}')
    print(f'arm joints: {[n for _, n in arm]}  (full DOF={full_dof})')
    print(f'command interface: {cmd} -> {ctrl_type}')
    print(f'gazebo references yaml: {old_yaml}')
    print(f'tag: {tag}\n')

    for k in args.dofs:
        if not (1 <= k < full_dof):
            print(f'skip {k}dof (must be 1..{full_dof - 1})')
            continue
        urdf_folder = f'urdf_{tag}_{k}dof'
        yaml_name = f'command_controller_{tag}_{k}dof.yaml'
        ctrl_launch_name = f'controller_{tag}_{k}dof.launch.py'
        gz_launch_name = f'gazebo_{tag}_{k}dof.launch.py'
        bringup_name = f'sim_robot_{tag}_{k}dof.launch.py'

        root_k, kept = make_kdof_urdf(template_root, arm, k)

        targets = {
            desc / 'urdf' / urdf_folder / 'daadbot.urdf': ('urdf', root_k),
            controller / 'config' / yaml_name:
                ('text', controller_yaml(arm_ctrl, ctrl_type, kept)),
            controller / 'launch' / ctrl_launch_name:
                ('text', controller_launch(arm_ctrl)),
            desc / 'launch' / gz_launch_name:
                ('text', gazebo_launch(urdf_folder, old_yaml, yaml_name)),
            bringup / 'launch' / bringup_name:
                ('text', bringup_launch(gz_launch_name, ctrl_launch_name)),
        }

        print(f'[{k}dof] joints={kept}')
        for path, (kind, payload) in targets.items():
            rel = path.relative_to(desc.parent)
            if args.dry_run:
                print(f'    would write {rel}')
                continue
            if kind == 'urdf':
                write_urdf(payload, path)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(payload)
            print(f'    wrote {rel}')
        print(f'    -> ros2 launch daadbot_bringup {bringup_name}\n')

    if not args.dry_run:
        print('Done. Rebuild to install:  colcon build --packages-select '
              'daadbot_desc daadbot_controller daadbot_bringup')


if __name__ == '__main__':
    main()
