# DAADBot Manipulator

> Core ROS 2 stack for **DAADBot** — a custom-built 7-DOF robotic arm with a multi-finger gripper, driven by `ros2_control`, simulated in Gazebo, and planned with MoveIt 2.

<!-- Replace with a real render/photo of the robot -->
<!-- ![DAADBot in Gazebo](docs/daadbot_gazebo.png) -->
<!-- ![DAADBot in MoveIt 2](docs/daadbot_moveit.png) -->

This repository is the foundation of the DAADBot project. It contains the robot description (URDF/Xacro + meshes), the `ros2_control` hardware interface and controllers, MoveIt 2 motion-planning configuration, custom message/action interfaces, and a set of C++/Python examples.

---

## Table of Contents

- [About](#about)
- [Packages](#packages)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Visualize the Robot](#visualize-the-robot-in-rviz)
  - [Simulation](#simulation-gazebo--ros2_control)
  - [MoveIt 2 Motion Planning](#moveit-2-motion-planning)
  - [Real Hardware](#real-hardware)
- [Repository Map](#repository-map)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Contact](#contact)

---

## About

**DAADBot** is a 7 degree-of-freedom serial manipulator equipped with a tendon/gear-actuated multi-finger gripper. The stack is designed to run the *same* MoveIt 2 and `ros2_control` pipeline against either:

- a **Gazebo simulation** (`gz_ros2_control` / `ign_ros2_control`), or
- the **physical robot**, through a custom `hardware_interface::SystemInterface` plugin that talks to the onboard microcontroller over a serial/CAN link.

Multiple robot variants are provided out of the box — table-mounted, inverted/ceiling-mounted, reduced-DOF (2-link, 4-DOF), and a pendulum test rig — together with position, velocity, and effort/torque controller configurations. This makes the repository a flexible test bed for both classical motion planning and advanced control experiments.

### Key Features

- 🦾 Full 7-DOF URDF/Xacro description with collision + visual meshes and a multi-finger gripper
- ⚙️ `ros2_control` integration with position, velocity, and effort/torque controllers
- 🎮 Custom hardware interface for the real robot (serial / CAN bridge to the firmware)
- 🧭 MoveIt 2 configuration (SRDF, kinematics, joint limits, planning pipelines) for sim and hardware
- 🧪 Several robot variants (table, inverted, 4-DOF, 2-link, pendulum) for control experiments
- 📦 Custom `daadbot_msgs` action/message interfaces
- 📚 C++ and Python examples covering MoveIt scripting, trajectory control, IK, and data collection

---

## Packages

| Package | Build type | Description |
|---|---|---|
| **`daadbot_desc`** | `ament_cmake` | Robot description: URDF/Xacro, meshes, Gazebo worlds, and display/Gazebo launch files for every robot variant. |
| **`daadbot_controller`** | `ament_cmake` | `ros2_control` hardware interface (`DaadbotInterface`) for the real robot plus all controller YAML configs and spawner launch files. |
| **`daadbot_moveit`** | `ament_cmake` | MoveIt 2 configuration package — SRDF, kinematics, joint limits, controllers, RViz configs, and `move_group` launch files. |
| **`daadbot_bringup`** | `ament_cmake` | Top-level launch files that compose description + controllers (+ MoveIt) into ready-to-run simulation and hardware bringups. |
| **`daadbot_msgs`** | `ament_cmake` | Shared interfaces: the `DaadbotTaskServer` action, `ZTorques` message, etc. Consumed by all three repositories. |
| **`some_examples_cpp`** | `ament_cmake` | C++ examples: MoveIt interface, pick-and-place executors, joint-trajectory control, reachability, CAN driver, lifecycle nodes. |
| **`some_examples_py`** | `ament_python` | Python examples: MoveIt/CTC demos, IK tools, learning/data-collection utilities, GUI trajectory drivers, MediaPipe demo. |

---

## Getting Started

### Prerequisites

- **Ubuntu 22.04** (or the distro matching your ROS 2 release)
- **ROS 2 Humble or Iron** — [installation guide](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html)
- **Gazebo** (`gz_ros2_control` on Iron+, `ign_ros2_control` on Humble)
- A C++17 toolchain, `colcon`, and `rosdep`

Install the commonly required ROS 2 libraries:

```bash
sudo apt-get update && sudo apt-get install -y \
    ros-${ROS_DISTRO}-ros2-control \
    ros-${ROS_DISTRO}-ros2-controllers \
    ros-${ROS_DISTRO}-moveit \
    ros-${ROS_DISTRO}-xacro \
    ros-${ROS_DISTRO}-joint-state-publisher-gui \
    ros-${ROS_DISTRO}-ros-gz \
    libserial-dev
```

> The real-robot hardware interface depends on **`libserial-dev`** for the serial/CAN link to the firmware.

### Installation

```bash
# 1. Create a workspace
mkdir -p ~/daadbot_manipulator_ws/src
cd ~/daadbot_manipulator_ws/src

# 2. Clone this repository
git clone git@github.com:Maryam-Mahmood-1/daadbot_manipulator.git

# (optional) clone the companion research repositories
git clone git@github.com:Maryam-Mahmood-1/daadbot_manipulator_research.git
git clone git@github.com:Maryam-Mahmood-1/daadbot_manipulator_vision_research.git

# 3. Resolve dependencies
cd ~/daadbot_manipulator_ws
rosdep update
rosdep install --from-paths src --ignore-src -r -y

# 4. Build
source /opt/ros/${ROS_DISTRO}/setup.bash
colcon build --symlink-install
```

Source the workspace in **every** terminal you use it in:

```bash
source ~/daadbot_manipulator_ws/install/setup.bash
```

---

## Usage

> Launch-file names encode the robot variant and control mode, e.g.
> `sim_robot_table` (table-mounted sim), `sim_robot_inverted_forw_torque` (inverted, forward torque command),
> `real_robot_table` (hardware). Browse `daadbot_bringup/launch/`, `daadbot_desc/launch/`, and
> `daadbot_moveit/launch/` for the full set.

### Visualize the Robot in RViz

```bash
ros2 launch daadbot_desc display_table.launch.py
```

### Simulation (Gazebo + ros2_control)

Bring up the table-mounted robot in Gazebo with its controllers spawned:

```bash
ros2 launch daadbot_bringup sim_robot_table.launch.py
```

Other useful bringups:

```bash
# Inverted (ceiling-mounted) robot, forward position command interface
ros2 launch daadbot_bringup sim_robot_inverted_forw_pos.launch.py

# Inverted robot, forward torque/effort command interface (used by the CLF/CBF research)
ros2 launch daadbot_bringup sim_robot_inverted_forw_torque.launch.py

# Reduced 4-DOF inverted variant
ros2 launch daadbot_bringup sim_robot_inverted_forw_torque_4_dof.launch.py
```

### MoveIt 2 Motion Planning

Start `move_group` + RViz against the table-mounted simulation:

```bash
# Terminal 1 — simulation + controllers
ros2 launch daadbot_bringup sim_robot_table.launch.py

# Terminal 2 — MoveIt 2
ros2 launch daadbot_moveit moveit_table_sim.launch.py
```

You can then plan and execute motions interactively from the MoveIt RViz plugin, or drive the arm/gripper programmatically via the examples in `some_examples_cpp` / `some_examples_py`.

### Real Hardware

The `daadbot_controller` package provides the `DaadbotInterface` hardware plugin, which opens a serial port (passed as the `port` hardware parameter in the URDF) to communicate with the robot's microcontroller.

```bash
# Bring up the physical table-mounted robot
ros2 launch daadbot_bringup real_robot_table.launch.py

# MoveIt 2 against hardware
ros2 launch daadbot_moveit moveit_table_hw.launch.py
```

> ⚠️ Before running on hardware, confirm the serial port and baud rate in the URDF/Xacro match your setup, that the user has permission to access the port (e.g. `sudo usermod -aG dialout $USER`), and that the robot has a clear, safe workspace.

---

## Repository Map

```
daadbot_manipulator/
├── daadbot_desc/          # URDF/Xacro, meshes, Gazebo worlds, display & gazebo launch files
│   ├── urdf/              # Robot variants: table, inverted, 4-DOF, 2-link, pendulum, ...
│   ├── meshes/  dae/      # Visual & collision geometry
│   ├── worlds/            # Gazebo world/SDF files (table, bottle, empty, ...)
│   └── launch/            # display_*.launch.py, gazebo_*.launch.py
├── daadbot_controller/    # ros2_control hardware interface + controller configs
│   ├── src/               # DaadbotInterface (SystemInterface plugin)
│   ├── config/            # position / velocity / effort controller YAMLs
│   └── launch/            # controller spawner launch files
├── daadbot_moveit/        # MoveIt 2 config (SRDF, kinematics, joint limits, RViz, launch)
├── daadbot_bringup/       # Top-level sim & hardware bringup launch files
├── daadbot_msgs/          # Shared action/message interfaces (DaadbotTaskServer, ZTorques)
├── some_examples_cpp/     # C++ examples (MoveIt, pick-place, JTC, reachability, CAN)
└── some_examples_py/      # Python examples (demos, ik_tools, learning, data_collection)
```

---

## Troubleshooting

1. **MoveIt joint-limit type error** (e.g. `parameter ... max_velocity ... is of type {double}, setting it to {string} is not allowed`):
   set the numeric locale before launching —
   ```bash
   export LC_NUMERIC=en_US.UTF-8
   ```

2. **Serial port permission denied** when launching the real robot: add your user to the `dialout` group and re-login —
   ```bash
   sudo usermod -aG dialout $USER
   ```

3. **Controllers fail to spawn / stay in `unconfigured`:** verify the controller YAML in `daadbot_controller/config/` matches the command interface (position/velocity/effort) declared in the URDF you launched.

4. **A stale Gazebo instance blocks startup:**
   ```bash
   pkill -f gz_sim ; pkill -f gzserver
   ```

---

## License

Interfaces in `daadbot_msgs` are released under the **MIT** license; `some_examples_cpp` under **BSD-3-Clause**. Other packages are research code — please contact the maintainer before redistribution.

---

## Contact

**Maryam Mahmood** — School of Electrical Engineering & Computer Science (SEECS), NUST
📧 mmahmood.msee23seecs@seecs.edu.pk
🔗 https://github.com/Maryam-Mahmood-1
