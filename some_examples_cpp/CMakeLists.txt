cmake_minimum_required(VERSION 3.8)
project(some_examples_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(daadbot_msgs REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)

# MoveIt core + Eigen (needed for workspace node)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# C++ executables
add_executable(simple_publisher src/simple_publisher.cpp)
ament_target_dependencies(simple_publisher rclcpp std_msgs)

add_executable(my_moveit_interface src/my_moveit_interface.cpp)
ament_target_dependencies(my_moveit_interface rclcpp moveit_ros_planning_interface)

add_executable(lifecycle_node_ex src/lifecycle_node_ex.cpp)
ament_target_dependencies(lifecycle_node_ex rclcpp rclcpp_lifecycle std_msgs)

add_executable(joint_angles_jtc src/joint_angles_jtc.cpp)
ament_target_dependencies(joint_angles_jtc
  rclcpp std_msgs sensor_msgs trajectory_msgs
)

add_executable(joint_angles_jtc_2 src/joint_angles_jtc_2.cpp)
ament_target_dependencies(joint_angles_jtc_2
  rclcpp std_msgs sensor_msgs trajectory_msgs
)

add_executable(joint_angles_jtc_target_increments src/joint_angles_jtc_target_increments.cpp)
ament_target_dependencies(joint_angles_jtc_target_increments
  rclcpp std_msgs sensor_msgs trajectory_msgs
)

add_executable(joint_angles_jtc_target_increments_multi src/joint_angles_jtc_target_increments_multi.cpp)
ament_target_dependencies(joint_angles_jtc_target_increments_multi
  rclcpp std_msgs sensor_msgs trajectory_msgs
)

add_executable(pick_place_sim src/pick_place_sim.cpp)
ament_target_dependencies(pick_place_sim
  rclcpp std_msgs sensor_msgs trajectory_msgs
)

add_executable(hardware_position_controller_trajectory src/hardware_position_controller_trajectory.cpp)
ament_target_dependencies(hardware_position_controller_trajectory
  rclcpp std_msgs sensor_msgs trajectory_msgs
)

add_executable(executor_pick_place src/executor_pick_place.cpp)
ament_target_dependencies(executor_pick_place
  rclcpp std_msgs sensor_msgs trajectory_msgs
)

add_executable(executor_pick_place_2 src/executor_pick_place_2.cpp)
ament_target_dependencies(executor_pick_place_2
  rclcpp std_msgs sensor_msgs trajectory_msgs
)

add_executable(can_driver src/can_driver.cpp)
ament_target_dependencies(can_driver
  rclcpp
  std_msgs
  sensor_msgs
)

add_executable(reachability_ws src/reachability_ws.cpp)
ament_target_dependencies(reachability_ws
  rclcpp
  moveit_ros_planning
  moveit_core
  tf2_eigen
  geometry_msgs
  visualization_msgs
)

# Install C++ executables
install(TARGETS
  simple_publisher
  my_moveit_interface
  lifecycle_node_ex
  joint_angles_jtc
  joint_angles_jtc_2
  joint_angles_jtc_target_increments
  joint_angles_jtc_target_increments_multi
  pick_place_sim
  hardware_position_controller_trajectory
  executor_pick_place
  executor_pick_place_2
  can_driver
  reachability_ws
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python script
install(
  PROGRAMS
    scripts/target_angle_gui.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
