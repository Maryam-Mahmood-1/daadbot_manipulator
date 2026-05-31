cmake_minimum_required(VERSION 3.8)
project(moveit_pose)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(geometry_msgs REQUIRED)

# Define the executable
add_executable(hand_pose_node src/hand_pose.cpp)

# Include directories
target_include_directories(hand_pose_node PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# Link dependencies
ament_target_dependencies(hand_pose_node
  rclcpp
  moveit_ros_planning_interface
  geometry_msgs
)

add_executable(arm_pose_node src/arm_pose.cpp)

# Include directories
target_include_directories(arm_pose_node PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# Link dependencies
ament_target_dependencies(arm_pose_node
  rclcpp
  moveit_ros_planning_interface
  geometry_msgs
)

# Install the executable
install(TARGETS hand_pose_node arm_pose_node
  DESTINATION lib/${PROJECT_NAME}
)

# Export package
ament_package()
