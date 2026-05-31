#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Quaternion.hpp>

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("moveit_pose_node");
    auto logger = node->get_logger();

    RCLCPP_INFO(logger, "Starting MoveIt pose-based movement node...");
    
    moveit::planning_interface::MoveGroupInterface move_group(node, "arm");
    move_group.setPoseReferenceFrame("base_link");
    
    move_group.setPlanningTime(10.0);
    move_group.setMaxVelocityScalingFactor(1.0);
    move_group.setMaxAccelerationScalingFactor(1.0);
    move_group.allowReplanning(true);
    move_group.setGoalTolerance(0.01);
    move_group.setNumPlanningAttempts(10);
    move_group.setPlannerId("RRTConnect");

    RCLCPP_INFO(logger, "Waiting for valid joint states...");
    while (rclcpp::ok()) {
        auto current_state = move_group.getCurrentState();
        if (current_state && current_state->getVariableCount() > 0) {
            break;
        }
        rclcpp::sleep_for(std::chrono::milliseconds(100));
    }
    if (!rclcpp::ok()) {
        RCLCPP_ERROR(logger, "Node was shut down before receiving valid joint states.");
        return 1;
    }
    RCLCPP_INFO(logger, "Received valid joint states!");

    // Get current pose
    auto current_pose = move_group.getCurrentPose().pose;
    RCLCPP_INFO(logger, "Current Pose: x=%f, y=%f, z=%f", current_pose.position.x, current_pose.position.y, current_pose.position.z);

    geometry_msgs::msg::Pose target_pose;
    
    // Set position
    target_pose.position.x = 0.35;
    target_pose.position.y = 0.0;
    target_pose.position.z = 0.6;
    target_pose.orientation.w = 0.015;
    target_pose.orientation.x = 0.1;
    target_pose.orientation.y = 0.01;
    target_pose.orientation.z = 0.0;

    
    
    move_group.setPoseTarget(target_pose);

    // Verify if target pose is within joint limits
    auto joint_model_group = move_group.getCurrentState()->getJointModelGroup("arm");
    moveit::core::RobotStatePtr kinematic_state(new moveit::core::RobotState(*move_group.getCurrentState()));

    if (!kinematic_state->satisfiesBounds(joint_model_group)) {
        RCLCPP_ERROR(logger, "Target pose is outside joint limits!");
        return 1;
    }
    
    // Plan and execute the motion
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool success = (move_group.plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);
    if (success) {
        RCLCPP_INFO(logger, "Executing planned motion...");
        move_group.move();
    } else {
        RCLCPP_ERROR(logger, "Failed to plan motion");
    }

    rclcpp::shutdown();
    return 0;
}
