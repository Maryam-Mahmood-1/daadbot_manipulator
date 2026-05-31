#ifndef SLIDING_MODE_CONTROL_HPP
#define SLIDING_MODE_CONTROL_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>
#include <vector>

class SlidingModeControl : public rclcpp::Node
{
public:
  SlidingModeControl();

private:
  void jointStateCallback(const sensor_msgs::msg::JointState &msg);

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr pub_;

  std::vector<double> desired_positions_;
  double lambda_;
  double k_s_;
};

#endif // SLIDING_MODE_CONTROL_HPP
