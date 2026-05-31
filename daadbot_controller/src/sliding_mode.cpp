#include "/home/maryam-mahmood/xdaadbot_ws/src/daadbot_controller/include/daadbot_controller/sliding_mode_control.hpp""

SlidingModeControl::SlidingModeControl() : Node("sliding_mode_control")
{
  sub_ = create_subscription<sensor_msgs::msg::JointState>(
      "joint_states", 10, std::bind(&SlidingModeControl::jointStateCallback, this, std::placeholders::_1));

  pub_ = create_publisher<trajectory_msgs::msg::JointTrajectory>("arm_controller/joint_trajectory", 10);

  desired_positions_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // Set desired joint positions
  lambda_ = 5.0;  // SMC gain
  k_s_ = 2.0;  // Switching gain

  RCLCPP_INFO(get_logger(), "Sliding Mode Control Node started");
}

void SlidingModeControl::jointStateCallback(const sensor_msgs::msg::JointState &msg)
{
  if (msg.position.size() < 7 || msg.velocity.size() < 7) {
    RCLCPP_WARN(get_logger(), "Insufficient joint data received!");
    return;
  }

  trajectory_msgs::msg::JointTrajectory command;
  command.joint_names = {"joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"};

  trajectory_msgs::msg::JointTrajectoryPoint control_signal;
  control_signal.velocities.resize(7);

  for (size_t i = 0; i < 7; i++)
  {
    double e = msg.position[i] - desired_positions_[i];   // Position error
    double e_dot = msg.velocity[i];  // Velocity error

    double s = e_dot + lambda_ * e;  // Sliding surface
    double u = -k_s_ * ((s > 0) - (s < 0));  // Sign function

    control_signal.velocities[i] = u;  // Send velocity command
  }

  command.points.push_back(control_signal);
  pub_->publish(command);
}

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<SlidingModeControl>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
