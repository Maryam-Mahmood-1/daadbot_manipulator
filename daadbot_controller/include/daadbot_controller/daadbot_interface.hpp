#ifndef DAADBOT_INTERFACE_H
#define DAADBOT_INTERFACE_H



#include <rclcpp/rclcpp.hpp>
#include <hardware_interface/system_interface.hpp>
#include <libserial/SerialPort.h>
#include <rclcpp_lifecycle/state.hpp>
#include <rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp>

#include <vector>
#include <string>

namespace daadbot_controller
{

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

class DaadbotInterface : public hardware_interface::SystemInterface
{
public:
  DaadbotInterface();
  virtual ~DaadbotInterface();

  // Implementing rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface
  virtual CallbackReturn on_activate(const rclcpp_lifecycle::State &previous_state) override;
  virtual CallbackReturn on_deactivate(const rclcpp_lifecycle::State &previous_state) override;

  // Implementing hardware_interface::SystemInterface
  virtual CallbackReturn on_init(const hardware_interface::HardwareInfo &hardware_info) override;
  virtual std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  virtual std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;
  virtual hardware_interface::return_type read(const rclcpp::Time & time, const rclcpp::Duration & period) override;
  virtual hardware_interface::return_type write(const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  LibSerial::SerialPort esp_;
  std::string port_;
  bool rmp_sent_ = false;    // Flag to check if the initial position read has happened (9C vs A2)
  bool use_dummy_data_ = false;
  bool controllers_ready_ = false;
  bool initial_read_ = false;
  bool power_ = false;


  std::vector<double> velocity_commands_;  // Changed from position_commands_
  std::vector<double> prev_velocity_commands_;  // Changed from prev_speed_commands_
  std::vector<double> prev_position_states_;  // Changed from prev_position_states_
  std::vector<double> position_states_; // Keeping position state for tracking
  std::vector<double> init_position_states_; // Keeping position state for tracking
  std::vector<double> velocity_states_; // Added velocity states for tracking
  std::vector<double> effort_states_; // Added velocity states for tracking
};

}  // namespace daadbot_controller

#endif  // DAADBOT_INTERFACE_H
