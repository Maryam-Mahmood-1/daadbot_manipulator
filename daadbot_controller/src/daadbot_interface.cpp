#include "daadbot_controller/daadbot_interface.hpp"
#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <iomanip>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace daadbot_controller
{
  
DaadbotInterface::DaadbotInterface()
{
}
DaadbotInterface::~DaadbotInterface()
{
  if (esp_.IsOpen())
  {
    try
    {
      esp_.Close();
    }
    catch (...)
    {
      RCLCPP_FATAL_STREAM(rclcpp::get_logger("DaadbotInterface"),
                          "Something went wrong while closing connection with port " << port_);
    }
  }
}

CallbackReturn DaadbotInterface::on_init(const hardware_interface::HardwareInfo &hardware_info)
{
  CallbackReturn result = hardware_interface::SystemInterface::on_init(hardware_info);
  if (result != CallbackReturn::SUCCESS)
  {
    return result;
  }

  try
  {
    port_ = info_.hardware_parameters.at("port");
  }
  catch (const std::out_of_range &e)
  {
    RCLCPP_FATAL(rclcpp::get_logger("DaadbotInterface"), "No Serial Port provided! Aborting");
    return CallbackReturn::FAILURE;
  }

  RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"),
                     "Detected Joints: " << info_.joints.size());

  velocity_commands_.reserve(info_.joints.size());
  velocity_states_.reserve(info_.joints.size());
  effort_states_.reserve(info_.joints.size());
  position_states_.reserve(info_.joints.size());
  init_position_states_.reserve(info_.joints.size());
  prev_velocity_commands_.reserve(info_.joints.size());
  prev_position_states_.reserve(info_.joints.size());

  return CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> DaadbotInterface::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> state_interfaces;

  for (size_t i = 0; i < info_.joints.size(); i++)
  {
    RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"),
                "Exporting state interface: %s/position", info_.joints[i].name.c_str());
    state_interfaces.emplace_back(hardware_interface::StateInterface(
        info_.joints[i].name, hardware_interface::HW_IF_POSITION, &position_states_[i]));

    RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"),
                "Exporting state interface: %s/velocity", info_.joints[i].name.c_str());
    state_interfaces.emplace_back(hardware_interface::StateInterface(
        info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &velocity_states_[i]));

    RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"),
                "Exporting state interface: %s/effort", info_.joints[i].name.c_str());
    state_interfaces.emplace_back(hardware_interface::StateInterface(
        info_.joints[i].name, hardware_interface::HW_IF_EFFORT, &effort_states_[i]));
  }

  return state_interfaces;
}

std::vector<hardware_interface::CommandInterface> DaadbotInterface::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> command_interfaces;

  for (size_t i = 0; i < info_.joints.size(); i++)
  {
    RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"),
                "Exporting command interface: %s/velocity", info_.joints[i].name.c_str());
    command_interfaces.emplace_back(hardware_interface::CommandInterface(
        info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &velocity_commands_[i]));
  }

  return command_interfaces;
}


#include "rclcpp/rclcpp.hpp"  // Include this header for rclcpp::sleep_for

CallbackReturn DaadbotInterface::on_activate(const rclcpp_lifecycle::State &previous_state)
{
  RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "Starting robot hardware ...");

  velocity_commands_.assign(info_.joints.size(), 0.0);
  prev_velocity_commands_.assign(info_.joints.size(), 0.0);
  prev_position_states_.assign(info_.joints.size(), 0.0);
  position_states_.assign(info_.joints.size(), 0.0);
  init_position_states_.assign(info_.joints.size(), 0.0);
  velocity_states_.assign(info_.joints.size(), 0.0);
  effort_states_.assign(info_.joints.size(), 0.0);

  try
  {
    esp_.Open(port_);
    esp_.SetBaudRate(LibSerial::BaudRate::BAUD_921600);

    // Send "rmp\n" only once
    esp_.Write("rmp\n");
    esp_.DrainWriteBuffer();
    rmp_sent_ = true;

    // Add a delay of 500 milliseconds (adjust as needed)
    rclcpp::sleep_for(std::chrono::milliseconds(50));

    RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"),
                "Sent initial rmp command, ready to take commands.");
  }
  catch (...)
  {
    RCLCPP_FATAL_STREAM(rclcpp::get_logger("DaadbotInterface"),
                        "Something went wrong while interacting with port " << port_);
    return CallbackReturn::FAILURE;
  }

  return CallbackReturn::SUCCESS;
}


CallbackReturn DaadbotInterface::on_deactivate(const rclcpp_lifecycle::State &previous_state)
{
  RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "Stopping robot hardware ...");

  if (esp_.IsOpen())
  {
    try
    {
      esp_.Close();
    }
    catch (...)
    {
      RCLCPP_FATAL_STREAM(rclcpp::get_logger("DaadbotInterface"),
                          "Something went wrong while closing connection with port " << port_);
    }
  }

  RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "Hardware stopped");
  return CallbackReturn::SUCCESS;
}

hardware_interface::return_type DaadbotInterface::read(const rclcpp::Time &time,
                                                       const rclcpp::Duration &period)
{
    RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "Position States ...");
    RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "position_states_ size: %zu", position_states_.size());
  
    RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "Reading ...");

    if (use_dummy_data_)
    {
        static double dummy_angle = 0.0;
        dummy_angle += 0.01;

        for (size_t i = 0; i < position_states_.size(); ++i)
        {
            position_states_[i] = std::sin(dummy_angle + i);
            velocity_states_[i] = std::cos(dummy_angle + i);
            effort_states_[i] = 0.1 * std::sin(dummy_angle + i);
        }
        return hardware_interface::return_type::OK;
    }

    if (!esp_.IsOpen())
    {
        RCLCPP_ERROR(rclcpp::get_logger("DaadbotInterface"), "Serial port not open, cannot read.");
        return hardware_interface::return_type::ERROR;
    }

    try
    {
        std::string response;
        esp_.ReadLine(response, '\n', 500); // timeout = 500 ms

        // Trim leading and trailing whitespace
        response = response.substr(response.find_first_not_of(" \t\n\r"), response.find_last_not_of(" \t\n\r") + 1);
        RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "Trimmed response: '%s'", response.c_str());

        // Check < > and 'R' prefix
        if (response.empty() || response.front() != '<' || response.back() != '>')
        {
            RCLCPP_WARN(rclcpp::get_logger("DaadbotInterface"), "Invalid response: '%s'", response.c_str());
            return hardware_interface::return_type::ERROR;
        }

        response = response.substr(1, response.size() - 2); // remove < >
        if (response.empty() || response.front() != 'R')
        {
            RCLCPP_WARN(rclcpp::get_logger("DaadbotInterface"), "Unexpected response format: '%s'", response.c_str());
            return hardware_interface::return_type::ERROR;
        }
        // RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "Here after checking < > and 'R' prefix");
        std::istringstream ss(response.substr(1)); // Skip the 'R'
        std::string token;
        size_t joint_index = 4;
        while (ss >> token && joint_index <= 6)
        {
            try
            {
                size_t sep1 = token.find('_');
                size_t sep2 = token.rfind('_');

                if (sep1 == std::string::npos || sep2 == std::string::npos || sep1 == sep2)
                {
                    throw std::runtime_error("Invalid token: expected p_v_e format (e.g., 0_0_0)");
                }

                std::string pos_str = token.substr(0, sep1);
                // RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "Position String: '%s'", pos_str.c_str());
                std::string vel_str = token.substr(sep1 + 1, sep2 - sep1 - 1);
                // RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "Velocity String: '%s'", vel_str.c_str());
                std::string eff_str = token.substr(sep2 + 1);
                // RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "Effort String: '%s'", eff_str.c_str());

                auto parse_or_zero = [&](const std::string& str, const std::string& label, size_t idx) -> double {
                    if (str == "E")
                    {
                        RCLCPP_WARN(rclcpp::get_logger("DaadbotInterface"), "Joint %zu: '%s' value estimated, set to 0.", idx, label.c_str());
                        return 0.0;
                    }
                    return std::stod(str);
                };

                double pos = parse_or_zero(pos_str, "position", joint_index);
                double vel = parse_or_zero(vel_str, "velocity", joint_index);
                double eff = parse_or_zero(eff_str, "effort", joint_index);
                RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Position: '" << pos << "'");
                RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Velocity: '" << vel << "'");
                RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Effort: '" << eff << "'");

                RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "prev position: '" << prev_position_states_[joint_index] + init_position_states_[joint_index]<< "'");
                RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "init position: '" << init_position_states_[joint_index] << "'");
                RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "prev position rel: '" << prev_position_states_[joint_index] << "'");


                if (initial_read_)
                {
                  RCLCPP_INFO_STREAM(
                    rclcpp::get_logger("DaadbotInterface"),
                    "1 time step diff: '" << abs(prev_position_states_[joint_index] - ((pos - init_position_states_[joint_index]) * M_PI / 180.0)) << "'"
                );
                                    // Converting degrees to radians before visualizing in Rviz (joint_states)
                    if (abs(prev_position_states_[joint_index] - ((pos - init_position_states_[joint_index]) * M_PI / 180.0)) > 0.2){
                      position_states_[joint_index] = position_states_[joint_index];
                      RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Prev Position: '" << prev_position_states_[joint_index] << "'");
                      RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Currrent Position: '" << (pos - init_position_states_[joint_index]) * M_PI / 180.0 << "'");
                      RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Difference: '" << abs(prev_position_states_[joint_index] - ((pos - init_position_states_[joint_index]) * M_PI / 180.0)) << "'");
                      
                      RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Difference is too high");
                    }
                    else{
                        position_states_[joint_index] = (pos - init_position_states_[joint_index]) * M_PI / 180.0;
                        RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Difference: '" << abs(prev_position_states_[joint_index] - ((pos - init_position_states_[joint_index]) * M_PI / 180.0)) << "'");
                        RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Difference is good");
                        // RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Here after initial read and now setting position state '" << position_states_[joint_index] << "'");
                    }
                    
                    if (abs(velocity_states_[joint_index] - vel) > 0.2){
                      velocity_states_[joint_index] = velocity_states_[joint_index];
                    }
                    else{
                        velocity_states_[joint_index] = vel * M_PI / 180.0;
                        
                    }

                    if (abs(effort_states_[joint_index] - eff) > 2){
                      effort_states_[joint_index] = effort_states_[joint_index];
                    }
                    else{
                      effort_states_[joint_index] = effort_states_[joint_index] = eff * 0.88;
                      RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "effort after torque constant: '" << effort_states_[joint_index] << "'");
                    }

                }
                else
                {
                    // RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Here before initial read and now setting initial position state '" << init_position_states_[joint_index] << "'");
                    init_position_states_[joint_index] = pos;
                    // RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Here before initial read and after setting initial position state '" << init_position_states_[joint_index] << "'");

                }
                // RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Here after initial read and now setting position state '" << position_states_[joint_index] << "'");

                velocity_states_[joint_index] = vel * M_PI / 180.0; // Converting dps to rad/s
                effort_states_[joint_index] = eff * 0.88;
                prev_position_states_[joint_index] = position_states_[joint_index];
            }
            catch (const std::exception &e)
            {
                RCLCPP_WARN(rclcpp::get_logger("DaadbotInterface"), "Invalid token format: '%s'", token.c_str());
                return hardware_interface::return_type::ERROR;
            }

            joint_index++;
        }
    }
    catch (const std::exception &e)
    {
        RCLCPP_ERROR_STREAM(rclcpp::get_logger("DaadbotInterface"), "Read failed: " << e.what());
        return hardware_interface::return_type::ERROR;
    }

    if (!initial_read_)
    {
        initial_read_ = true;
        RCLCPP_INFO(rclcpp::get_logger("DaadbotInterface"), "Initial read completed.");
    }

    return hardware_interface::return_type::OK;
}


hardware_interface::return_type DaadbotInterface::write(const rclcpp::Time &time,
                                                        const rclcpp::Duration &period)
{
  if (!esp_.IsOpen())
  {
    RCLCPP_ERROR(rclcpp::get_logger("DaadbotInterface"), "Serial port not open, cannot write.");
    return hardware_interface::return_type::ERROR;
  }

  std::ostringstream msg;
  msg << "<W";

  bool any_change = false;

  for (size_t i = 4; i <= 6; i++)
  {
    // Converting radians to degrees before sending to hardware (speeds in 0.01 dsp)
    float value_to_send = initial_read_ ? velocity_commands_[i] * (180.0f / M_PI) : 0.0f;

    // Change detection (only matters if initial_read_ is true)
    if (initial_read_ && velocity_commands_[i] != prev_velocity_commands_[i])
    {
      any_change = true;
    }

    msg << std::fixed << std::setprecision(2) << " " << value_to_send;
    prev_velocity_commands_[i] = value_to_send;
  }

  msg << ">";
  std::string command = msg.str();

  //Skip serial write if nothing changed (only relevant during initial_read_)
  // if (initial_read_ && !any_change)
  // {
  //   return hardware_interface::return_type::OK;
  // }

  try
  {
    esp_.Write(command + "\n");
    esp_.DrainWriteBuffer();
    RCLCPP_INFO_STREAM(rclcpp::get_logger("DaadbotInterface"), "Sent: " << command);
  }
  catch (const std::exception &e)
  {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger("DaadbotInterface"), "Write failed: " << e.what());
    return hardware_interface::return_type::ERROR;
  }

  return hardware_interface::return_type::OK;
}

}  // namespace daadbot_controller

PLUGINLIB_EXPORT_CLASS(daadbot_controller::DaadbotInterface, hardware_interface::SystemInterface)