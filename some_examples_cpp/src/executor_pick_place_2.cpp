
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>

using namespace std::chrono_literals;

class ConfigExecutorNode : public rclcpp::Node
{
public:
    ConfigExecutorNode()
        : Node("config_executor_node"),
          current_mode_("manual"),
          target_config_(""),
          current_angles_(7, 0.0),
          last_target_(7, 0.0),
          last_state_(7, 0.0)
    {
        sub_manual_ = create_subscription<std_msgs::msg::Float64MultiArray>(
            "/manual_joint_angles_deg", 10,
            [this](std_msgs::msg::Float64MultiArray::SharedPtr msg) {
                manual_angles_ = msg->data;
                if (current_mode_ == "manual")
                {
                    publish_target(manual_angles_);
                }
            });

        sub_config_ = create_subscription<std_msgs::msg::String>(
            "/target_config", 10,
            [this](std_msgs::msg::String::SharedPtr msg) {
                target_config_ = msg->data;
                if (current_mode_ == "config")
                {
                    execute_config(target_config_);
                }
            });

        sub_mode_ = create_subscription<std_msgs::msg::String>(
            "/control_mode", 10,
            [this](std_msgs::msg::String::SharedPtr msg) {
                current_mode_ = msg->data;
                RCLCPP_INFO(get_logger(), "Control mode switched to: %s", current_mode_.c_str());
            });

        // Sub to joint states (feedback from driver)
        sub_joint_states_ = create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10,
            [this](sensor_msgs::msg::JointState::SharedPtr msg) {
                // Assuming msg->position has 7 entries (joint_1..joint_7)
                if (msg->position.size() >= 7)
                {
                    for (size_t i = 0; i < 7; i++)
                    {
                        // Convert radians → degrees for error calculation
                        last_state_[i] = msg->position[i] * 180.0 / M_PI;
                    }
                    print_error();
                }
            });

        // Sub to target angles (for error comparison)
        sub_target_echo_ = create_subscription<std_msgs::msg::Float64MultiArray>(
            "/target_joint_angles_deg", 10,
            [this](std_msgs::msg::Float64MultiArray::SharedPtr msg) {
                if (msg->data.size() == 7)
                {
                    last_target_ = msg->data;
                }
            });

        pub_target_ = create_publisher<std_msgs::msg::Float64MultiArray>(
            "/target_joint_angles_deg", 10);

        // Define configs
        home_  = {0, 0, 0, 0, 0, 0, 0};
        pick_ = {0, -90, -30, -70, -50, 85, -120};
        place_  = {0, -45, 30, -45, 70, 75, 5};
    }

private:
    void execute_config(const std::string &config)
    {
        std::vector<double> target;
        if (config == "home")
        {
            target = home_;
            sequential_update(target, {4, 2, 1, 3, 6, 5}); // joints 6→2        5, 3, 2, 4, 7, 6 
            
        }
        else if (config == "pick")
        {
            target = pick_;
            sequential_update(target, {5, 3, 1, 4, 6, 2}); // joints 2,3,4,5
        }
        else if (config == "place")
        {
            target = place_;
            sequential_update(target, {5, 2, 1, 3, 4, 6}); // joints 3,6
        }
        else
        {
            RCLCPP_WARN(get_logger(), "Unknown config: %s", config.c_str());
        }
    }

    void sequential_update(const std::vector<double> &target, const std::vector<int> &order)
    {
        auto step = [this](const std::vector<double> &angles) {
            std_msgs::msg::Float64MultiArray msg;
            msg.data = angles;
            pub_target_->publish(msg);
            rclcpp::sleep_for(2000ms); // small delay between joints
        };

        std::vector<double> intermediate = current_angles_;

        for (int idx : order)
        {
            intermediate[idx] = target[idx]; // only update this joint
            step(intermediate);
        }

        current_angles_ = target; // update after full sequence
    }

    void publish_target(const std::vector<double> &angles)
    {
        std_msgs::msg::Float64MultiArray msg;
        msg.data = angles;
        pub_target_->publish(msg);
        current_angles_ = angles;
    }

    void print_error()
    {
        if (last_target_.size() != 7 || last_state_.size() != 7)
            return;

        std::ostringstream oss;
        oss << "Joint errors (deg): ";
        for (size_t i = 0; i < 7; i++)
        {
            double err = last_target_[i] - last_state_[i];
            oss << "J" << (i+1) << ": " << err << "  ";
        }
        RCLCPP_INFO(get_logger(), "%s", oss.str().c_str());
    }

    // ROS interfaces
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_manual_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_config_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_mode_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_target_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_joint_states_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_target_echo_;

    // State
    std::string current_mode_;
    std::string target_config_;
    std::vector<double> current_angles_;
    std::vector<double> manual_angles_;

    std::vector<double> last_target_; // latest commanded
    std::vector<double> last_state_;  // latest feedback

    // Configs
    std::vector<double> home_, pick_, place_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ConfigExecutorNode>());
    rclcpp::shutdown();
    return 0;
}







// #include <rclcpp/rclcpp.hpp>
// #include <std_msgs/msg/string.hpp>
// #include <std_msgs/msg/float64_multi_array.hpp>
// #include <sensor_msgs/msg/joint_state.hpp>
// #include <vector>
// #include <string>
// #include <chrono>
// #include <cmath>
// #include <sstream>

// using namespace std::chrono_literals;

// class ConfigExecutorNode : public rclcpp::Node
// {
// public:
//     ConfigExecutorNode()
//         : Node("config_executor_node"),
//           current_mode_("manual"),
//           target_config_(""),
//           current_angles_(7, 0.0),
//           last_target_(7, 0.0),
//           last_state_(7, 0.0)
//     {
//         // Create reentrant callback groups so callbacks can run concurrently
//         joint_state_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
//         config_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
//         manual_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

//         // Subscription options where we assign callback groups
//         rclcpp::SubscriptionOptions joint_opts;
//         joint_opts.callback_group = joint_state_cb_group_;

//         rclcpp::SubscriptionOptions config_opts;
//         config_opts.callback_group = config_cb_group_;

//         rclcpp::SubscriptionOptions manual_opts;
//         manual_opts.callback_group = manual_cb_group_;

//         // Manual angles subscription (keeps working in manual mode)
//         sub_manual_ = create_subscription<std_msgs::msg::Float64MultiArray>(
//             "/manual_joint_angles_deg", 10,
//             [this](std_msgs::msg::Float64MultiArray::SharedPtr msg) {
//                 manual_angles_ = msg->data;
//                 if (current_mode_ == "manual")
//                 {
//                     publish_target(manual_angles_);
//                 }
//             }, manual_opts);

//         // Config subscription (triggers sequential execution)
//         sub_config_ = create_subscription<std_msgs::msg::String>(
//             "/target_config", 10,
//             [this](std_msgs::msg::String::SharedPtr msg) {
//                 target_config_ = msg->data;
//                 RCLCPP_INFO(get_logger(), "Received config: %s", target_config_.c_str());
//                 if (current_mode_ == "config")
//                 {
//                     execute_config(target_config_);
//                 }
//             }, config_opts);

//         // Mode subscription
//         sub_mode_ = create_subscription<std_msgs::msg::String>(
//             "/control_mode", 10,
//             [this](std_msgs::msg::String::SharedPtr msg) {
//                 current_mode_ = msg->data;
//                 RCLCPP_INFO(get_logger(), "Control mode switched to: %s", current_mode_.c_str());
//             });

//         // Joint states subscription — put this in its own reentrant group so it can run while sequential_update is running
//         sub_joint_states_ = create_subscription<sensor_msgs::msg::JointState>(
//             "/joint_states", 50,
//             [this](sensor_msgs::msg::JointState::SharedPtr msg) {
//                 // debug print raw length and first few values (optional)
//                 if (msg->position.size() >= 7)
//                 {
//                     for (size_t i = 0; i < 7; ++i)
//                     {
//                         last_state_[i] = msg->position[i] * 180.0 / M_PI;
//                     }
//                     // Keep a copy of the whole message count if you want:
//                     last_joint_msg_size_ = msg->position.size();
//                     print_error();
//                     print_states();
//                 } else {
//                     RCLCPP_WARN_THROTTLE(get_logger(), *this->get_clock(), 2000, "joint_states has fewer than 7 positions: size=%zu", msg->position.size());
//                 }
//             }, joint_opts);

//         // Echo of target angles (optional)
//         sub_target_echo_ = create_subscription<std_msgs::msg::Float64MultiArray>(
//             "/target_joint_angles_deg", 10,
//             [this](std_msgs::msg::Float64MultiArray::SharedPtr msg) {
//                 if (msg->data.size() == 7)
//                 {
//                     last_target_ = msg->data;
//                 }
//             });

//         pub_target_ = create_publisher<std_msgs::msg::Float64MultiArray>(
//             "/target_joint_angles_deg", 10);

//         // Define configs
//         home_  = {0, 0, 0, 0, 0, 0, 0};
//         place_ = {0, -45, 30, -45, 65, 75, -170};
//         pick_  = {0, -45, -30, -45, -60, 60, -10};
//     }

// private:
//     void execute_config(const std::string &config)
//     {
//         std::vector<double> target;
//         if (config == "home")
//         {
//             target = home_;
//             sequential_update(target, {4, 2, 6, 5, 1, 3});
//         }
//         else if (config == "pick")
//         {
//             target = pick_;
//             sequential_update(target, {3, 1, 4, 5, 6, 2});
//         }
//         else if (config == "place")
//         {
//             target = place_;
//             // sequential_update(target, {2, 4, 5, 6});
//             sequential_update(target, {3, 1, 2, 4, 5, 6});
//         }
//         else
//         {
//             RCLCPP_WARN(get_logger(), "Unknown config: %s", config.c_str());
//         }
//     }

//     // Sequential update: debug prints joint_states repeatedly while waiting for error < threshold
//     void sequential_update(const std::vector<double> &target, const std::vector<int> &order)
//     {
//         std::vector<double> intermediate = current_angles_;

//         for (int idx : order)
//         {
//             // bounds check
//             if (idx < 0 || idx >= static_cast<int>(intermediate.size()))
//             {
//                 RCLCPP_WARN(get_logger(), "Skipping invalid joint index %d", idx);
//                 continue;
//             }

//             intermediate[idx] = target[idx]; // set new target for this joint
//             publish_target(intermediate);

//             RCLCPP_INFO(get_logger(), "Starting move for joint %d -> target %.2f deg", idx+1, target[idx]);

//             // Wait until the error is below threshold. During this loop, joint_states callbacks can run concurrently
//             while (rclcpp::ok())
//             {
//                 // Print joint states directly here (debug)
//                 std::ostringstream oss;
//                 oss << "[SEQ DEBUG] Joint states (deg): ";
//                 for (size_t i = 0; i < 7; i++)
//                 {
//                     oss << "J" << (i+1) << ": " << last_state_[i] << "  ";
//                 }
//                 oss << " | joint_msg_size: " << last_joint_msg_size_;
//                 RCLCPP_INFO(get_logger(), "%s", oss.str().c_str());

//                 double err = std::fabs(last_target_[idx] - last_state_[idx]);
//                 RCLCPP_INFO(get_logger(), "[SEQ DEBUG] Joint %d error = %.3f deg", idx+1, err);

//                 if (err < 0.75)
//                 {
//                     RCLCPP_INFO(get_logger(), "Joint %d reached target (err=%.3f deg)", idx+1, err);
//                     rclcpp::sleep_for(400ms); // extra wait before moving to next joint
//                     break;
//                 }

//                 // Small sleep so we don't spam logs too fast
//                 rclcpp::sleep_for(200ms);
//             }
//         }

//         current_angles_ = target;
//     }

//     void publish_target(const std::vector<double> &angles)
//     {
//         std_msgs::msg::Float64MultiArray msg;
//         msg.data = angles;
//         pub_target_->publish(msg);
//         current_angles_ = angles;
//         last_target_ = angles;
//     }

//     void print_error()
//     {
//         if (last_target_.size() != 7 || last_state_.size() != 7)
//             return;

//         std::ostringstream oss;
//         oss << "Joint errors (deg): ";
//         for (size_t i = 0; i < 7; i++)
//         {
//             double err = last_target_[i] - last_state_[i];
//             oss << "J" << (i+1) << ": " << err << "  ";
//         }
//         RCLCPP_INFO(get_logger(), "%s", oss.str().c_str());
//     }

//     void print_states()
//     {
//         std::ostringstream oss;
//         oss << "Joint states (deg): ";
//         for (size_t i = 0; i < 7; i++)
//         {
//             oss << "J" << (i+1) << ": " << last_state_[i] << "  ";
//         }
//         RCLCPP_INFO(get_logger(), "%s", oss.str().c_str());
//     }

//     // ROS interfaces
//     rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_manual_;
//     rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_config_;
//     rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_mode_;
//     rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_target_;
//     rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_joint_states_;
//     rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_target_echo_;

//     // Callback groups to allow concurrent callbacks
//     rclcpp::CallbackGroup::SharedPtr joint_state_cb_group_;
//     rclcpp::CallbackGroup::SharedPtr config_cb_group_;
//     rclcpp::CallbackGroup::SharedPtr manual_cb_group_;

//     // State
//     std::string current_mode_;
//     std::string target_config_;
//     std::vector<double> current_angles_;
//     std::vector<double> manual_angles_;

//     std::vector<double> last_target_;
//     std::vector<double> last_state_;
//     size_t last_joint_msg_size_{0};

//     // Configs
//     std::vector<double> home_, pick_, place_;
// };

// int main(int argc, char **argv)
// {
//     rclcpp::init(argc, argv);
//     auto node = std::make_shared<ConfigExecutorNode>();

//     // Multi-threaded executor so reentrant callback groups can run concurrently
//     rclcpp::executors::MultiThreadedExecutor exec;
//     exec.add_node(node);
//     exec.spin();

//     rclcpp::shutdown();
//     return 0;
// }












