#include <rclcpp/rclcpp.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>
#include <vector>
#include <cmath>
#include <chrono>

class SequenceNode : public rclcpp::Node
{
public:
    SequenceNode()
        : Node("sequence_node"),
          current_angles_rad_(7, 0.0),
          increment_rad_(0.3 * M_PI / 180.0), // 0.3° in radians
          state_(0), step_done_(false), waiting_(false), config_transition_(false)
    {
        traj_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
            "/position_arm_controller/joint_trajectory", 10);

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10),
            std::bind(&SequenceNode::timer_callback, this));

        // Define sequences in degrees
        home_ = {0, 0, 0, 0, 0, 0, 0};
        pick_ = {0, -51, 0, -48, 90, 90, 0};
        place_ = {0, -51, 20, -48, 90, -90, 0};

        // Start from home
        target_angles_rad_ = deg_to_rad(home_);
    }

private:
    void timer_callback()
    {
        auto now = this->now();

        // Handle waiting logic
        if (waiting_ && now < wait_until_)
            return;
        waiting_ = false;

        if (!step_done_)
        {
            bool all_reached = true;
            for (size_t i = 0; i < 7; i++)
            {
                if (std::fabs(target_angles_rad_[i] - current_angles_rad_[i]) >= increment_rad_)
                {
                    all_reached = false;
                    if (current_angles_rad_[i] < target_angles_rad_[i])
                        current_angles_rad_[i] += increment_rad_;
                    else
                        current_angles_rad_[i] -= increment_rad_;
                }
                else
                {
                    current_angles_rad_[i] = target_angles_rad_[i];
                }
            }

            publish_angles(current_angles_rad_);

            if (all_reached)
            {
                step_done_ = true;
                if (config_transition_)
                {
                    RCLCPP_INFO(this->get_logger(), "Config complete. Waiting 2.5s...");
                    wait_until_ = now + rclcpp::Duration::from_seconds(2.5);
                }
                else
                {
                    RCLCPP_INFO(this->get_logger(), "Joint move complete. Waiting 0.75s...");
                    wait_until_ = now + rclcpp::Duration::from_seconds(0.75);
                }
                waiting_ = true;
            }
        }
        else
        {
            // Move to next motion
            step_done_ = false;
            advance_sequence();
        }
    }

    void advance_sequence()
    {
        config_transition_ = false; // default is intra-config move

        switch (state_)
        {
        case 0: // go home -> pick (joint 2)
            target_angles_rad_ = current_angles_rad_;
            target_angles_rad_[1] = pick_[1] * M_PI / 180.0;
            state_++;
            break;

        case 1: // pick (joint 4)
            target_angles_rad_ = current_angles_rad_;
            target_angles_rad_[3] = pick_[3] * M_PI / 180.0;
            state_++;
            break;

        case 2: // pick (joint 5)
            target_angles_rad_ = current_angles_rad_;
            target_angles_rad_[4] = pick_[4] * M_PI / 180.0;
            state_++;
            break;

        case 3: // pick (joint 6)
            target_angles_rad_ = current_angles_rad_;
            target_angles_rad_[5] = pick_[5] * M_PI / 180.0;
            state_++;
            config_transition_ = true; // pick finished
            break;

        case 4: // place (joint 3 = 20 deg)
            target_angles_rad_ = current_angles_rad_;
            target_angles_rad_[2] = place_[2] * M_PI / 180.0;
            state_++;
            break;

        case 5: // place (joint 6 turn -180 from +90 to -90)
            target_angles_rad_ = current_angles_rad_;
            target_angles_rad_[5] = place_[5] * M_PI / 180.0;
            state_++;
            config_transition_ = true; // place finished
            break;

        case 6: // return home: joint 6
            target_angles_rad_ = current_angles_rad_;
            target_angles_rad_[5] = 0.0;
            state_++;
            break;

        case 7: // return home: joint 5
            target_angles_rad_ = current_angles_rad_;
            target_angles_rad_[4] = 0.0;
            state_++;
            break;

        case 8: // return home: joint 4
            target_angles_rad_ = current_angles_rad_;
            target_angles_rad_[3] = 0.0;
            state_++;
            break;

        case 9: // return home: joint 3
            target_angles_rad_ = current_angles_rad_;
            target_angles_rad_[2] = 0.0;
            state_++;
            break;

        case 10: // return home: joint 2
            target_angles_rad_ = current_angles_rad_;
            target_angles_rad_[1] = 0.0;
            state_++;
            config_transition_ = true; // back home finished
            break;

        default:
            RCLCPP_INFO(this->get_logger(), "Sequence complete!");
            rclcpp::shutdown();
            break;
        }
    }


    void publish_angles(const std::vector<double> &angles_rad)
    {
        trajectory_msgs::msg::JointTrajectory traj_msg;
        traj_msg.joint_names = {
            "joint_1", "joint_2", "joint_3", "joint_4",
            "joint_5", "joint_6", "joint_7"};

        trajectory_msgs::msg::JointTrajectoryPoint point;
        point.positions = angles_rad;
        point.time_from_start = rclcpp::Duration::from_seconds(0.01);

        traj_msg.points.push_back(point);
        traj_pub_->publish(traj_msg);
    }

    std::vector<double> deg_to_rad(const std::vector<double> &deg_vec)
    {
        std::vector<double> rad_vec;
        for (double d : deg_vec)
            rad_vec.push_back(d * M_PI / 180.0);
        return rad_vec;
    }

    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr traj_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::vector<double> current_angles_rad_;
    std::vector<double> target_angles_rad_;

    std::vector<double> home_, pick_, place_;

    double increment_rad_;
    int state_;
    bool step_done_;

    // Waiting logic
    bool waiting_;
    rclcpp::Time wait_until_;
    bool config_transition_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SequenceNode>());
    rclcpp::shutdown();
    return 0;
}
