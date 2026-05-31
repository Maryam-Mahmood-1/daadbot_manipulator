import tkinter as tk
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
import threading


class AngleGuiPublisher(Node):
    def __init__(self):
        super().__init__('angle_gui_publisher')
        # Publishers
        self.manual_pub = self.create_publisher(Float64MultiArray, '/manual_joint_angles_deg', 10)
        self.config_pub = self.create_publisher(String, '/target_config', 10)
        self.mode_pub = self.create_publisher(String, '/control_mode', 10)
        self.gripper_pub = self.create_publisher(String, '/gripper_command', 10)  # NEW


def main():
    rclpy.init()
    node = AngleGuiPublisher()

    root = tk.Tk()
    root.title("Joint Angle Input (Degrees)")

    sliders = []
    entries = []

    # Define configs (in degrees)
    configs = {
        "home":  [0, 0, 0, 0, 0, 0, 0],
        "pick":  [0, -45, -30, -45, -60, 60, -10],
        "place": [0, -45, 30, -45, 65, 75, -170],
    }

    # Current mode flag
    mode_var = tk.StringVar(value="manual")  # "manual" or "config"

    def update_from_slider(i, val):
        entries[i].delete(0, tk.END)
        entries[i].insert(0, f"{float(val):.1f}")

    def update_from_entry(i):
        try:
            val = float(entries[i].get())
            if -180 <= val <= 180:
                sliders[i].set(val)
            else:
                entries[i].delete(0, tk.END)
                entries[i].insert(0, f"{sliders[i].get():.1f}")
        except ValueError:
            entries[i].delete(0, tk.END)
            entries[i].insert(0, f"{sliders[i].get():.1f}")

    def publish_mode(mode_name: str):
        msg_mode = String()
        msg_mode.data = mode_name
        node.mode_pub.publish(msg_mode)
        mode_var.set(mode_name)
        print("Published mode:", mode_name)

    def send_manual_angles():
        values = [sliders[i].get() for i in range(7)]
        msg = Float64MultiArray()
        msg.data = values
        node.manual_pub.publish(msg)
        publish_mode("manual")
        print("Published manual angles:", values)

    def set_config(config_name):
        if config_name not in configs:
            return

        values = configs[config_name]
        for i in range(7):
            sliders[i].set(values[i])
            entries[i].delete(0, tk.END)
            entries[i].insert(0, f"{values[i]:.1f}")

        msg_cfg = String()
        msg_cfg.data = config_name
        node.config_pub.publish(msg_cfg)
        publish_mode("config")
        print("Published config:", config_name)

    def send_gripper_command(cmd: str):
        msg = String()
        msg.data = cmd
        node.gripper_pub.publish(msg)
        print("Published gripper command:", cmd)

    # Create sliders + entries
    for i in range(7):
        tk.Label(root, text=f"Joint {i+1}").grid(row=i, column=0)

        slider = tk.Scale(
            root,
            from_=-180, to=180,
            orient=tk.HORIZONTAL,
            resolution=0.1,
            length=300,
            command=lambda val, i=i: update_from_slider(i, val)
        )
        slider.set(0.0)
        slider.grid(row=i, column=1)
        sliders.append(slider)

        entry = tk.Entry(root, width=8)
        entry.insert(0, "0.0")
        entry.grid(row=i, column=2)
        entries.append(entry)

        tk.Button(root, text="Set", command=lambda i=i: update_from_entry(i)).grid(row=i, column=3)

    # Manual send button
    tk.Button(root, text="Send All (Manual)", command=send_manual_angles).grid(row=8, column=0, columnspan=4, pady=10)

    # Config buttons
    tk.Label(root, text="Configs").grid(row=9, column=0, pady=(20, 5))
    tk.Button(root, text="Home", command=lambda: set_config("home")).grid(row=10, column=0, columnspan=4, sticky="we")
    tk.Button(root, text="Pick", command=lambda: set_config("pick")).grid(row=11, column=0, columnspan=4, sticky="we")
    tk.Button(root, text="Place", command=lambda: set_config("place")).grid(row=12, column=0, columnspan=4, sticky="we")

    # Gripper controls
    tk.Label(root, text="Gripper").grid(row=13, column=0, pady=(20, 5))
    tk.Button(root, text="Open", command=lambda: send_gripper_command("o")).grid(row=14, column=0, columnspan=2, sticky="we")
    tk.Button(root, text="Close", command=lambda: send_gripper_command("c")).grid(row=14, column=2, columnspan=2, sticky="we")

    # Mode indicator
    tk.Label(root, text="Current Mode:").grid(row=15, column=0, pady=(20, 0))
    tk.Label(root, textvariable=mode_var, fg="blue").grid(row=15, column=1, columnspan=3, sticky="w")

    # Run ROS spin in background
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        root.mainloop()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
