import tkinter as tk
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import threading


class AngleGuiPublisher(Node):
    def __init__(self):
        super().__init__('angle_gui_publisher')
        self.pub = self.create_publisher(Float64MultiArray, '/target_joint_angles_deg', 10)


def main():
    rclpy.init()
    node = AngleGuiPublisher()

    root = tk.Tk()
    root.title("Joint Angle Input (Degrees)")

    sliders = []
    entries = []

    def update_from_slider(i, val):
        """Update entry when slider moves"""
        entries[i].delete(0, tk.END)
        entries[i].insert(0, f"{float(val):.1f}")

    def update_from_entry(i):
        """Update slider when entry changes"""
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

    def send_angles():
        values = [sliders[i].get() for i in range(7)]
        msg = Float64MultiArray()
        msg.data = values
        node.pub.publish(msg)
        print("Published:", values)

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

    tk.Button(root, text="Send All", command=send_angles).grid(row=8, column=0, columnspan=4)

    # Run ROS spin in background
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        root.mainloop()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
