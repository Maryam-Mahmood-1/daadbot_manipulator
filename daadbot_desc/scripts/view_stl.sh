#!/usr/bin/env bash
# View a single STL mesh on its own in RViz (no full robot/urdf needed).
# Usage: ./view_stl.sh /path/to/mesh.STL
set -euo pipefail
STL="$(readlink -f "${1:?usage: view_stl.sh <mesh.STL>}")"
[ -f "$STL" ] || { echo "no such file: $STL" >&2; exit 1; }

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
URDF="$(mktemp /tmp/view_stl_XXXX.urdf)"
cat > "$URDF" <<EOF
<?xml version="1.0"?>
<robot name="stl_view">
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="file://$STL"/></geometry>
    </visual>
  </link>
</robot>
EOF

echo ">> viewing $STL"
trap 'kill 0' EXIT INT TERM
ros2 run robot_state_publisher robot_state_publisher \
  --ros-args -p robot_description:="$(cat "$URDF")" &
CFG="$PKG_DIR/rviz/display.rviz"; [ -f "$CFG" ] || CFG="$PKG_DIR/rviz/camera_bot.rviz"
ros2 run rviz2 rviz2 -d "$CFG"
