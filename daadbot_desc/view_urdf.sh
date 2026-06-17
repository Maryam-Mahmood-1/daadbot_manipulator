#!/usr/bin/env bash
# Browse and visualize the "main" URDF of each folder under urdf/ in RViz.
# No colcon build needed - runs xacro + RViz nodes directly from source.
#
# Usage:
#   ./view_urdf.sh            # interactive numbered menu
#   ./view_urdf.sh <number>   # launch that entry directly
#   ./view_urdf.sh <path>     # launch a specific urdf/xacro by path

set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
URDF_DIR="$PKG_DIR/urdf"
RVIZ_CFG="$PKG_DIR/rviz/display.rviz"   # fallback handled below

# Pick the main file in a folder: prefer daadbot.urdf.xacro, then any
# *.urdf.xacro, then any plain *.urdf.
pick_main() {
    local dir="$1" f
    if [[ -f "$dir/daadbot.urdf.xacro" ]]; then
        echo "$dir/daadbot.urdf.xacro"; return
    fi
    f=$(find "$dir" -maxdepth 1 -name '*.urdf.xacro' \
        ! -name '*gazebo*' ! -name '*ros2_control*' ! -name '*table*' \
        ! -name '*bottle*' | sort | head -n1)
    [[ -n "$f" ]] && { echo "$f"; return; }
    f=$(find "$dir" -maxdepth 1 -name '*.urdf' | sort | head -n1)
    [[ -n "$f" ]] && { echo "$f"; return; }
    echo ""
}

# Build the list of (folder -> main file).
mapfile -t FOLDERS < <(find "$URDF_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
declare -a ENTRIES=()
for d in "${FOLDERS[@]}"; do
    main="$(pick_main "$d")"
    [[ -n "$main" ]] && ENTRIES+=("$main")
done

launch() {
    local model="$1"
    [[ -f "$model" ]] || { echo "File not found: $model" >&2; exit 1; }
    local cfg="$RVIZ_CFG"
    [[ -f "$cfg" ]] || cfg="$PKG_DIR/rviz/camera_bot.rviz"

    echo ">> Visualizing: $model"
    echo ">> RViz config: $cfg"
    local robot_desc
    robot_desc="$(xacro "$model")"

    # Launch nodes; Ctrl-C cleans them all up.
    trap 'kill 0' EXIT INT TERM
    ros2 run robot_state_publisher robot_state_publisher \
        --ros-args -p robot_description:="$robot_desc" &
    ros2 run joint_state_publisher_gui joint_state_publisher_gui &
    ros2 run rviz2 rviz2 -d "$cfg"
}

# Direct launch by path or index.
if [[ $# -ge 1 ]]; then
    if [[ -f "$1" ]]; then launch "$1"; exit 0; fi
    if [[ "$1" =~ ^[0-9]+$ ]]; then
        idx=$(( $1 - 1 ))
        [[ $idx -ge 0 && $idx -lt ${#ENTRIES[@]} ]] || { echo "Bad index" >&2; exit 1; }
        launch "${ENTRIES[$idx]}"; exit 0
    fi
    echo "Unknown argument: $1" >&2; exit 1
fi

# Interactive menu.
echo "Main URDF per folder under urdf/:"
echo
i=1
for e in "${ENTRIES[@]}"; do
    printf "  %2d) %-26s -> %s\n" "$i" "$(basename "$(dirname "$e")")" "$(basename "$e")"
    i=$((i+1))
done
echo
read -rp "Pick a number to visualize (q to quit): " choice
[[ "$choice" == "q" ]] && exit 0
[[ "$choice" =~ ^[0-9]+$ ]] || { echo "Not a number." >&2; exit 1; }
idx=$(( choice - 1 ))
[[ $idx -ge 0 && $idx -lt ${#ENTRIES[@]} ]] || { echo "Out of range." >&2; exit 1; }
launch "${ENTRIES[$idx]}"
