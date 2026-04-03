#!/bin/bash
# ~/seabird/scripts/kill_seabird.sh
# Kills all Seabird-related processes EXCEPT Isaac Sim itself.
# Usage: bash ~/seabird/scripts/kill_seabird.sh

RED='\033[0;31m'
GRN='\033[0;32m'
YEL='\033[0;33m'
RST='\033[0m'

echo -e "${YEL}[seabird] Killing all Seabird processes...${RST}"

# --- PX4 SITL ---
if pkill -9 -f "px4" 2>/dev/null; then
    echo -e "${RED}  killed: PX4 SITL${RST}"
else
    echo -e "${GRN}  clean:  PX4 SITL (not running)${RST}"
fi

# --- MAVSDK-based scripts (takeoff_test, sweep_and_detect, etc.) ---
if pkill -9 -f "mavsdk" 2>/dev/null; then
    echo -e "${RED}  killed: MAVSDK process${RST}"
else
    echo -e "${GRN}  clean:  MAVSDK (not running)${RST}"
fi

# --- Any python running seabird scripts ---
if pkill -9 -f "seabird/scripts" 2>/dev/null; then
    echo -e "${RED}  killed: seabird Python scripts${RST}"
else
    echo -e "${GRN}  clean:  seabird scripts (not running)${RST}"
fi

# --- ROS2 nodes we own ---
if pkill -9 -f "sim_camera\|buoy_detector\|sweep_and_detect\|record_training" 2>/dev/null; then
    echo -e "${RED}  killed: ROS2 seabird nodes${RST}"
else
    echo -e "${GRN}  clean:  ROS2 seabird nodes (not running)${RST}"
fi

# --- Stale microXRCE / DDS agent (PX4↔ROS2 bridge) ---
if pkill -9 -f "MicroXRCEAgent\|micro_ros_agent" 2>/dev/null; then
    echo -e "${RED}  killed: micro-XRCE agent${RST}"
else
    echo -e "${GRN}  clean:  micro-XRCE agent (not running)${RST}"
fi

echo ""
echo -e "${YEL}[seabird] Done. Isaac Sim left untouched.${RST}"
echo -e "${YEL}[seabird] NOTE: labeler runs INSIDE Isaac — stop it with:${RST}"
echo -e "${YEL}          _labeler_sub = None   (in Script Editor)${RST}"
