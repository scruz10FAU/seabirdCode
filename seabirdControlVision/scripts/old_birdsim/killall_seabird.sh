#!/bin/bash
echo "[kill] Stopping all Seabird processes..."
pkill -9 -f sweep_and_detect
pkill -9 -f buoy_detector
pkill -9 -f flyover_controller
pkill -9 -f mavsdk_server
pkill -9 -f mavlink_shell
pkill -9 -f px4
pkill -9 -f rqt_image_view
pkill -9 -f mavsdk
sleep 2
echo "[kill] Clearing shared memory ports..."
sudo rm -rf /dev/shm/fastrtps_*
sudo rm -rf /dev/shm/sem.fastrtps*
echo "[kill] Done — all clear"
