#!/bin/bash
source /home/yxk-vtd/miniforge3/etc/profile.d/conda.sh
conda activate ttc_efrl
source /opt/ros/galactic/setup.bash
cd /home/yxk-vtd/TTC_EFRL/ros2_ws
source install/setup.bash

export ROS_PYTHON_VERSION=3.8

ros2 launch vehicle_dynamics_sim action_launch.py "$@"
