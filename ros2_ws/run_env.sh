#!/bin/bash
source /home/yxk-vtd/miniforge3/etc/profile.d/conda.sh
conda activate ttc_efrl
source /opt/ros/galactic/setup.bash
cd /home/yxk-vtd/TTC_EFRL/ros2_ws
source install/setup.bash

export ROS_PYTHON_VERSION=3.8

export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export CUDA_VISIBLE_DEVICES=""
export XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "✅ Env 节点已禁用 GPU，强制使用 CPU 运行"


ros2 launch vehicle_dynamics_sim env_launch.py "$@"
