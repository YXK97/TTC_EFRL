#!/bin/bash
# 适配Python 3.8的运行脚本：ROS2 Galactic + conda ttc_efrl（Python 3.8）
# 功能：激活conda环境 → 加载ROS2环境 → 启动launch文件 → 转发命令行参数

# 1. 激活conda的Python 3.8环境（替换为你的conda 3.8环境名，此处为defmarl_py38）
# 若conda路径不同，可通过`conda env list`查看并修改
source /home/yxk-vtd/miniforge3/etc/profile.d/conda.sh
conda activate ttc_efrl

# 2. 加载ROS2 Galactic系统环境（必须步骤，确保ros2命令可用）
source /opt/ros/galactic/setup.bash

# 3. 加载当前ROS2工作空间的编译环境（识别vehicle_dynamics_sim包）
cd /home/yxk-vtd/TTC_EFRL/ros2_ws
source install/setup.bash

# 4. 显式指定ROS2使用Python 3.8（可选，增强鲁棒性，避免意外切换版本）
export ROS_PYTHON_VERSION=3.8

# 5. 启动仿真launch文件，转发所有终端传入的参数（如--path、--num_agents）
# "$@"：保留所有命令行参数，实现参数传递
ros2 launch vehicle_dynamics_sim sim_launch.py "$@"

# 6. 运行完成后，退出conda环境，恢复终端原始状态（可选，避免污染全局环境）
conda deactivate
