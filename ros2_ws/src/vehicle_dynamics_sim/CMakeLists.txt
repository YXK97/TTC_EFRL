cmake_minimum_required(VERSION 3.8)
project(vehicle_dynamics_sim)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(action_msgs REQUIRED)
find_package(rclpy REQUIRED)
# find_package(<dependency> REQUIRED)

set(msg_files
  "msg/SingleObjectState.msg"
  "msg/ObjectState.msg"
  "msg/AgentControl.msg"
  "msg/StateEval.msg"
  "msg/SingleAgentControl.msg"
)

set(action_files
  "action/AgentAction.action"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${action_files}
  DEPENDENCIES std_msgs action_msgs
)

ament_python_install_package(defmarl DESTINATION ${PYTHON_INSTALL_DIR})

ament_python_install_package(dpax DESTINATION ${PYTHON_INSTALL_DIR})

ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS
  scripts/start_env_node.py
  scripts/start_action_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# 确保你的附加文件被安装到 ROS2 标准 share 目录
install(DIRECTORY
  msg
  action
  launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
