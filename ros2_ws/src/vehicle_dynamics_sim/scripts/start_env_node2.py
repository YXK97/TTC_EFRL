#!/usr/bin/env python3
import os
import rclpy
import time
import yaml
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from typing import Optional
from rclpy.node import Node
from rclpy.action import ActionClient

from defmarl.env import make_env
from defmarl.env.mve import MVE, MVEEnvGraphsTuple
from defmarl.utils.utils import parse_jax_array
from vehicle_dynamics_sim.action import AgentControl
from vehicle_dynamics_sim.msg import ObjectState, SingleObjectState, SingleAgentControl, AgentControl, ObjectEval


class EnvNode(Node):
    def __init__(self):
        super().__init__ ('start_env_node')

        # 必填参数（无默认值，后续在launch中强制传入）
        self.declare_parameter('path')  # 核心必填参数，无默认值
        # 可选参数（与原argparse默认值一致）
        self.declare_parameter('num_agents')
        self.declare_parameter('env')
        self.declare_parameter('full_observation', False)  # ROS2参数名不支持连字符，用下划线
        self.declare_parameter('cpu', False)
        self.declare_parameter('max_step')
        self.declare_parameter('seed', 1234)
        self.declare_parameter('debug', False)
        self.declare_parameter('area_size')  # jax数组先传字符串，再内部解析

        # 取所有参数（封装为类似原args的对象，方便后续调用
        self.args = self.get_all_parameters()

        # 校验必填参数（path不能为空）
        if not self.args.path:
            self.get_logger().fatal('Parameter "path" is required! Please set it in launch file.')
            rclpy.shutdown()
            raise SystemExit('Missing required parameter: path')

        n_gpu = jax.local_device_count()
        self.get_logger().info(f"> initializing EnvNode ...")
        self.get_logger().info(f"> Using {n_gpu} devices")

        # set up environment variables and seed
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        if self.args.cpu:
            os.environ["JAX_PLATFORM_NAME"] = "cpu"
        if self.args.debug:
            jax.config.update("jax_disable_jit", True)

        # load config
        if self.args.path is not None:
            with open(os.path.join(self.args.path, "config.yaml"), "r") as f:
                config = yaml.load(f, Loader=yaml.UnsafeLoader)

        # create environments
        num_agents = config.num_agents if self.args.num_agents is None else self.args.num_agents
        env = make_env(
            env_id=config.env if self.args.env is None else self.args.env,
            num_agents=num_agents,
            num_obs=config.obs,
            max_step=self.args.max_step,
            full_observation=self.args.full_observation,
            area_size=config.area_size if self.args.area_size is None else self.args.area_size,
            reward_min=config.reward_min,
            reward_max=config.reward_max
        )
        self.env = env

        # 初始化状态和步数
        np.random.seed(self.args.seed)
        key_0 = jr.PRNGKey(self.args.seed)
        self.current_graph, _ = self.env.reset(key_0)
        self.current_step = 0

        # ros通信
        # 发布环境状态（移除action/eval）
        self.state_pub = self.create_publisher(ObjectState, '/ros_env/state', 10)
        # 发布控制指令（action）
        self.action_pub = self.create_publisher(AgentControl, '/ros_env/action', 10)
        # 发布评估指标（reward/cost）
        self.eval_pub = self.create_publisher(ObjectEval, '/ros_env/eval', 10)
        # 创建Action客户端（向ros_action节点请求控制量）
        self.control_client = ActionClient(self, AgentControl, '/ros_action')
        # 定时器（初始启动，触发第一步逻辑）
        self.timer = self.create_timer(self.env.dt, self.step_callback)
        self.control_result = None  # 缓存 Action 响应结果
        self.default_action = np.zeros((self.env.num_agents, 2))  # 提前初始化默认控制
        # 运行标记 + 异步step标记（避免定时器重复触发）
        self.is_running = True
        self.is_waiting_for_action = False  # 标记是否正在等待action响应
        self.get_logger().info('Env node initialized, max step: %d, freq: %dHz' % (self.env.max_episode_steps, 1/self.env.dt))

    def get_all_parameters(self):
        """获取所有ROS2参数，封装为简单对象（模拟argparse的args）"""
        class EnvArgs:
            path:str
            num_agents:Optional[int]
            env:Optional[str]
            full_observation:bool
            cpu:bool
            max_step:Optional[int]
            seed:int
            debug:bool
            area_size:jnp.ndarray
        args = EnvArgs()
        # 赋值所有参数
        args.path = self.get_parameter('path').get_parameter_value().string_value
        args.num_agents = self.get_parameter('num_agents').get_parameter_value().integer_value
        args.env = self.get_parameter('env').get_parameter_value().string_value
        args.full_observation = self.get_parameter('full_observation').get_parameter_value().bool_value
        args.cpu = self.get_parameter('cpu').get_parameter_value().bool_value
        args.max_step = self.get_parameter('max_step').get_parameter_value().integer_value
        args.seed = self.get_parameter('seed').get_parameter_value().integer_value
        args.debug = self.get_parameter('debug').get_parameter_value().bool_value
        # 处理自定义jax数组（解析字符串为jax数组，复用原parse_jax_array）
        area_size_str = self.get_parameter('area_size').get_parameter_value().string_value
        args.area_size = parse_jax_array(area_size_str) if area_size_str else None
        # 处理None值（ROS2参数未指定时，integer_value/string_value会返回0/空字符串，修正为None）
        for param in ['num_agents', 'env', 'max_step']:
            val = getattr(args, param)
            if val == 0 or val == '':
                setattr(args, param, None)
        return args

    def step_callback(self):
        """核心修改：先发布state → 暂停定时器 → 请求action → 等action返回后执行step"""
        if not self.is_running or self.current_step >= self.env.max_episode_steps:
            self.terminate_sim()
            return

        # 1. 先暂停定时器，避免重复触发（等step执行完再重启）
        self.timer.cancel()
        self.is_waiting_for_action = True

        # 2. 发布当前最新的state（未step的原始state）给action_node
        # 此时还未执行step，发布的是当前graph的真实状态，action_node基于此计算action
        self.publish_state(
            graph=self.current_graph,
            reward=0.0,  # 第一步无reward，后续step后更新
            cost=0.0,
            cost_real=0.0,
            ad_action=self.default_action  # 首次发布用默认action，不影响
        )
        self.get_logger().info(f'Published current state (step {self.current_step}), waiting for action...')

        # 3. 发送action请求（基于刚发布的最新state）
        self.send_new_action_goal()

    def execute_step_with_action(self, action):
        """收到action后执行step，封装为独立函数"""
        if not self.is_running or self.current_step >= self.env.max_episode_steps:
            self.terminate_sim()
            return

        step_start = time.time()
        # 1. 执行动力学step（用最新收到的action，基于当前state）
        next_graph, dsYddts, reward, a_cost, a_cost_real, done, info = self.env.step(self.current_graph, action)
        reward = float(reward)
        cost = float(jnp.max(a_cost))
        cost_real = float(jnp.max(a_cost_real))

        # 2. 更新状态和步数
        self.current_step += 1
        self.current_graph = next_graph
        self.get_logger().info(f'Executed step {self.current_step}/{self.env.max_episode_steps}, reward: {reward:.2f}, cost: {cost:.2f}')

        # 3. 同步现实时间（补足时延）
        step_elapsed = time.time() - step_start
        if step_elapsed < self.env.dt:
            time.sleep(self.env.dt - step_elapsed)

        # 4. 重启定时器，触发下一轮step逻辑
        self.is_waiting_for_action = False
        self.timer = self.create_timer(self.env.dt, self.step_callback)

    def goal_response_callback(self, future):
        """Action Goal 响应回调（异步，无阻塞）"""
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().warn(f'Action goal rejected by server, use default control {self.default_action}')
                # 即使goal被拒绝，也用默认action执行step，避免仿真卡住
                self.execute_step_with_action(self.default_action)
                return

            # Goal 被接受，异步获取结果，注册结果回调
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self.result_callback)

        except Exception as e:
            self.get_logger().warn(f'Failed to get goal response: {e}, use default {self.default_action}')
            self.execute_step_with_action(self.default_action)

    def result_callback(self, future):
        """Action 结果回调：收到action后立即执行step（核心修改）"""
        try:
            result = future.result().result

            if result.success:
                ad_action_list = [[single_action.ax, single_action.delta] for single_action in result.ad_action]
                ad_action = jnp.array(ad_action_list, dtype=jnp.float32)
                # 验证控制量形状
                if ad_action.shape == (self.env.num_agents, 2):
                    self.get_logger().info(f'Success get multi-agent control (shape: {ad_action.shape}): {ad_action}')
                    # 用最新收到的action执行step
                    self.execute_step_with_action(ad_action)
                else:
                    self.get_logger().warn(
                        f'Control shape mismatch: {ad_action.shape} vs ({self.env.num_agents},2), use default {self.default_action}')
                    self.execute_step_with_action(self.default_action)
            else:
                self.get_logger().warn(f'Action result invalid (success={result.success}), use default {self.default_action}')
                self.execute_step_with_action(self.default_action)

        except Exception as e:
            self.get_logger().warn(f'Failed to get action result: {e}, use default {self.default_action}')
            self.execute_step_with_action(self.default_action)

    def send_new_action_goal(self):
        """封装异步发送 Action Goal 的逻辑（仅发送请求，不返回结果）"""
        # 等待action服务器上线
        if not self.control_client.wait_for_server(timeout_sec=1.):
            self.get_logger().warn('Action server not found, use default control {self.default_action}')
            self.execute_step_with_action(self.default_action)
            return
        # 构造空Goal，异步发送
        goal_msg = AgentControl.Goal()
        future = self.control_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

    def publish_state(self, graph: MVEEnvGraphsTuple):
        """发布自车+障碍车状态到/ros_env/state"""
        aS_agent_states = graph.type_states(type_idx=MVE.AGENT, n_type=self.env.num_agents)
        aS_goal_states = graph.type_states(type_idx=MVE.GOAL, n_type=self.env.num_agents)
        oS_obst_states = graph.type_states(type_idx=MVE.OBST, n_type=self.env.num_obsts)
        agent_states_np = np.asarray(aS_agent_states)
        goal_states_np = np.asarray(aS_goal_states)
        obst_states_np = np.asarray(oS_obst_states)
        msg = ObjectState()

        def _array_to_single_state(state_arr: np.ndarray) -> SingleObjectState:
            """将单个物体的状态数组（长度8）转为SingleObjectState msg"""
            single_state = SingleObjectState()
            single_state.x = float(state_arr[0])
            single_state.y = float(state_arr[1])
            single_state.vx = float(state_arr[2])
            single_state.vy = float(state_arr[3])
            single_state.theta = float(state_arr[4])
            single_state.dthetadt = float(state_arr[5])
            single_state.bw = float(state_arr[6])
            single_state.bh = float(state_arr[7])
            return single_state

        msg.as_agent_states = [_array_to_single_state(state) for state in agent_states_np]
        msg.as_goal_states = [_array_to_single_state(state) for state in goal_states_np]
        msg.os_obst_states = [_array_to_single_state(state) for state in obst_states_np]
        msg.state_dim = 8
        self.state_pub.publish(msg)


    def publish_action(self, ad_action: jnp.ndarray):
        """仅发布控制指令到 /ros_env/action"""
        msg = AgentControl()

        def _array_to_single_control(action_arr) -> SingleAgentControl:
            """将单个agent的动作数组（长度2）转为SingleAgentControl msg"""
            single_control = SingleAgentControl()
            single_control.ax = float(action_arr[0])
            single_control.delta = float(action_arr[1])
            return single_control

        msg.ad_action = [_array_to_single_control(d_action) for d_action in ad_action]
        msg.action_dim = 2
        self.action_pub.publish(msg)

    def publish_eval(self, reward: float, cost: float, cost_real: float):
        """仅发布评估指标到 /ros_env/eval"""
        msg = ObjectEval()
        msg.reward = reward
        msg.cost = cost
        msg.cost_real = cost_real
        self.eval_pub.publish(msg)


    def terminate_sim(self):
        """终止仿真，发布终止信号"""
        self.is_running = False
        if self.is_waiting_for_action:
            self.timer.cancel()
        # 发布终止信号（供action节点感知）
        terminate_state = ObjectState()
        self.state_pub.publish(terminate_state)
        self.get_logger().info(f'Simulation finished! Total steps: {self.current_step}')
        rclpy.shutdown()


def main():
    rclpy.init()
    env_node = EnvNode()
    try:
        rclpy.spin(env_node)
    finally:
        env_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()