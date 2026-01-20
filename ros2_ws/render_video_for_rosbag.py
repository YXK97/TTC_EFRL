"""运行本程序之前应当先colcon build"""

import os
import sys
import numpy as np
import rclpy
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rosidl_runtime_py.utilities import get_message

# ===================== 核心：添加ROS2编译后的msg路径 =====================
# 你指定的目标路径
ROS2_MSG_PATH = "/home/yxk-vtd/TTC_EFRL/ros2_ws/install/vehicle_dynamics_sim/lib/python3.8/site-packages"

# 1. 检查路径是否存在
if not os.path.exists(ROS2_MSG_PATH):
    print(f"错误：指定的路径不存在 → {ROS2_MSG_PATH}")
    print("请检查路径拼写或确认colcon build已成功编译msg包")
    sys.exit(1)

# 2. 将路径添加到sys.path（确保Python能找到msg模块）
if ROS2_MSG_PATH not in sys.path:
    sys.path.append(ROS2_MSG_PATH)
    print(f"已将路径添加到sys.path：{ROS2_MSG_PATH}")

# 3. 导入自定义msg（请替换为你实际的msg模块名和类名）
# 说明：这里的导入路径对应你msg包的结构，例如：
# 如果你的msg文件是 vehicle_dynamics_sim/msg/State.msg
# 则导入路径为 from vehicle_dynamics_sim.msg import State
try:
    from vehicle_dynamics_sim.msg import State, Action, Eval

    print("✅ 自定义msg导入成功！")
except ImportError as e:
    print(f"❌ 自定义msg导入失败：{e}")
    print("请检查：")
    print("  1. msg包名是否正确（vehicle_dynamics_sim）")
    print("  2. msg类名是否正确（State/Action/Eval）")
    print("  3. colcon build是否成功编译了msg包")
    sys.exit(1)


# ===================== ROSBag分析核心逻辑 =====================
def analyze_rosbag(bag_path, target_topics):
    """
    分析rosbag文件，统计指定话题的消息数量和录制频率

    参数:
        bag_path: rosbag文件路径（.db3文件）
        target_topics: 需要分析的话题列表
    """
    # 初始化存储配置（指定bag文件路径）
    storage_options = StorageOptions(
        uri=bag_path,
        storage_id='sqlite3'  # ROS2 galactic默认的存储格式
    )
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')

    # 创建bag读取器
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    # 初始化统计数据结构
    topic_stats = {
        topic: {
            'count': 0,  # 消息数量
            'timestamps': [],  # 时间戳列表（纳秒）
            'messages': []  # 存储解析后的消息（可选）
        } for topic in target_topics
    }

    # 获取所有可用话题及其类型
    topic_types = reader.get_all_topics_and_types()
    topic_type_map = {t.name: t.type for t in topic_types}

    # 验证目标话题是否存在
    for topic in target_topics:
        if topic not in topic_type_map:
            print(f"⚠️ 警告：话题 {topic} 不存在于rosbag中！")
            continue

    # 遍历所有消息
    print(f"\n开始解析rosbag文件：{bag_path}")
    msg_total = 0  # 总消息计数
    while reader.has_next():
        try:
            # 读取消息：(话题名, 消息数据, 时间戳(纳秒))
            topic_name, data, timestamp = reader.read_next()
            msg_total += 1

            # 只处理目标话题
            if topic_name in target_topics:
                # 统计消息数量和时间戳
                topic_stats[topic_name]['count'] += 1
                topic_stats[topic_name]['timestamps'].append(timestamp)

                # （可选）解析并存储完整消息（如需查看具体数据可取消注释）
                # msg_type = get_message(topic_type_map[topic_name])
                # msg = deserialize_message(data, msg_type)
                # topic_stats[topic_name]['messages'].append(msg)
                # 示例：打印第一条state消息的内容
                # if topic_name == "/ros_env/state" and topic_stats[topic_name]['count'] == 1:
                #     print(f"\n📌 第一条/ros_env/state消息内容：{msg}")

        except Exception as e:
            print(f"\n❌ 读取消息时出错：{e}")
            continue

    # 计算并输出统计结果
    print("\n" + "=" * 50)
    print("📊 ROSBag 分析结果")
    print("=" * 50)
    print(f"总解析消息数：{msg_total}")
    for topic, stats in topic_stats.items():
        if stats['count'] == 0:
            print(f"\n{topic}:")
            print(f"  ├─ 消息数量：0")
            print(f"  └─ 录制频率：无（无消息）")
            continue

        # 转换时间戳为秒（ROS2时间戳单位是纳秒）
        timestamps_sec = np.array(stats['timestamps']) / 1e9
        total_duration = timestamps_sec[-1] - timestamps_sec[0]  # 总录制时长（秒）
        avg_frequency = stats['count'] / total_duration  # 平均频率（Hz）

        # 计算相邻消息的时间差（用于查看频率稳定性）
        time_diff = np.diff(timestamps_sec)
        avg_time_diff = np.mean(time_diff)  # 平均间隔（秒）
        min_time_diff = np.min(time_diff)  # 最小间隔（秒）
        max_time_diff = np.max(time_diff)  # 最大间隔（秒）

        print(f"\n{topic}:")
        print(f"  ├─ 消息数量：{stats['count']}")
        print(f"  ├─ 总录制时长：{total_duration:.2f} 秒")
        print(f"  ├─ 平均录制频率：{avg_frequency:.2f} Hz")
        print(f"  ├─ 消息平均间隔：{avg_time_diff:.4f} 秒")
        print(f"  ├─ 消息最小间隔：{min_time_diff:.4f} 秒")
        print(f"  └─ 消息最大间隔：{max_time_diff:.4f} 秒")


def render_video(
        self,
        rollout: Rollout,
        video_path: pathlib.Path,
        Ta_is_unsafe=None,
        viz_opts: Optional[dict] = None,
        n_goals: Optional[int] = None,
        **kwargs
) -> None:
    T_goal_states = jax.vmap(lambda x: x.type_states(type_idx=MVE.GOAL, n_type=self.num_agents))(rollout.graph)
    ref_goals = T_goal_states[:, :, :2]
    n_goals = self.num_agents if n_goals is None else n_goals

    ax: Axes
    xlim = self.params["rollout_state_range"][:2]
    ylim = self.params["default_state_range"][2:4]
    fig, ax = plt.subplots(1, 1, figsize=(30,
                                          (ylim[1] + 3 - (ylim[0] - 3)) * 20 / (xlim[1] + 3 - (xlim[0] - 3)) + 4)
                           , dpi=100)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0] - 3, ylim[1] + 3)
    ax.set(aspect="equal")
    plt.axis("on")
    if viz_opts is None:
        viz_opts = {}

    # 画车道线
    two_yms_bold, l_yms_scatter = process_lane_marks(self.params["default_state_range"][2:4], self.params["lane_width"])
    ax.axhline(y=two_yms_bold[0], linewidth=1.5, color='b')
    ax.axhline(y=two_yms_bold[1], linewidth=1.5, color='b')
    if l_yms_scatter is not None:
        for ym in l_yms_scatter:
            ax.axhline(y=ym, linewidth=1, color='b', linestyle='--')

    # plot the first frame
    T_graph = rollout.graph
    graph0 = tree_index(T_graph, 0)

    agent_color = "#0068ff"
    goal_color = "#2fdd00"
    obst_color = "#8a0000"
    edge_goal_color = goal_color

    # plot obstacles
    obsts_state = graph0.type_nodes(type_idx=MVE.OBST, n_type=self.num_obsts)
    # state: x, y, vx, vy, θ, dθ/dt, bw, bh
    obsts_pos = obsts_state[:, :2]
    obsts_theta = obsts_state[:, 4]
    obsts_bb_size = obsts_state[:, 6:8]
    obsts_radius = jnp.linalg.norm(obsts_bb_size, axis=1)
    plot_obsts_arrow = [FancyArrow(x=obsts_pos[i, 0], y=obsts_pos[i, 1],
                                   dx=jnp.cos(obsts_theta[i] * jnp.pi / 180) * obsts_radius[i] / 2,
                                   dy=jnp.sin(obsts_theta[i] * jnp.pi / 180) * obsts_radius[i] / 2,
                                   length_includes_head=True,
                                   width=0.3, color=obst_color, alpha=1.0) for i in range(len(obsts_theta))]
    plot_obsts_rec = [plt.Rectangle(xy=tuple(obsts_pos[i, :] - obsts_bb_size[i, :] / 2),
                                    width=obsts_bb_size[i, 0], height=obsts_bb_size[i, 1],
                                    angle=obsts_theta[i], rotation_point='center',
                                    color=obst_color, linewidth=0.0, alpha=0.6) for i in range(len(obsts_theta))]
    col_obsts = MutablePatchCollection(plot_obsts_arrow + plot_obsts_rec, match_original=True, zorder=5)
    ax.add_collection(col_obsts)

    # plot agents
    agents_state = graph0.type_states(type_idx=MVE.AGENT, n_type=self.num_agents)
    # state: x, y, vx, vy, θ, dθ/dt, δ, bb_w, bb_h, a0 ... a5
    agents_pos = agents_state[:, :2]
    agents_theta = agents_state[:, 4]
    agents_bb_size = agents_state[:, 6:8]
    agents_radius = jnp.linalg.norm(agents_bb_size, axis=1)
    plot_agents_arrow = [FancyArrow(x=agents_pos[i, 0], y=agents_pos[i, 1],
                                    dx=jnp.cos(agents_theta[i] * jnp.pi / 180) * agents_radius[i] / 2,
                                    dy=jnp.sin(agents_theta[i] * jnp.pi / 180) * agents_radius[i] / 2,
                                    width=agents_radius[i] / jnp.mean(obsts_radius) * 0.3,
                                    length_includes_head=True,
                                    alpha=1.0, color=agent_color) for i in range(self.num_agents)]
    plot_agents_rec = [plt.Rectangle(xy=tuple(agents_pos[i, :] - agents_bb_size[i, :] / 2),
                                     width=agents_bb_size[i, 0], height=agents_bb_size[i, 1],
                                     angle=agents_theta[i], rotation_point='center',
                                     color=agent_color, linewidth=0.0, alpha=0.6) for i in range(self.num_agents)]
    col_agents = MutablePatchCollection(plot_agents_arrow + plot_agents_rec, match_original=True, zorder=6)
    ax.add_collection(col_agents)

    # plot reference points
    # state: x, y, vx, vy, θ, dθ/dt, bw,
    all_ref_xs = ref_goals[:, :, 0].reshape(-1)
    all_ref_ys = ref_goals[:, :, 1].reshape(-1)
    ax.scatter(all_ref_xs, all_ref_ys, color=goal_color, zorder=7, s=5, alpha=1.0, marker='.')

    # plot edges
    all_pos = graph0.states[:, :2]
    edge_index = np.stack([graph0.senders, graph0.receivers], axis=0)
    is_pad = np.any(edge_index == self.num_agents + n_goals + self.num_obsts, axis=0)
    e_edge_index = edge_index[:, ~is_pad]
    e_start, e_end = all_pos[e_edge_index[0, :]], all_pos[e_edge_index[1, :]]
    e_lines = np.stack([e_start, e_end], axis=1)  # (e, n_pts, dim)
    e_is_goal = (self.num_agents <= graph0.senders) & (graph0.senders < self.num_agents + n_goals)
    e_is_goal = e_is_goal[~is_pad]
    e_colors = [edge_goal_color if e_is_goal[ii] else "0.2" for ii in range(len(e_start))]
    col_edges = LineCollection(e_lines, colors=e_colors, linewidths=2, alpha=0.5, zorder=3)
    ax.add_collection(col_edges)

    # texts
    text_font_opts = dict(
        size=16,
        color="k",
        family="sans-serif",
        weight="normal",
        transform=ax.transAxes,
    )
    cost_text = ax.text(0.02, 1.00, "Cost: 1.0\nReward: 1.0", va="bottom", **text_font_opts)
    if Ta_is_unsafe is not None:
        safe_text = [ax.text(0.99, 1.00, "Unsafe: {}", va="bottom", ha="right", **text_font_opts)]
    kk_text = ax.text(0.99, 1.04, "kk=0", va="bottom", ha="right", **text_font_opts)
    if rollout.zs is not None:
        z_text = ax.text(0.5, 1.04, "z: []", va="bottom", ha="center", **text_font_opts)

    # add agent labels
    label_font_opts = dict(
        size=20,
        color="k",
        family="sans-serif",
        weight="normal",
        ha="center",
        va="center",
        transform=ax.transData,
        clip_on=True,
        zorder=8,
        alpha=0.
    )
    agent_labels = [ax.text(float(agents_pos[ii, 0]), float(agents_pos[ii, 1]), f"{ii}", **label_font_opts)
                    for ii in range(self.num_agents)]

    if "Vh" in viz_opts:
        Vh_text = ax.text(0.99, 0.99, "Vh: []", va="top", ha="right", **text_font_opts)

    # init function for animation
    def init_fn() -> List[plt.Artist]:
        return [col_obsts, col_agents, col_edges, *agent_labels, cost_text, *safe_text, kk_text]

    def update(kk: int) -> List[plt.Artist]:
        graph = tree_index(T_graph, kk)
        n_pos_t = graph.states[:-1, :2]  # 最后一个node是padding，不要
        n_theta_t = graph.states[:-1, 4]
        n_bb_size_t = graph.nodes[:-1, 6:8]
        n_radius = jnp.linalg.norm(n_bb_size_t, axis=1)

        # update agents' positions and labels
        for ii in range(self.num_agents):
            plot_agents_arrow[ii].set_data(x=n_pos_t[ii, 0], y=n_pos_t[ii, 1],
                                           dx=jnp.cos(n_theta_t[ii] * jnp.pi / 180) * n_radius[ii] / 2,
                                           dy=jnp.sin(n_theta_t[ii] * jnp.pi / 180) * n_radius[ii] / 2)
            plot_agents_rec[ii].set_xy(xy=tuple(n_pos_t[ii, :] - n_bb_size_t[ii, :] / 2))
            plot_agents_rec[ii].set_angle(angle=n_theta_t[ii])
            agent_labels[ii].set_position(n_pos_t[ii, :])
        # update obstacles' positions
        for ii in range(self.num_obsts):
            plot_obsts_arrow[ii].set_data(x=n_pos_t[self.num_agents + n_goals + ii, 0],
                                          y=n_pos_t[self.num_agents + n_goals + ii, 1],
                                          dx=jnp.cos(n_theta_t[self.num_agents + n_goals + ii] * jnp.pi / 180) *
                                             n_radius[
                                                 self.num_agents + n_goals + ii] / 2,
                                          dy=jnp.sin(n_theta_t[self.num_agents + n_goals + ii] * jnp.pi / 180) *
                                             n_radius[
                                                 self.num_agents + n_goals + ii] / 2)
            plot_obsts_rec[ii].set_xy(xy=tuple(
                n_pos_t[self.num_agents + n_goals + ii, :] - n_bb_size_t[self.num_agents + n_goals + ii, :] / 2))
            plot_obsts_rec[ii].set_angle(angle=n_theta_t[self.num_agents + n_goals + ii])

        # update edges
        e_edge_index_t = np.stack([graph.senders, graph.receivers], axis=0)
        is_pad_t = np.any(e_edge_index_t == self.num_agents + n_goals + self.num_obsts, axis=0)
        e_edge_index_t = e_edge_index_t[:, ~is_pad_t]
        e_start_t, e_end_t = n_pos_t[e_edge_index_t[0, :]], n_pos_t[e_edge_index_t[1, :]]
        e_is_goal_t = (self.num_agents <= graph.senders) & (graph.senders < self.num_agents + n_goals)
        e_is_goal_t = e_is_goal_t[~is_pad_t]
        e_colors_t = [edge_goal_color if e_is_goal_t[ii] else "0.2" for ii in range(len(e_start_t))]
        e_lines_t = np.stack([e_start_t, e_end_t], axis=1)
        col_edges.set_segments(e_lines_t)
        col_edges.set_colors(e_colors_t)

        # update cost and safe labels
        if kk < len(rollout.costs):
            all_costs = ""
            for i_cost in range(rollout.costs[kk].shape[1]):
                all_costs += f"    {self.cost_components[i_cost]}: {rollout.costs[kk][:, i_cost].max():5.4f}\n"
            all_costs = all_costs[:-2]
            cost_text.set_text(f"Cost:\n{all_costs}\nReward: {rollout.rewards[kk]:5.4f}")
        else:
            cost_text.set_text("")
        if kk < len(Ta_is_unsafe):
            a_is_unsafe = Ta_is_unsafe[kk]
            unsafe_idx = np.where(a_is_unsafe)[0]
            safe_text[0].set_text("Unsafe: {}".format(unsafe_idx))
        else:
            safe_text[0].set_text("Unsafe: {}")

        kk_text.set_text("kk={:04}".format(kk))

        # Update the z text.
        if rollout.zs is not None:
            z_text.set_text(f"z: {rollout.zs[kk]}")

        if "Vh" in viz_opts:
            Vh_text.set_text(f"Vh: {viz_opts['Vh'][kk]}")

        return [col_obsts, col_agents, col_edges, *agent_labels, cost_text, *safe_text, kk_text]

    fps = 30.0
    spf = 1 / fps
    mspf = 1_000 * spf
    anim_T = len(T_graph.n_node)
    ani = FuncAnimation(fig, update, frames=anim_T, init_func=init_fn, interval=mspf, blit=True)
    save_anim(ani, video_path)


if __name__ == "__main__":
    # ===================== 配置参数（请根据实际修改） =====================
    BAG_FILE_PATH = "/path/to/your/bag_file.db3"  # 替换为你的rosbag文件路径
    TARGET_TOPICS = ["/ros_env/state", "/ros_env/action", "/ros_env/eval"]

    # 检查bag文件是否存在
    if not os.path.exists(BAG_FILE_PATH):
        print(f"❌ 错误：ROSBag文件不存在 → {BAG_FILE_PATH}")
        sys.exit(1)

    # 初始化rclpy（解析msg需要）
    rclpy.init()

    # 执行分析
    try:
        analyze_rosbag(BAG_FILE_PATH, TARGET_TOPICS)
    except Exception as e:
        print(f"\n❌ 分析过程出错：{e}")
    finally:
        # 确保rclpy正常关闭
        rclpy.shutdown()
        print("\n✅ 分析完成，rclpy已正常关闭")



