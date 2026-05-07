import os
import numpy as np

def _ensure_TA(x, T, A, name="x"):
    x = np.array(x)
    if x.ndim == 1:          # (T,) -> (T,A,1)
        x = np.broadcast_to(x[:, None, None], (T, A, 1))
    elif x.ndim == 2:        # (T,A) -> (T,A,1)
        x = x[:, :, None]
    elif x.ndim == 3:        # (T,A,dim) OK
        pass
    else:
        raise ValueError(f"{name} shape unexpected: {x.shape}")
    return x

def _auto_scene_dir_uftstc_csv(scene_tag=None,uftstc_root=None):
    """
    返回 UFTSTC/csv/s{n} 目录（自动递增）。
    scene_tag: 你也可以传 "s1" / "scene_xxx" 之类固定名称；不传则自动 s1,s2...
    """
    # 以“当前文件所在目录”为基准（把这个函数放在 UFTSTC/utils.py 或 UFTSTC/xxx.py 里最稳）
    if uftstc_root is None:
        # 从当前运行目录找 UFTSTC
        uftstc_root = os.path.join(os.getcwd(), "UFTSTC")
    #uftstc_dir = os.path.dirname(os.path.abspath(__file__))  # .../UFTSTC
    base_csv_dir = os.path.join(uftstc_root, "csv")
    os.makedirs(base_csv_dir, exist_ok=True)

    if scene_tag is not None:
        scene_dir = os.path.join(base_csv_dir, str(scene_tag))
        os.makedirs(scene_dir, exist_ok=True)
        return scene_dir

    # 自动找下一个 sN
    exist = [
        d for d in os.listdir(base_csv_dir)
        if os.path.isdir(os.path.join(base_csv_dir, d)) and d.startswith("s") and d[1:].isdigit()
    ]
    if len(exist) == 0:
        next_id = 1
    else:
        next_id = max(int(d[1:]) for d in exist) + 1

    scene_dir = os.path.join(base_csv_dir, f"s{next_id}")
    os.makedirs(scene_dir, exist_ok=True)
    return scene_dir

def dump_rollout_record_to_csv6(rollout, record, prefix="epi0", scene_tag=None,uftstc_root=None):
    """
    自动输出到 UFTSTC/csv/sN/ 下面
    - scene_tag=None -> 自动 s1,s2...
    - scene_tag="s3" -> 固定写到 UFTSTC/csv/s3/
    """
    out_dir = _auto_scene_dir_uftstc_csv(scene_tag=scene_tag, uftstc_root=uftstc_root)

    OBST_COLS = ["x", "y", "vx", "vy", "theta", "dtheta", "bw", "bh"]
    header_T = ["x", "y", "vx", "vy", "psi", "dpsi", "bw", "bh", "timestep"]

    # ===== 1) 拿数据（转 numpy）=====
    rewards = -np.array(rollout.rewards)          # (T,) or (T,A)
    costs_real = np.array(rollout.costs_real)    # 常见 (T,A,4)

    aS_agent = np.array(record.aS_agent_states)  # (T,A,agent_dim)
    oS_obst  = np.array(record.oS_obst_states)   # (T,O,obst_dim)
    a_Yds    = np.array(record.a_Yd)             # (T,A) or (T,A,1)
    action_sums = np.array(record.action_sum)
    T_goal_states = np.array(record.T_goal_states)# (T,A,2)

    T, A, agent_dim = aS_agent.shape
    timestep = np.arange(T, dtype=np.int32)

    rewards_TA1 = _ensure_TA(rewards, T, A, "rewards")   # (T,A,1)

    # costs_real -> (T,A,C)
    if costs_real.ndim == 2:
        costs_real_TAC = costs_real[:, :, None]
    elif costs_real.ndim == 3:
        costs_real_TAC = costs_real
    else:
        raise ValueError(f"costs_real shape unexpected: {costs_real.shape}")
    C = costs_real_TAC.shape[-1]

    a_Yds_TA1 = _ensure_TA(a_Yds, T, A, "a_Yd")  # (T,A,1)

    if action_sums.ndim != 3:
        raise ValueError(f"action_sum expected (T,A,2), got {action_sums.shape}")

    timestep_TA1 = np.broadcast_to(timestep[:, None, None], (T, A, 1))

    # =========================================================
    # 1) agent.csv
    # =========================================================
    agent_rows = np.concatenate([aS_agent, timestep_TA1], axis=-1)  # (T,A,agent_dim+1)
    agent_rows_2d = agent_rows.reshape(T*A, -1)
    header_agent = [f"agent_s{i}" for i in range(agent_dim)] + ["timestep"]
    np.savetxt(os.path.join(out_dir, f"{prefix}_agent.csv"),
               agent_rows_2d, delimiter=",", header=",".join(OBST_COLS + ["timestep"]), comments="")

    # =========================================================
    # 2) obstacle 单独输出：obstacle0.csv, obstacle1.csv ...
    # =========================================================
    if oS_obst.size > 0:
        T2, O, obst_dim = oS_obst.shape
        assert T2 == T, f"obstacle T={T2} != agent T={T}"
        if obst_dim < 8:
            raise ValueError(f"obstacle dim too small: {obst_dim}")

        oS8 = oS_obst[:, :, :8]  # (T,O,8)
        for oi in range(O):
            obst_i = oS8[:, oi, :]                       # (T,8)
            obst_i_rows = np.concatenate([obst_i, timestep[:, None]], axis=1)  # (T,9)
            np.savetxt(os.path.join(out_dir, f"{prefix}_obstacle{oi}.csv"),
                       obst_i_rows, delimiter=",",
                       header=",".join(OBST_COLS + ["timestep"]), comments="")
    else:
        # 没有障碍物就不写（或者你也可以写空文件）
        pass

    # =========================================================
    # 3) action.csv
    # =========================================================
    action_rows = np.concatenate([action_sums, timestep_TA1], axis=-1)  # (T,A,3)
    action_rows_2d = action_rows.reshape(T*A, -1)
    np.savetxt(os.path.join(out_dir, f"{prefix}_action.csv"),
               action_rows_2d, delimiter=",",
               header="ax_clip,deltaf_clip_deg,timestep", comments="")

    # =========================================================
    # 4) reward.csv
    # =========================================================
    reward_rows = np.concatenate([rewards_TA1, timestep_TA1], axis=-1)  # (T,A,2)
    reward_rows_2d = reward_rows.reshape(T*A, -1)
    np.savetxt(os.path.join(out_dir, f"{prefix}_reward.csv"),
               reward_rows_2d, delimiter=",",
               header="reward,timestep", comments="")

    # =========================================================
    # 5) cost_real.csv
    # =========================================================
    cost_rows = np.concatenate([costs_real_TAC, timestep_TA1], axis=-1)  # (T,A,C+1)
    cost_rows_2d = cost_rows.reshape(T*A, -1)
    header_cost = [f"cost_real_{i}" for i in range(C)] + ["timestep"]
    np.savetxt(os.path.join(out_dir, f"{prefix}_cost_real.csv"),
               cost_rows_2d, delimiter=",",
               header=",".join(header_cost), comments="")

    # =========================================================
    # 6) Yd.csv
    # =========================================================
    yd_rows = np.concatenate([a_Yds_TA1, timestep_TA1], axis=-1)  # (T,A,2)
    yd_rows_2d = yd_rows.reshape(T*A, -1)
    np.savetxt(os.path.join(out_dir, f"{prefix}_Yd.csv"),
               yd_rows_2d, delimiter=",",
               header="Yd,timestep", comments="")
    # =========================================================
    # 7) T_states.csv  (T_goal_states: 8维 x y vx vy psi dpsi bw bh)
    # =========================================================
    T_goal_states = np.array(record.T_goal_states)  # (T,8) 或 (T,A,8)

    if T_goal_states.ndim == 2:          # (T,8) -> (T,1,8)
        T_goal_states_TA8 = T_goal_states[:, None, :]
    elif T_goal_states.ndim == 3:        # (T,A,8)
        T_goal_states_TA8 = T_goal_states
    else:
        raise ValueError(f"T_goal_states shape unexpected: {T_goal_states.shape}")

    if T_goal_states_TA8.shape[-1] != 8:
        raise ValueError(f"T_goal_states last dim should be 8, got {T_goal_states_TA8.shape}")

    T_rows = np.concatenate([T_goal_states_TA8, timestep_TA1], axis=-1)  # (T,A,9)
    T_rows_2d = T_rows.reshape(T*A, -1)

    header_T = ["x", "y", "vx", "vy", "theta", "dtheta", "bw", "bh", "timestep"]
    np.savetxt(os.path.join(out_dir, f"{prefix}_T_states.csv"),
               T_rows_2d, delimiter=",",
               header=",".join(header_T), comments="")



    print(f"[csv] saved to: {out_dir}")
