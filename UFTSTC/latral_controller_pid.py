import jax
import jax.numpy as jnp
import functools as ft

from typing import Tuple

from scipy.optimize import curve_fit

from defmarl.env.mve import MVE
from defmarl.utils.typing import AgentState, ObstState, Action
from defmarl.utils.utils import nthroot
from defmarl.utils.graph import GraphsTuple

class UFTSTCController_pid:
    """使用LvChen论文里的UFTSTC算法计算前轮转角控制量，去掉了转角上限估计器，改为直接对转角求饱和函数
    state: X(m) Y(m) vx(km/h) vy(km/h) Psi(deg) dPsi(gamma)(deg/s) bw(m) bh(m)
    action: ax(=0)(m/s^2) deltaf(deg/s)"""

    def __init__(self, num_agents:int, num_obsts:int, dt:float, Af: float, r:float, mu:float, c:float, k1:float,
                 k2:float, k3:float, k4:float, v:float, Delta1:float, Delta2:float, p_num:int, p_den:int, alpha:float,
                 Cf:float, Cr:float, Lf:float, Lr:float, m:float, Iz:float,
                 max_deltaf:float, # deg
                 min_deltaf:float, # deg
                 y_min: float,              # 左边界 y
                 y_max: float,              # 右边界 y
                 kp_d:float,
                 ki_d:float,
                 kd_d:float,
                 max_integral_d:float,
                 min_integral_d:float,
                 Af_lane: float = 18.0,     # 边界势场幅值
                 r_lane: float = 1.0,       # “强惩罚区”距离（你自己调）
                 mu_lane: float = 3.0,      # “开始生效”距离（你自己调）
                 c_lane: float = 2.0,       # 形状（你自己调）
                 leak_near_lane: float = 0.08,
                 leak_far_lane: float = 0.35,
                 k_lane_violate: float = 100.0,

                 ):
        # 环境参数设置
        self.num_agents = num_agents
        self.num_obsts = num_obsts
        self.k_lane_violate = k_lane_violate
        # 缓存
        self.delta_xoe = 20 * jnp.ones((num_agents, num_obsts), dtype=jnp.float32)
        self.delta_dxoe = jnp.zeros((num_agents, num_obsts), dtype=jnp.float32)
        self.BD_int = jnp.zeros((num_agents, num_obsts), dtype=jnp.float32)
        self.a_last_deltaf = jnp.zeros((num_agents,), dtype=jnp.float32)


        # 算法参数设置
        self.dt = dt
        self.Af = Af
        self.r = r
        self.mu = mu
        self.c = c
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.v = v
        self.Delta1 = Delta1
        self.Delta2 = Delta2
        self.p_num = p_num
        self.p_den = p_den
        self.alpha = alpha

        # 动力学参数设置
        self.Cf = Cf
        self.Cr = Cr
        self.Lf = Lf
        self.Lr = Lr
        self.m = m
        self.Iz = Iz
        self.max_deltaf = max_deltaf
        self.min_deltaf = min_deltaf

        self.y_min = y_min
        self.y_max = y_max
        self.Af_lane = Af_lane
        self.r_lane = r_lane
        self.mu_lane = mu_lane
        self.c_lane = c_lane
        self.leak_near_lane = leak_near_lane
        self.leak_far_lane = leak_far_lane

        self.cur_step = 0 #debug

        self.kp = kp_d
        self.ki = ki_d
        self.kd = kd_d
        self.i_max = max_integral_d
        self.i_min = min_integral_d

        self.int_epsi = jnp.zeros((num_agents,), dtype=jnp.float32)   # 积分项
        self.prev_epsi = jnp.zeros((num_agents,), dtype=jnp.float32)  # 上一步误差（用于D项差分）


    @ft.partial(jax.jit, static_argnums=(0,))
    def calc_delta_xoe_dxoe_ddxoe(self, aS_agent_states: AgentState, oS_obst_states: ObstState
                                  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        a2_agent_pos = aS_agent_states[:, :2]
        o2_obst_pos = oS_obst_states[:, :2]
        o2_obst_pos_y = oS_obst_states[:,1]
        yy=o2_obst_pos_y
        # distance between agents and obstacles
        ao_delta_xoe = jnp.linalg.norm(jnp.expand_dims(a2_agent_pos, 1) - jnp.expand_dims(o2_obst_pos, 0), axis=-1)
        ao_delta_dxoe = (ao_delta_xoe - self.delta_xoe) / self.dt
        self.delta_xoe = ao_delta_xoe
        ao_delta_ddxoe = (ao_delta_dxoe - self.delta_dxoe) / self.dt
        self.delta_dxoe = ao_delta_dxoe
        jax.debug.print("obst_pos(O,2) = {}", o2_obst_pos)
        jax.debug.print("delta_xoe for agent0 (O,) = {}", ao_delta_xoe[0])
        return ao_delta_xoe, ao_delta_dxoe, ao_delta_ddxoe

    @ft.partial(jax.jit, static_argnums=(0,))
    def calc_Uod_dUod_ddUod(self, aS_agent_states: AgentState, oS_obst_states: ObstState
                            ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        delta_xoe, delta_dxoe, delta_ddxoe = self.calc_delta_xoe_dxoe_ddxoe(aS_agent_states, oS_obst_states)
        dist_ratio = jnp.clip((delta_xoe - self.r) / (self.mu - self.r), 0, 1)
        dist_ratio_pow = dist_ratio**(2*self.c)
        ao_Uod = jnp.where(delta_xoe > self.mu, 0,
                           jnp.where(delta_xoe > self.r,
                                     self.Af*jnp.cos(jnp.pi/2 * dist_ratio_pow),
                                     self.Af))
        ao_dUod = jnp.where(delta_xoe > self.mu, 0,
                            jnp.where(delta_xoe > self.r,
                                      -self.Af*jnp.sin(jnp.pi/2 * dist_ratio_pow) * \
                                      jnp.pi*self.c*delta_dxoe * dist_ratio_pow  / (delta_xoe-self.r),
                                      0))
        ao_ddUod = jnp.where(delta_xoe > self.mu, 0,
                             jnp.where(delta_xoe > self.r,
                                       -self.Af*jnp.cos(jnp.pi/2 * dist_ratio_pow) * \
                                       jnp.pi**2 * self.c**2 * delta_dxoe**2 * dist_ratio**(4*self.c) / (delta_xoe-self.r)**2 \
                                       -self.Af*jnp.sin(jnp.pi/2 * dist_ratio_pow) * \
                                       jnp.pi * self.c * \
                                       ((2*self.c-1) * delta_dxoe**2 * dist_ratio_pow / (delta_xoe-self.r)**2 \
                                        +delta_ddxoe * dist_ratio_pow / (delta_xoe-self.r)),
                                       0))
        return ao_Uod, ao_dUod, ao_ddUod

    @ft.partial(jax.jit, static_argnums=(0,))
    def calc_BD_dBD_ddBD(self, aS_agent_states: AgentState, oS_obst_states: ObstState
                         ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        Uod, dUod, ddUod = self.calc_Uod_dUod_ddUod(aS_agent_states, oS_obst_states)
        delta_xoe, _, _ = self.calc_delta_xoe_dxoe_ddxoe( aS_agent_states, oS_obst_states )
        eps = 1e-4
        ao_BD = jnp.log(self.Af**2 / (self.Af**2 - Uod**2 + eps))
        ao_dBD = 2 * Uod * dUod / (self.Af**2 - Uod**2 + eps)
        ao_ddBD = 2 * (dUod**2 + Uod*ddUod) / (self.Af**2 - Uod**2 + eps) + 4 * Uod**2 * dUod**2 / ((self.Af**2 - Uod**2)**2 + eps)

        # debug
        ao_mask = jnp.ones_like(ao_BD)
        #ao_mask = ao_mask.at[:, 0].set(jnp.zeross((ao_BD.shape[0],)))
        ao_BD = ao_BD * ao_mask
        ao_dBD = ao_dBD * ao_mask
        ao_ddBD = ao_ddBD * ao_mask
        return ao_BD, ao_dBD, ao_ddBD, delta_xoe

    @ft.partial(jax.jit, static_argnums=(0,))
    def calc_lane_BD_terms(self, a_Y: jnp.ndarray, a_dY: jnp.ndarray, a_ddY: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,jnp.ndarray ]:
        """
        计算道路边界的势场 BD, dBD, ddBD，并返回 (A,) 的 sum 和用于泄露判定的 delta_lane。
        """
        eps = 1e-3

        # 计算车辆与道路两条边界的距离
        d_left = a_Y - self.y_min  # (A,) 车辆与左边界的距离
        d_right = self.y_max - a_Y # (A,) 车辆与右边界的距离

        # 计算导数
        dL_dot = a_dY          # (A,) 左边界的速度
        dR_dot = -a_dY         # (A,) 右边界的速度
        dL_ddot = a_ddY        # (A,) 左边界的加速度
        dR_ddot = -a_ddY       # (A,) 右边界的加速度

        # 堆叠得到 (A, 2)，表示左边界和右边界
        d = jnp.stack([d_left, d_right], axis=1)    # (A,2)
        d_dot = jnp.stack([dL_dot, dR_dot], axis=1) # (A,2)
        d_ddot = jnp.stack([dL_ddot, dR_ddot], axis=1) # (A,2)

        # 计算道路势场的比率（dist_ratio）
        dist_ratio = jnp.clip((d - self.r_lane) / (self.mu_lane - self.r_lane), 0.0, 1.0)  # (A,2)
        dist_ratio_pow = dist_ratio ** (2 * self.c_lane)  # (A,2)

        # 计算道路势场值 Uod
        Uod = jnp.where(
            d > self.mu_lane, 0.0,
            jnp.where(
                d > self.r_lane,
                self.Af_lane * jnp.cos(jnp.pi / 2 * dist_ratio_pow),
                self.Af_lane
            )
        )  # (A, 2)

        # 计算导数 dUod
        dUod = jnp.where(
            d > self.mu_lane, 0.0,
            jnp.where(
                d > self.r_lane,
                -self.Af_lane * jnp.sin(jnp.pi / 2 * dist_ratio_pow)
                * jnp.pi * self.c_lane * d_dot * dist_ratio_pow / (d - self.r_lane),
                0.0
            )
        )  # (A, 2)

        # 计算二阶导数 ddUod
        ddUod = jnp.where(
            d > self.mu_lane, 0.0,
            jnp.where(
                d > self.r_lane,
                -self.Af_lane * jnp.cos(jnp.pi / 2 * dist_ratio_pow)
                * (jnp.pi ** 2) * (self.c_lane ** 2) * (d_dot ** 2) * (dist_ratio ** (4 * self.c_lane))
                / ((d - self.r_lane) ** 2)
                - self.Af_lane * jnp.sin(jnp.pi / 2 * dist_ratio_pow)
                * jnp.pi * self.c_lane
                * (
                        (2 * self.c_lane - 1) * (d_dot ** 2) * dist_ratio_pow / ((d - self.r_lane) ** 2)
                        + d_ddot * dist_ratio_pow / (d - self.r_lane)
                ),
                0.0
            )
        )  # (A, 2)

        # 计算 BD 和其导数
        violate = jnp.maximum(-d, 0.0)   # d<0 时才有值

        k_violate = self.k_lane_violate  # 建议 50 ~ 300 起步
        BD = jnp.log(self.Af_lane**2 / (self.Af_lane**2 - Uod**2 + eps))  # (A, 2)
        dBD = 2.0 * Uod * dUod / (self.Af_lane**2 - Uod**2 + eps)  # (A, 2)
        #BD  = BD  + k_violate * (violate ** 2)
        #dBD = dBD + k_violate * 2.0 * violate * (-d_dot)
        #BD = jnp.log(self.Af_lane**2 / (self.Af_lane**2 - Uod**2 + eps))  # (A, 2)
        #dBD = 2.0 * Uod * dUod / (self.Af_lane**2 - Uod**2 + eps)  # (A, 2)
        ddBD = (
                2.0 * (dUod**2 + Uod * ddUod) / (self.Af_lane**2 - Uod**2 + eps)
                + 4.0 * (Uod**2) * (dUod**2) / ((self.Af_lane**2 - Uod**2) ** 2 + eps)
        )  # (A, 2)

        # 返回 sum (A,) + 以及 d 用于泄露判定
        return BD, dBD, ddBD, d ,Uod # (A,)


    '''
    @ft.partial(jax.jit, static_argnums=(0,))
    def integrate_BD_update(self, ao_BD:jnp.ndarray) -> jnp.ndarray:
        #ao_BD_int = self.BD_int + ao_BD*self.dt
       # self.BD_int = ao_BD_int
        #return ao_BD_int
        
        tau = 1.2  # 秒，建议先试 1.0~3.0
        rho = jnp.exp(-self.dt / tau)   # 每步衰减系数
        self.BD_int = rho * self.BD_int + ao_BD * self.dt
        return self.BD_int

        leak_near = 0.08   # 近障：慢泄露（保留一点“提前避”）
        leak_far  = 0.35   # 远离：快泄露（避免“多推一下”）

        near = (ao_delta_xoe < self.mu).astype(jnp.float32)  # (A,O)
        leak = leak_near * near + leak_far * (1.0 - near)

        # 只在近障积分；远离后不再累加，只快速遗忘
        self.BD_int = (1.0 - leak) * self.BD_int + near * ao_BD * self.dt
        return self.BD_int
    '''

    @ft.partial(jax.jit, static_argnums=(0,))
    def integrate_BD_update(self,ao_BD: jnp.ndarray,ao_delta_xoe: jnp.ndarray)-> jnp.ndarray:
        leak_near = 0.08
        leak_far  = 0.35
        near = (ao_delta_xoe < self.mu).astype(jnp.float32)  # (A, O)
        leak = leak_near * near + leak_far * (1.0 - near)
        self.BD_int = (1.0 - leak) * self.BD_int + near * ao_BD * self.dt
        #self.BD_int =self.BD_int + ao_BD * self.dt
        return self.BD_int

    @ft.partial(jax.jit, static_argnums=(0,))
    def calc_Y_dY_ddY(self, aS_agent_states: AgentState, a_deltaf: jnp.ndarray
                      ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        a_Y = aS_agent_states[:, 1]
        a_vx = aS_agent_states[:, 2] / 3.6 # metric
        a_vy = aS_agent_states[:, 3] / 3.6 # metric
        a_Psi = aS_agent_states[:, 4] * jnp.pi/180 # metric
        a_dPsi = a_gamma = aS_agent_states[:, 5] * jnp.pi/180 # metric
        a_deltaf = a_deltaf * jnp.pi/180 # metric

        a_dvy = (self.Cf+self.Cr)*a_vy / (self.m*a_vx) - self.Cf * a_deltaf / self.m \
                + ((self.Lf*self.Cf - self.Lr*self.Cr)/(self.m*a_vx) - a_vx)*a_gamma
        a_dY = a_vx * a_Psi + a_vy
        a_ddy = a_vx * a_gamma + a_dvy

        return a_Y, a_dY, a_ddy


    @ft.partial(jax.jit, static_argnums=(0,))
    def calc_beta_dbeta_ddbeta_metric(self, aS_agent_states: AgentState, a_deltaf: jnp.ndarray
                                      ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        a_vx = aS_agent_states[:, 2] / 3.6 # metric
        a_vy = aS_agent_states[:, 3] / 3.6 # metric
        a_dPsi = a_gamma = aS_agent_states[:, 5] * jnp.pi/180 # metric
        a_deltaf = a_deltaf * jnp.pi/180 # metric

        a_beta_metric = a_vy / a_vx
        a_dvy = (self.Cf + self.Cr) * a_vy / (self.m * a_vx) - self.Cf * a_deltaf / self.m \
                + ((self.Lf * self.Cf - self.Lr * self.Cr) / (self.m * a_vx) - a_vx) * a_gamma
        a_dbeta_metric = a_dvy / a_vx
        a_ddbeta_metric = jnp.zeros_like(a_beta_metric) # 无法得知deltaf的导数如何求得，暂时先设置为0，看看控制效果再说

        return a_beta_metric, a_dbeta_metric, a_ddbeta_metric


    @ft.partial(jax.jit, static_argnums=(0,))
    def calc_Psid_dPsid_ddPsid_metric(self, aS_agent_states: AgentState, oS_obst_states: ObstState, a_deltaf: jnp.ndarray,
                                      a_Yd: jnp.ndarray, a_dYd: jnp.ndarray, a_ddYd: jnp.ndarray, a_dddYd: jnp.ndarray
                                      ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # 计算每个障碍物与agent的误差，静态，动态，下边界，上边界
        A = aS_agent_states.shape[0]
        N = oS_obst_states.shape[0]
        Y = aS_agent_states[:, 1:2]
        obs_y = oS_obst_states[:, 1]

        lane_y = jnp.array([self.y_min, self.y_max], dtype=obs_y.dtype)
        Y0 = jnp.concatenate([obs_y, lane_y], axis=0)  # (O,)
        Y0_AO = jnp.broadcast_to(Y0[None, :], (A, N + 2))   # (A,O)
        Y_AO  = jnp.broadcast_to(Y,          (A, N + 2))    # (A,O)
        ao_bar_Ye = Y0_AO - Y_AO          # (A,O)
        ao_bar_sgn_Ye=jnp.sign(ao_bar_Ye)

        a_Y, a_dY, a_ddY = self.calc_Y_dY_ddY(aS_agent_states, a_deltaf)
        a_Ye = a_Yd - a_Y
        a_sgn_Ye = jnp.sign(a_Ye)
        a_dYe = a_dYd - a_dY
        a_ddYe = a_ddYd - a_ddY

        a_beta, a_dbeta, a_ddbeta = self.calc_beta_dbeta_ddbeta_metric(aS_agent_states, a_deltaf) # metric

        a_vx = aS_agent_states[:, 2] / 3.6 # metric

        ao_BD, ao_dBD, ao_ddBD,delta_xoe= self.calc_BD_dBD_ddBD(aS_agent_states, oS_obst_states)
        BD_lane, dBD_lane, ddBD_lane, d_lane,Uod = self.calc_lane_BD_terms(a_Y, a_dY, a_ddY)

        #边界
        BD_AO = jnp.concatenate([ao_BD, BD_lane], axis=1)

        # [静态车, 动态车, y_min边界, y_max边界]
        w = jnp.array([0.7,0.5, 2.5, 1], dtype=BD_AO.dtype)  #
        BD_weighted_sum = jnp.sum(BD_AO * ao_bar_sgn_Ye * w[None, :], axis=1)

        weight_obstacle = 1
        weight_lane = 1
        a_BD_sum = (weight_obstacle * ao_BD.sum(axis=1)) + (weight_lane * BD_lane.sum(axis=1))
        a_dBD_sum = (weight_obstacle * ao_dBD.sum(axis=1)) + (weight_lane * dBD_lane.sum(axis=1))
        a_ddBD_sum = (weight_obstacle * ao_ddBD.sum(axis=1)) + (weight_lane * ddBD_lane.sum(axis=1))

        ao_BD_int = self.integrate_BD_update(ao_BD,delta_xoe)
        a_BD_int_sum = ao_BD_int.sum(axis=1)
        # ===== [ADD] lane boundary terms =====
        # a_BD_int_sum = a_BD_int_sum + a_BD_int_sum_lane

        # a_Psid_metric = self.k1*jnp.exp(-self.v*a_BD_sum)*a_Ye/a_vx - self.k2*a_BD_sum*a_sgn_Ye/((a_BD_int_sum+1)*a_vx) \
        #               +a_dYd/a_vx*0.6 - a_beta*0.1
        a_Psid_metric = self.k1*jnp.exp(-self.v*a_BD_sum)*a_Ye/a_vx - self.k2*BD_weighted_sum/((a_BD_int_sum+1)*a_vx) \
                       *0 +a_dYd/a_vx*0.999 - a_beta*0
        a_Psid_metric=jnp.clip(a_Psid_metric,-jnp.pi/2, jnp.pi/2)
        self.cur_step += 1
        jax.debug.print("step:{cur_step} \n"
                        "a_Psid:{a_Psid} \n"
                        "ao_BD={ao_BD} \n"
                        "bd_lane:{bd_lane} \n"
                        "Uod:{Uod} \n"
                        ,
                        cur_step=self.cur_step,
                        a_Psid=a_Psid_metric*180/jnp.pi,
                        ao_BD=ao_BD,
                        bd_lane=BD_lane,
                        Uod=Uod)

        return a_Psid_metric,ao_BD, BD_lane ,a_Ye


    @ft.partial(jax.jit, static_argnums=(0,))
    def calc_omegat(self):
        return 0 # 暂时将噪声置为0


    @ft.partial(jax.jit, static_argnums=(0,))
    def clip_deltaf(self, a_deltaf:jnp.ndarray) -> jnp.ndarray:
        """处理的单位为deg"""
        a_clipped_deltaf = jnp.clip(a_deltaf, self.min_deltaf, self.max_deltaf)
        return a_clipped_deltaf


    @ft.partial(jax.jit, static_argnums=(0,))
    def nomalize_deltaf(self, a_deltaf:jnp.ndarray) -> jnp.ndarray:
        """处理单位为deg，将[max_deltaf,min_deltaf]区间内的转向角线性映射至[-1,1]区间"""
        deltaf_center = (self.max_deltaf + self.min_deltaf)/2
        deltaf_half = self.max_deltaf - deltaf_center
        a_normalized_deltaf = (a_deltaf - deltaf_center) / deltaf_half
        return a_normalized_deltaf


    @ft.partial(jax.jit, static_argnums=(0,))
    def wrap_pi(self, ang):
        return (ang + jnp.pi) % (2*jnp.pi) - jnp.pi

    @ft.partial(jax.jit, static_argnums=(0,))
    def calc_deltaf(self, graph: GraphsTuple, a4_dsYddt: jnp.ndarray)-> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray,jnp.ndarray, jnp.ndarray , jnp.ndarray, jnp.ndarray,jnp.ndarray, jnp.ndarray , jnp.ndarray]:
        aS_agent_states = graph.type_states(MVE.AGENT, n_type=self.num_agents)
        oS_obst_states  = graph.type_states(MVE.OBST,  n_type=self.num_obsts)
        T_goal_states  = graph.type_states(MVE.GOAL,  n_type=self.num_agents)

        a_Yd, a_dYd, a_ddYd, a_dddYd = a4_dsYddt[:,0], a4_dsYddt[:,1], a4_dsYddt[:,2], a4_dsYddt[:,3]

        # 外环：仍然用你现在的方式算期望航向角（以及调试输出）
        a_Psid_metric, ao_BD, BD_lane ,a_Ye= \
            self.calc_Psid_dPsid_ddPsid_metric(aS_agent_states, oS_obst_states,
                                               self.a_last_deltaf, a_Yd, a_dYd, a_ddYd, a_dddYd)

        # 实际航向角/角速度（rad）
        a_Psi_metric  = aS_agent_states[:, 4] * jnp.pi/180
        a_dPsi_metric = aS_agent_states[:, 5] * jnp.pi/180

        # PID 误差（wrap 到 [-pi, pi]）
        e_psi = self.wrap_pi(a_Psid_metric - a_Psi_metric)

        # P
        p = self.kp * e_psi

        # I（积分防饱和：先积分再夹紧）
        self.int_epsi = jnp.clip(self.int_epsi + e_psi * self.dt, self.i_min, self.i_max)
        i = self.ki * self.int_epsi

        # D：推荐用角速度误差，而不是差分（更稳）
        #e_dpsi = dPsi_d - dPsi
        #e_dpsi = a_dPsid_metric - a_dPsi_metric
        #d = self.kd * e_dpsi
        d=0
        # PID 输出：这里输出的是前轮转角命令（rad）
        deltaf_cmd_rad = p + i + d

        # 饱和到物理转角（deg）
        deltaf_cmd_deg  = deltaf_cmd_rad * 180/jnp.pi
        deltaf_clip_deg = self.clip_deltaf(deltaf_cmd_deg)

        # 存起来给下一步用（注意：你 a_last_deltaf 单位是 deg）
        self.a_last_deltaf = deltaf_clip_deg

        # 归一化给环境
        deltaf_norm = self.nomalize_deltaf(deltaf_clip_deg)

        return deltaf_norm, a_Psid_metric*180/jnp.pi, ao_BD, BD_lane, a_Ye ,aS_agent_states , oS_obst_states , a_Yd , deltaf_clip_deg , T_goal_states

