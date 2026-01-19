# 状态量
SingleObjectState[] as_agent_states # ego
SingleObjectState[] as_goal_states  # goal
SingleObjectState[] os_obst_states  # obstacles
int32 state_dim # state_dim=8，方便解析

# 动作量
SingleAgentControl[] ad_action
int32 action_dim # action_dim=2, 方便解析

# 评价量
float32 reward # -l(xk)
float32 cost # max{h(xk)}
float32 cost_real # 真实的cost，thresh=1
