"""
SO101 Lift 真实感训练辅助 Wrapper。
"""

import numpy as np

# 兼容 gymnasium 和 openai gym
try:
    import gymnasium as gym
except ImportError:
    import gym


# =============================================================================
# 奖励分量清单
# =============================================================================
# SO101LiftRewardShapingWrapper 在每步 info 中写入的奖励分项键名。
# 诊断脚本、可视化脚本、analyze_rollouts 都从这里取键，保证各工具一致、
# 改奖励分项时只需改这一处。
# 顺序刻意分为：任务进度（正）→ 运动质量（负）→ 行为先验（正）→ 总量。
REWARD_COMPONENTS = [
    # 任务进度（正信号，反映策略是否在完成任务）
    "reward_reach",         # 接近 cube
    "reward_grasp",         # 抓住 cube
    "reward_lift",          # 抬起 cube
    # 运动质量惩罚（负信号，反映运动是否平稳/合理）
    "reward_smooth",        # 动作平滑度
    "reward_vel",           # 关节速度
    "reward_ee_vel",        # 末端速度
    "reward_z_float",       # 末端悬浮
    # 行为先验
    "reward_gripper_move",  # 夹爪运动
    # 总量（用于校验分解求和是否等于实际奖励）
    "original_reward",      # robosuite 原始奖励（成功时为 2.25）
    "shaped_reward",        # 合成后的最终单步奖励
]

# 仅“任务进度”类分量（正信号），用于诊断时聚焦策略行为
REWARD_PROGRESS_COMPONENTS = ["reward_reach", "reward_grasp", "reward_lift"]
# 仅“惩罚”类分量（负信号），用于诊断时聚焦运动质量
REWARD_PENALTY_COMPONENTS = ["reward_smooth", "reward_vel", "reward_ee_vel", "reward_z_float"]


class SO101LiftObservationWrapper(gym.Wrapper):
    """
    为 realistic_state 模式补充“可感知”的物体相对状态。

    真实机器人没有仿真器中的 oracle 物体位姿，但可以通过相机、手眼标定或
    外部运动捕捉估计末端到物体的相对位置。本 Wrapper 把 eef -> cube 的
    三维相对位置拼接到原始观测中，使策略在 realistic_state 下仍能感知
    目标方向，显著降低探索难度。

    注意：只在 use_object_obs=False 时使用；easy 模式已提供完整物体观测。
    """

    def __init__(self, env):
        super().__init__(env)
        self._base_env = None
        self._obs_dim = self.observation_space.shape[0]
        # 拼接 3D 相对位置
        self._rel_pos_dim = 3
        new_dim = self._obs_dim + self._rel_pos_dim
        low = np.full(new_dim, -np.inf, dtype=np.float32)
        high = np.full(new_dim, np.inf, dtype=np.float32)
        # 继承原始观测空间的边界
        low[:self._obs_dim] = self.observation_space.low
        high[:self._obs_dim] = self.observation_space.high
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    def _get_base_env(self):
        if self._base_env is None:
            env = self.env
            while hasattr(env, "env"):
                env = env.env
            self._base_env = env
        return self._base_env

    def _get_eef_pos(self):
        base_env = self._get_base_env()
        eef_site_id = base_env.robots[0].eef_site_id["right"]
        return np.array(base_env.sim.data.site_xpos[eef_site_id])

    def _get_cube_pos(self):
        base_env = self._get_base_env()
        # 与 _check_success / reward 使用一致的方式读取 cube 位置
        return np.array(base_env.sim.data.body_xpos[base_env.cube_body_id])

    def _get_rel_pos(self):
        return self._get_cube_pos() - self._get_eef_pos()

    def _augment_obs(self, obs):
        rel_pos = self._get_rel_pos()
        return np.concatenate([obs, rel_pos]).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._augment_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment_obs(obs), reward, terminated, truncated, info


class SO101LiftRewardShapingWrapper(gym.Wrapper):
    """
    为 SO101 Lift 的 sparse/realistic_state 模式提供更合理、更稠密的奖励塑形。

    设计分层：
      1. 任务进度奖励（可访问仿真内部状态，但不在观测中暴露给策略）：
         - 接近奖励 r_reach：PBRS 势能差分形式，夹爪到 cube 距离的相对改善
         - 抓取奖励 r_grasp：夹爪与 cube 接触
         - 抬升奖励 r_lift：cube 被抬起的高度
      2. 运动质量惩罚（基于本体感受，抑制不稳定行为）：
         - 动作平滑度 r_smooth：抑制高频大振幅动作
         - 关节速度惩罚 r_vel：鼓励低速平稳运动
         - 末端抖动惩罚 r_ee_vel：抑制末端高频抖动
         - 末端悬浮惩罚 r_z_float：抑制末端长期悬浮在桌面上方不下降
      3. 行为先验奖励（引导更自然的抓取-抬升行为）：
         - 夹爪运动奖励 r_gripper_move：避免夹爪僵硬

    注意：
      - 移除了原版本中“无条件夹爪闭合奖励”，该奖励会鼓励策略在尚未接近
        cube 时就把夹爪闭合，反而阻碍下降与接近。
      - 任务进度奖励只在 shaping 模式下叠加，不替代原始 sparse reward。
      - 原始成功奖励仍为 2.25，是最终目标；shaping 奖励帮助策略在稀疏信号下探索。
      - 各项系数经过量级平衡，使得 shaping 奖励总值与原始成功奖励处于同一数量级。
      - easy 模式不启用（Lift 本身已提供稠密奖励）。

    单位说明：
      - 距离/位置单位均为米（m），例如 dist=0.05 表示 5cm。
      - 速度单位为 m/s 或 rad/s（取决于来源），惩罚中用平方求和。
      - 奖励为无量纲标量，最终与原始 sparse reward（成功时 2.25）相加。

    调参指南（按重要性排序）：
      1. 量级平衡：所有分项之和在同一步内应在 [-1, +1.5] 量级，避免某一项
         独大而淹没其他信号。改任意系数后建议用 analyze_rollouts 检查各分项
         均值是否仍在合理区间。
      2. 正负信号配比：任务进度（正）应略大于运动惩罚（负）总量，否则策略
         会学成"不动最安全"。经验比例约为 正:负 ≈ 3:1。
      3. r_reach 采用 PBRS 势能差分形式（详见 __init__ 注释）：
         - α（w_reach_pbrs）：差分幅度，每步接近 1cm 给 α·0.005
         - k（reach_pbrs_scale）：tanh 陡峭度，控制远/近距离信号分布
         关键性质：静止时 r_reach=0，从原理上消除"远距离静止高原"。
      4. r_lift 基线 _cube_rest_z：必须用 cube 静止高度，不要用桌面
         table_offset[2]，否则 cube 不动也持续得分（历史 bug 根因）。
      5. r_z_float 阈值：控制"悬浮惩罚"触发高度。过大则正常接近被误伤；
         过小则无法抑制悬浮。与 r_reach PBRS 协同：悬浮时 r_reach=0、
         z_float 为负，强迫策略移动。
      6. 所有 last_* 状态（含 last_dist）在 reset 中初始化，避免首步用 None
         计算报错。
    """

    def __init__(self, env, mode="realistic_state"):
        super().__init__(env)
        self.mode = mode
        self.last_action = None
        self.last_eef_pos = None
        self.last_gripper_qpos = None

        # ==================================================================
        # 任务进度奖励系数（正信号：驱动策略完成 接近→抓取→抬升 三阶段）
        # 这部分是稀疏任务中"探索"的核心动力，应保证其量级在总奖励中占主导。
        # ==================================================================
        # ─────────────────────────────────────────────────────────────────
        # 接近奖励 r_reach：PBRS（势能差分）形式。
        # ─────────────────────────────────────────────────────────────────
        # 原设计（绝对形式）r_reach = w_reach · (1 - tanh(k·dist)) 已废弃！
        # 原因：实测策略即使把 reach_tanh_scale 从 5.0 → 2.0 → 1.0 一路降低，
        #   仍学到"悬浮在 z=1.22m 不动"。诊断显示：
        #     - 悬浮时 dist=0.4，原 r_reach=0.496/step（饱和到 62%）
        #     - 策略不动也得 +0.50/step，叠加 z_float=-0.18/step，净 +0.32/step
        #     - 这是 reward landscape 上的"远距离静止高原"，无法被梯度打破
        #   详见 reward_function_guide.md §9.4.2 与 so101_reward_diagnostic.md §8。
        #
        # 新设计：基于势函数 F(s) = -α·tanh(k·dist) 的差分（PBRS）
        #   公式：r_reach = α · (tanh(k·dist_prev) - tanh(k·dist_now))
        #
        # 各场景验证（α=20, k=5）：
        #   - 静止悬浮（dist_prev=dist_now=0.4）→ r_reach = 0     ← 关键：消除高原
        #   - 接近 1cm（0.05→0.04）              → r_reach = +0.96
        #   - 远离 3cm（0.07→0.10）              → r_reach = -1.18
        #   - 远距离大跳 10cm（0.4→0.30）         → r_reach = +1.18
        #   - 全 episode 累计（dist 0.4→0）       → +7.6（一次性接近给满）
        #
        # 与 z_float 配合：悬浮时 r_reach=0 但 z_float=-0.18/step → 净负奖励，
        #   策略必须移动才能避免负奖励。这是打破"悬浮局部最优"的关键。
        #
        # 调参指南：
        #   - α（幅度）：每步接近 1cm 给 α·0.005。α=20 → +0.1/step（合理）
        #     策略不动 → 提高 α；策略震荡（来回小幅移动刷分） → 降低 α
        #   - k（陡峭度）：值大 → 近距离信号强；值小 → 远距离信号也强
        #     远距离不动 → 调小 k（如 3.0）；近处不精细 → 调大 k（如 8.0）
        self.w_reach_pbrs = 20.0       # PBRS 差分幅度（量级与 lift 同档）
        self.reach_pbrs_scale = 5.0    # tanh 陡峭度（恢复原值，因为差分形式下远距离静止得 0）
        self.last_dist = None          # 上一步 dist，在 reset 中初始化

        # 抓取奖励 r_grasp：二值里程碑，抓到即给一次性奖励（每步都给，直到松开）。
        #   仅当夹爪与 cube 接触时为 w_grasp，否则 0。用于在 reach 与 lift 之间
        #   建立"抓取"过渡信号，避免策略直接从接近跳到抬升而未真正抓住。
        self.w_grasp = 0.50

        # 抬升奖励 r_lift：cube 相对其“静止放置高度”被抬起的高度（米），
        # 归一化到 [0, w_lift]。公式：r_lift = w_lift * clip(h / target, 0, 1)。
        # 关键：基线必须用 cube 静止时的高度（self._cube_rest_z），而不是桌面
        # table_offset[2]。cube 静止时中心已在 table_offset[2] 之上约 2cm，若以
        # table_offset[2] 为基线，cube 不动也会持续拿到 ~0.5/step 的“抬升奖励”，
        # 这正是“机械臂抬起来就不下来、任务全失败却奖励 ~85”的根因。
        # 调参：target 过小 → 轻抬即满分不再继续；过大 → 信号过稀疏。
        self.w_lift = 1.00            # 抬升满分（cube 抬到 target_height 时）
        self.lift_target_height = 0.04  # 目标抬升高度（米），4cm 即给满 lift 奖励
        self._cube_rest_z = None  # 在 reset 时记录 cube 静止放置时的 z（米）

        # ==================================================================
        # 运动质量惩罚系数（负信号：抑制不稳定/不合理运动）
        # 这部分用于让动作更平稳、更符合真实硬件约束，但总量必须小于任务进度
        # 正信号，否则策略会学成"完全不动最划算"。
        #
        # 调参历史（针对"悬浮不动"症状的全面下调）：
        #   实测 10 episodes 中 reward_ee_vel 均值仅 -0.0005（策略几乎完全静止），
        #   reward_smooth=-1.35、reward_vel=-0.15。说明原惩罚下策略通过"不动"
        #   来规避惩罚。本次整体下调 motion 惩罚，让策略敢于尝试运动下降。
        # ==================================================================
        # 动作平滑度 r_smooth：惩罚相邻两步动作的剧烈变化。
        #   公式：r_smooth = -w * sum((a_t - a_{t-1})^2)
        #   action 为归一化 [-1,1] 动作，平方和维度=action_dim。
        #   0.05 → 0.02：让策略敢于尝试不同动作（SAC 探索所需）
        self.w_action_smooth = 0.02

        # 关节速度 r_vel：惩罚关节角速度过大，鼓励低速平稳运动。
        #   公式：r_vel = -w * sum(qvel^2)，qvel 来自 MuJoCo sim.data.qvel。
        #   注意：qvel 量级可达几~十几，平方后较大，故 w 取很小值。
        #   0.005 → 0.002：适度降低，让策略敢于动关节
        self.w_joint_vel = 0.002

        # 末端速度 r_ee_vel：惩罚 eef 高频抖动（位移变化）。
        #   公式：r_ee_vel = -w * sum((eef_t - eef_{t-1})^2)
        #   旧值 0.50 过大，会连“向 cube 靠近”的正常下降运动一并惩罚，导致
        #   策略倾向于不动。降到 0.10 只抑制抖动。
        #   0.10 → 0.03：实测 ee_vel 均值仅 -0.0005（已几乎不动），说明 0.10
        #   仍过大，让策略不敢下降。降到 0.03 释放运动自由度。
        #   调参：若策略运动迟缓 → 调小；若末端抖动明显 → 调大。
        self.w_ee_vel = 0.03

        # 末端悬浮惩罚 r_z_float：eef 明显高于 cube 时按高度差给予负奖励，
        # 抑制“抬起来就不下来”。
        #   公式：r_z_float = -w * max(0, eef_z - cube_z - threshold)
        #
        # 重要：w_z_float 是引导下降的"关键负梯度来源"，不能降低！
        #   实测 w_z_float=0.40 时 z_float 累计 -25.4，但策略仍悬浮——说明
        #   单纯加大 z_float 没用，必须配合 reach 的强正梯度（已通过降低
        #   reach_tanh_scale 实现）。但若降低 w_z_float，会失去下降方向引导。
        #   0.40 → 0.50：略增，与新 reach 梯度配合形成"下降=正收益"的清晰信号
        #
        # 调参：threshold 过大 → 正常接近被误伤；过小 → 无法抑制悬浮。
        self.w_z_float = 0.50
        self.z_float_threshold = 0.04  # eef 比 cube 高 4cm 以上开始惩罚（米）

        # ==================================================================
        # 行为先验奖励系数（正信号：引导更自然的抓取-抬升行为）
        # 这部分量级很小，仅作"弱引导"，不主导策略。
        # ==================================================================
        # 夹爪运动奖励 r_gripper_move：鼓励夹爪尝试运动，避免僵硬（始终不动）。
        #   公式：r = w * min(|gripper_qpos_t - gripper_qpos_{t-1}|.sum(), max)
        #   用 min 截断防止策略靠"疯狂抖动夹爪"刷分。
        #   调参：若策略夹爪始终不动 → 适当提高 w；若抖动刷分 → 降低 max_gripper_move。
        self.w_gripper_move = 0.02
        self.max_gripper_move = 0.05  # 单步夹爪运动量上限（截断阈值）

    def reset(self, **kwargs):
        """重置环境并初始化所有"上一步"历史状态。

        必须在这里初始化 last_action / last_eef_pos / last_gripper_qpos /
        _cube_rest_z，否则 step() 首步会用 None 参与运算导致报错或 NaN。
        """
        obs, info = self.env.reset(**kwargs)
        # 动作初始化为零向量（与 action_space 同形状），保证首步 action_delta=0
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        # 末端位置/夹爪开度初始化为当前真实值，保证首步 delta=0
        self.last_eef_pos = self._get_eef_pos()
        self.last_gripper_qpos = self._get_gripper_qpos()
        # PBRS 差分 reach 的 last_dist 设为 None：第一步无差分，r_reach=0
        # 真正的 last_dist 在第一步 step() 末尾才会被设置
        self.last_dist = None
        # 记录 cube 静止放置时的中心高度（米），作为 r_lift 的基线。
        # 之后只有 cube 被真正抬起（高于此高度）才给抬升奖励。
        # 必须在 reset 后（cube 已稳定落地）立即采样，避免读到下落中的瞬时高度。
        self._cube_rest_z = float(self._get_cube_pos()[2])
        return obs, info

    def step(self, action):
        """执行一步，叠加奖励塑形后返回。

        流程：先让底层 env 执行动作拿到原始 obs/reward → 计算各分项奖励 →
        合成 shaped_reward → 写入 info → 更新历史状态 → 返回。

        注意：
          - realistic_state 模式：使用自定义奖励塑形，返回 shaped_reward
          - easy 模式：返回 robosuite 内置稠密奖励（reward），仅把 lift.py 的
            原始奖励分量写入 info 用于 TensorBoard 可视化，避免显示与训练无关的
            自定义 shaping 分项（如 reward_smooth、reward_vel 等）。
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 获取底层 robosuite 环境，用于计算任务进度奖励
        base_env = self._get_base_env()

        # easy 模式：只记录 lift.py 的原始奖励分量，不叠加自定义 shaping
        if self.mode == "easy":
            info["lift_reaching_reward"] = float(getattr(base_env, "_last_reaching_reward", 0.0))
            info["lift_grasping_reward"] = float(getattr(base_env, "_last_grasping_reward", 0.0))
            info["lift_success_reward"] = float(getattr(base_env, "_last_success_reward", 0.0))
            info["original_reward"] = reward
            info["shaped_reward"] = reward
            self.last_action = np.array(action, dtype=np.float32)
            return obs, reward, terminated, truncated, info
        eef_pos = self._get_eef_pos()          # 末端位置 (x,y,z)，米
        gripper_qpos = self._get_gripper_qpos()  # 夹爪关节位置
        cube_pos = self._get_cube_pos()         # cube 中心位置 (x,y,z)，米

        # ==================================================================
        # 任务进度奖励（正信号，反映策略是否在完成任务）
        # ==================================================================
        # 1) 接近奖励 r_reach：PBRS（势能差分）形式
        #    势函数 F(s) = -α·tanh(k·dist)，差分奖励 r_reach = α·(tanh(k·d_prev) - tanh(k·d_now))
        #    静止 → 0；接近 → 正；远离 → 负。详见 __init__ 中的注释。
        #    首步 last_dist=None，给 0 避免冷启动偏置。
        #    dist 仍用 _gripper_to_target 计算（绝对距离），差分基于历史值
        dist = base_env._gripper_to_target(
            gripper=base_env.robots[0].gripper,
            target=base_env.cube.root_body,
            target_type="body",
            return_distance=True,
        )
        if self.last_dist is None:
            r_reach = 0.0
        else:
            r_reach = self.w_reach_pbrs * (
                np.tanh(self.reach_pbrs_scale * self.last_dist)
                - np.tanh(self.reach_pbrs_scale * dist)
            )

        # 2) 抓取奖励 r_grasp：二值，夹爪与 cube 任一几何体接触即为 True
        grasped = base_env._check_grasp(
            gripper=base_env.robots[0].gripper,
            object_geoms=base_env.cube.contact_geoms,
        )
        r_grasp = self.w_grasp if grasped else 0.0

        # 3) 抬升奖励 r_lift：cube 当前高度相对静止高度的抬升量
        cube_height = base_env.sim.data.body_xpos[base_env.cube_body_id][2]
        # 基线用 cube 静止放置高度，而非 table_offset[2]（桌面中心）。
        # cube 静止时中心已高于 table_offset[2] 约 2cm，若以 table_offset[2] 为
        # 基线，cube 不动也会拿到 ~0.5/step 的抬升奖励，导致策略无需抓取即可得分。
        # 防御性兜底：若 reset 未正常设置 _cube_rest_z，退回当前高度（使 lift=0）。
        rest_z = self._cube_rest_z if self._cube_rest_z is not None else cube_height
        lift_height = max(0.0, cube_height - rest_z)  # 不允许负值（cube 下沉不算）
        lift_ratio = np.clip(lift_height / self.lift_target_height, 0.0, 1.0)
        r_lift = self.w_lift * lift_ratio

        # ==================================================================
        # 运动质量惩罚（负信号，反映运动是否平稳/合理）
        # 全部取负号，目的是"减分"，量级需小于上方正信号之和。
        # ==================================================================
        # 1) 动作平滑度：相邻两步动作差的平方和（越小越平滑）
        action_delta = action - self.last_action
        r_smooth = -self.w_action_smooth * float(np.sum(action_delta ** 2))

        # 2) 关节速度：关节角速度平方和（鼓励低速）
        joint_vel = self._get_joint_vel()
        r_vel = -self.w_joint_vel * float(np.sum(joint_vel ** 2))

        # 3) 末端速度：末端位移变化平方和（抑制高频抖动，不抑制整体趋近）
        ee_delta = eef_pos - self.last_eef_pos
        r_ee_vel = -self.w_ee_vel * float(np.sum(ee_delta ** 2))

        # 4) 末端悬浮惩罚：eef 比 cube 高出 threshold 以上时按高度差线性惩罚。
        #    只取 z 分量差，max(0,·) 保证 eef 低于 cube 时不惩罚（正常接近阶段）。
        height_above_cube = eef_pos[2] - cube_pos[2]
        r_z_float = -self.w_z_float * max(0.0, height_above_cube - self.z_float_threshold)

        # ==================================================================
        # 行为先验奖励（正信号，弱引导）
        # ==================================================================
        # 夹爪运动量：夹爪关节位置变化的绝对值之和；min 截断防止刷分
        gripper_move = float(np.abs(gripper_qpos - self.last_gripper_qpos).sum())
        r_gripper_move = self.w_gripper_move * min(gripper_move, self.max_gripper_move)

        # ==================================================================
        # 奖励合成：原始 sparse reward + 所有分项
        # 顺序与 REWARD_COMPONENTS 一致：任务进度(+) → 运动质量(-) → 行为先验(+)
        # 注意：reward 在成功时为 2.25，是最终目标；shaping 分项帮助探索。
        # ==================================================================
        shaped_reward = (
            reward
            + r_reach
            + r_grasp
            + r_lift
            + r_smooth
            + r_vel
            + r_ee_vel
            + r_z_float
            + r_gripper_move
        )

        # 将各分项写入 info，供诊断脚本/可视化/analyze_rollouts 读取。
        # 键名必须与模块顶部 REWARD_COMPONENTS 列表保持一致。
        info["original_reward"] = reward
        info["shaped_reward"] = shaped_reward
        info["reward_reach"] = r_reach
        info["reward_grasp"] = r_grasp
        info["reward_lift"] = r_lift
        info["reward_smooth"] = r_smooth
        info["reward_vel"] = r_vel
        info["reward_ee_vel"] = r_ee_vel
        info["reward_z_float"] = r_z_float
        info["reward_gripper_move"] = r_gripper_move

        # 更新历史状态：必须在计算完所有 delta 后再更新，否则本步用了"本步"值
        self.last_action = np.array(action, dtype=np.float32)
        self.last_eef_pos = eef_pos
        self.last_gripper_qpos = gripper_qpos
        # PBRS 差分 reach 需要保留本步 dist 供下一步差分
        self.last_dist = float(dist)

        return obs, shaped_reward, terminated, truncated, info

    def _get_base_env(self):
        """递归解包到底层 robosuite Lift 环境。

        Wrapper 套 Wrapper 时，需层层穿透 .env 才能拿到真正持有 sim 的底层
        环境；缓存 self._base_env 可避免每步重复解包（这里未缓存，因为开销小）。
        """
        env = self.env
        while hasattr(env, "env"):
            env = env.env
        return env

    def _get_eef_pos(self):
        """获取右臂末端执行器位置（世界坐标，米）。

        通过 MuJoCo site（eef_site_id）读取，site_xpos 在每次 sim.step 后由
        MuJoCo 自动更新，无需手动 forward。供 r_ee_vel / r_z_float 使用。
        """
        base_env = self._get_base_env()
        eef_site_id = base_env.robots[0].eef_site_id["right"]
        return np.array(base_env.sim.data.site_xpos[eef_site_id])

    def _get_cube_pos(self):
        """获取 cube 中心位置（世界坐标，米）。

        body_xpos[cube_body_id] 是 cube 根 body 的世界坐标，与 _check_success
        内部使用同一来源，保证奖励与成功判定一致。
        """
        base_env = self._get_base_env()
        return np.array(base_env.sim.data.body_xpos[base_env.cube_body_id])

    def _get_gripper_qpos(self):
        """获取夹爪关节位置（用于 r_gripper_move 与开合比例）。

        通过 _ref_gripper_joint_pos_indexes 取出夹爪关节在 qpos 中的索引，
        再逐个读取。注意返回的是关节位置而非控制器目标。
        """
        base_env = self._get_base_env()
        idx = base_env.robots[0]._ref_gripper_joint_pos_indexes["right"]
        return np.array([base_env.sim.data.qpos[x] for x in idx])

    def _get_gripper_opening_ratio(self):
        """
        获取夹爪开合比例，归一化到 [0, 1]。

        返回:
            float: 0 表示完全闭合，1 表示完全张开。
        """
        base_env = self._get_base_env()
        qpos = self._get_gripper_qpos()
        if len(qpos) == 0:
            return 0.0
        actuator_ids = base_env.robots[0]._ref_joint_gripper_actuator_indexes["right"]
        ratios = []
        for i, act_id in enumerate(actuator_ids):
            lo, hi = base_env.sim.model.actuator_ctrlrange[act_id]
            if hi > lo:
                ratios.append(np.clip((qpos[i] - lo) / (hi - lo), 0.0, 1.0))
        return float(np.mean(ratios)) if ratios else 0.0

    def _get_joint_vel(self):
        """获取机械臂关节速度（用于 r_vel 惩罚）。

        _ref_joint_vel_indexes 是所有可控关节（含手臂与可能的夹爪）在 qvel 中的
        索引。注意这里可能包含夹爪关节速度，若只想惩罚手臂部分需手动过滤。
        """
        base_env = self._get_base_env()
        idx = base_env.robots[0]._ref_joint_vel_indexes
        return np.array([base_env.sim.data.qvel[x] for x in idx])
