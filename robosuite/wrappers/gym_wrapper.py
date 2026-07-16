"""
本文件实现了一个 Wrapper，用于让 robosuite 兼容 OpenAI Gym / Gymnasium API。
当我们需要用假设 gym-like 接口的代码（如 Stable-Baselines3）来训练时，这个包装器非常有用。
"""
# =============================================================================
# 【RL 工程基础】GymWrapper——连接 robosuite 与主流 RL 库的桥梁
# =============================================================================
#
# 为什么需要 GymWrapper？
#   主流 RL 库（Stable-Baselines3 / CleanRL / rlkit / spinningup）都基于 Gym API，
#   而 robosuite 原生 API 与 Gym API 有三处不兼容：
#
#   1. 观测格式不同：
#      - robosuite 返回 OrderedDict（按模态分组，如 robot0_joint_pos, object-state）
#      - Gym API 要求返回 numpy 数组（或字典数组，用于多模态策略）
#
#   2. 空间定义缺失：
#      - Gym API 要求明确定义 observation_space 和 action_space
#      - 这两个空间是构建策略网络的依据（输入/输出维度）
#      - robosuite 原生不显式提供，需要从 env.action_spec 推导
#
#   3. step() 返回值元组长度不同：
#      - robosuite 原生: (obs, reward, done, info)            -> 4 元组
#      - Gym API (新版):  (obs, reward, terminated, truncated, info) -> 5 元组
#      - 多出来的 truncated 表示"外部截断"（如超时），与"任务自然结束"区分
#
# 通过这层适配，可以直接用以下代码训练 RL 策略：
#      from stable_baselines3 import SAC
#      env = GymWrapper(suite.make("Lift", robots="Panda", ...))
#      model = SAC("MlpPolicy", env, verbose=1)
#      model.learn(total_timesteps=100000)
#
# RL 中的两个核心空间概念：
#   - observation_space : 描述观测的形状和取值范围，决定策略网络输入维度
#   - action_space      : 描述动作的形状和取值范围，决定策略网络输出维度
#     robosuite 的动作空间由控制器决定（OSC_POSE=7维, JOINT_VELOCITY=按关节数）
# =============================================================================

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    # gym 和 gymnasium 的大部分 API 兼容
    # Most APIs between gym and gymnasium are compatible
    print("WARNING! gymnasium is not installed. We will try to use openai gym instead.")
    import gym
    from gym import spaces

    if not gym.__version__ >= "0.26.0":
        # 由于 gym>=0.26.0 有 API 变更，需要确保版本正确
        # 请查阅: https://github.com/openai/gym/releases/tag/0.26.0
        # Due to API Changes in gym>=0.26.0, we need to ensure that the version is correct
        raise ImportError("Please ensure version of gym>=0.26.0 to use the GymWrapper.")

from robosuite.wrappers import Wrapper


class GymWrapper(Wrapper, gym.Env):
    metadata = None
    render_mode = None
    """
    初始化 Gym 包装器。模仿 gym.core 模块中 Wrapper 类所需的许多功能。

    参数:
        env (MujocoEnv): 要包装的环境。
        keys (None or list of str): 若提供，每个观测将由被包装环境观测字典中
            指定键拼接而成。默认为 proprio-state 和 object-state。
        flatten_obs (bool): 是否将观测字典展平为一维数组。默认为 True。

    抛出:
        AssertionError: [若未启用物体观测且未提供 keys]
    """

    def __init__(self, env, keys=None, flatten_obs=True):
        # 调用父类初始化
        # Run super method
        super().__init__(env=env)

        # 为 gym 创建名称（机器人型号 + 任务名）
        # Create name for gym
        robots = "".join([type(robot.robot_model).__name__ for robot in self.env.robots])
        self.name = robots + "_" + type(self.env).__name__

        # =========================================================================
        # reward_range 告诉 RL 算法奖励的取值范围，有助于归一化回报（return）
        # =========================================================================
        # 例如 SAC 等算法会根据奖励量级调整学习率：
        #   - 奖励量级 0.01：学习率要大，否则梯度太小
        #   - 奖励量级 100：学习率要小，否则梯度爆炸
        # reward_scale 默认为 1.0，所以 reward_range 通常是 (0, 1)
        # =========================================================================
        self.reward_range = (0, self.env.reward_scale)

        # =========================================================================
        # 选择观测的键值——这是 RL 中"状态表示"的关键设计
        # =========================================================================
        # RL 中"给策略网络看什么"决定了它能学到什么：
        #   - 本体感受 (proprio-state): 关节角度、速度、夹爪状态等机器人自身状态
        #   - 物体状态 (object-state):  物体的位姿、速度等外部状态
        #   - 图像观测 (image):         相机拍摄的像素图像
        #
        # 默认包含：
        #   - object-state（物体位姿）：任务相关的外部信息
        #   - robot0_proprio-state（本体感受）：机器人自身状态
        #
        # RL 经验法则：
        #   - 低维状态（本体感受+物体位姿）比图像观测学得快得多，入门首选
        #   - 图像观测需要 CNN + 大量数据，样本效率低 1-2 个数量级
        # =========================================================================
        if keys is None:
            keys = []
            # 若启用了物体观测，添加 object-state
            # Add object obs if requested
            if self.env.use_object_obs:
                keys += ["object-state"]
            # 若启用了相机观测，添加所有相机的图像
            # Add image obs if requested
            # 注意：图像观测会让问题变成"视觉强化学习"，样本效率极低，需要 CNN + 大量数据
            if self.env.use_camera_obs:
                keys += [f"{cam_name}_image" for cam_name in self.env.camera_names]
            # 遍历所有机器人，添加本体感受状态
            # Iterate over all robots to add to state
            for idx in range(len(self.env.robots)):
                keys += ["robot{}_proprio-state".format(idx)]
        self.keys = keys

        # Gym 特定属性
        # Gym specific attributes
        self.env.spec = None

        # 准备观测和动作空间，需要先 reset 一次获取样本观测
        # set up observation and action spaces
        obs = self.env.reset()

        # =========================================================================
        # 是否展平观测空间——决定了策略网络的结构
        # =========================================================================
        # flatten_obs=True（默认）:
        #   - 把所有观测拼成一维向量，形状如 (32,)
        #   - 适配 MLP（多层感知机）策略网络
        #   - 训练快，适合低维状态 RL
        #
        # flatten_obs=False:
        #   - 保留字典结构，如 {"image": (84,84,3), "state": (20,)}
        #   - 适配多模态策略网络（CNN 处理图像 + MLP 处理状态）
        #   - 适合视觉 RL 或多传感器融合
        # =========================================================================
        self.flatten_obs: bool = flatten_obs

        if self.flatten_obs:
            # -----------------------------------------------------------------
            # 构建一维观测空间：形状为 (obs_dim,)，取值范围 (-inf, inf)
            # -----------------------------------------------------------------
            # Box 空间表示连续值的盒形空间，是 RL 中最常用的观测/动作空间类型
            #
            # RL 注意：实际训练时通常会对观测做归一化（running mean/std）以稳定训练
            #   - 原因：不同观测分量量级差异大（如关节角度 vs 像素值）
            #   - 方法：用 RunningMeanStd 估算统计量，实时归一化
            #   - 库：stable_baselines3.common.vec_env.VecNormalize
            # -----------------------------------------------------------------
            flat_ob = self._flatten_obs(obs)
            self.obs_dim = flat_ob.size
            high = np.inf * np.ones(self.obs_dim)
            low = -high
            self.observation_space = spaces.Box(low, high)
        else:
            # -----------------------------------------------------------------
            # 字典观测空间：每个键对应一个 Box 空间
            # -----------------------------------------------------------------
            # 用于多模态输入，例如：
            #   observation_space = {
            #       "robot0_proprio-state": Box(shape=(20,)),
            #       "agentview_image":      Box(shape=(84, 84, 3)),
            #   }
            # 策略网络需要分别处理不同模态（CNN for image, MLP for state）
            # -----------------------------------------------------------------

            def get_box_space(sample):
                """工具函数：从单个 numpy 样本数据获取对应的 Box 空间"""
                if np.issubdtype(sample.dtype, np.integer):
                    # 整数类型（如像素值 0-255）
                    low = np.iinfo(sample.dtype).min
                    high = np.iinfo(sample.dtype).max
                elif np.issubdtype(sample.dtype, np.inexact):
                    # 浮点类型（如关节角度，范围不限）
                    low = float("-inf")
                    high = float("inf")
                else:
                    raise ValueError()
                return spaces.Box(low=low, high=high, shape=sample.shape, dtype=sample.dtype)

            self.observation_space = spaces.Dict({key: get_box_space(obs[key]) for key in self.keys})

        # =========================================================================
        # 动作空间：Box 类型，连续动作（robosuite 都是连续控制）
        # =========================================================================
        # 动作维度由控制器决定：
        #   - OSC_POSE:       6 (位姿: 3平移+3旋转) + 1 (夹爪) = 7
        #   - OSC_POSITION:   3 (位置: 3平移)       + 1 (夹爪) = 4
        #   - JOINT_VELOCITY: 关节数                 + 1 (夹爪)
        #   - JOINT_TORQUE:   关节数                 + 1 (夹爪)
        #
        # RL 中连续动作用高斯策略输出：
        #   - 策略网络输出均值 μ(s) 和标准差 σ(s)
        #   - 动作采样: a ~ N(μ, σ²)
        #   - 再用 action_space 的 low/high 裁剪到合法范围
        # =========================================================================
        low, high = self.env.action_spec
        self.action_space = spaces.Box(low, high)

    def _flatten_obs(self, obs_dict, verbose=False):
        """
        过滤感兴趣的键，并将信息拼接起来。

        参数:
            obs_dict (OrderedDict): 有序观测字典
            verbose (bool): 是否在控制台打印处理的观测键

        返回:
            np.array: 展平为一维数组的观测
        """
        # 将字典中所有感兴趣的观测值展平并拼接成一维向量
        # 这是从"字典观测"到"向量观测"的转换，是 Gym 适配的核心步骤
        ob_lst = []
        for key in self.keys:
            if key in obs_dict:
                if verbose:
                    print("adding key: {}".format(key))
                ob_lst.append(np.array(obs_dict[key]).flatten())
        return np.concatenate(ob_lst)

    def _filter_obs(self, obs_dict) -> dict:
        """
        从观测字典中过滤出感兴趣的键，返回过滤后的字典。
        """
        # 保留感兴趣的键，丢弃其他（用于 flatten_obs=False 的情况）
        return {key: obs_dict[key] for key in self.keys if key in obs_dict}

    def reset(self, seed=None, options=None):
        """
        扩展父类的 reset 方法，返回观测（而非原生的 OrderedDict），并可选地重置随机种子。

        返回:
            2-tuple:
                - (np.array) 环境观测
                - (dict) 空字典，作为标准返回格式的一部分
        """
        # =========================================================================
        # seed 用于复现实验：相同 seed + 相同策略 -> 相同轨迹
        # =========================================================================
        # 这是 RL 论文复现的关键：
        #   - 环境的随机性（如物体初始位姿）由 np.random 控制
        #   - 设置相同 seed 后，每次 reset 产生的初始状态序列完全相同
        #   - 配合策略网络的固定初始化，可完整复现实验结果
        #
        # 注意：深度学习框架（PyTorch）的随机种子需要单独设置
        # =========================================================================
        if seed is not None:
            if isinstance(seed, int):
                np.random.seed(seed)
            else:
                raise TypeError("Seed must be an integer type!")
        ob_dict = self.env.reset()

        # 把字典观测转为数组（或过滤后的字典），符合 Gym API
        obs = self._flatten_obs(ob_dict) if self.flatten_obs else self._filter_obs(ob_dict)

        # Gym API: reset 返回 (obs, info)
        # robosuite 原生只返回 obs，这里补一个空字典以满足 Gym 规范
        return obs, {}

    def step(self, action):
        """
        扩展原生的 step() 函数，返回观测（而非原生的 OrderedDict）。

        参数:
            action (np.array): 要在环境中执行的动作

        返回:
            5-tuple:
                - (np.array) 环境观测
                - (float) 环境奖励
                - (bool) 到达环境终止状态后 episode 结束
                - (bool) 外部定义条件导致的 episode 结束
                - (dict) 杂项信息
        """
        # =========================================================================
        # RL 核心交互：执行 action，环境返回 (obs, reward, terminated, info)
        # =========================================================================
        # robosuite 原生返回 4 元组，Gym API 要求 5 元组（多一个 truncated）
        #
        # terminated vs truncated 的区别（Gym >= 0.26 引入）：
        #   - terminated: 任务自然结束
        #     * 成功完成（_check_success 返回 True）
        #     * 失败（如物体掉落）
        #     * 由环境内在逻辑决定
        #
        #   - truncated: 外部强制截断
        #     * 超过最大步数（horizon）
        #     * 人为中断
        #     * 与任务完成无关
        #
        # 为什么区分？因为在 RL 算法中（如 TD-error 计算）：
        #   - terminated=True:   不 bootstrap 未来价值（V(s')=0）
        #   - truncated=True:    仍 bootstrap 未来价值（V(s') 用当前估计）
        # =========================================================================
        ob_dict, reward, terminated, info = self.env.step(action)
        obs = self._flatten_obs(ob_dict) if self.flatten_obs else self._filter_obs(ob_dict)

        # 本框架固定返回 truncated=False，因为 robosuite 用 horizon 内部控制 episode 长度
        return obs, reward, terminated, False, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        为了兼容 gym 接口的占位函数，直接返回环境奖励。

        参数:
            achieved_goal: [未使用]
            desired_goal: [未使用]
            info: [未使用]

        返回:
            float: 环境奖励
        """
        # 这个函数是为兼容 GoalEnv 接口而存在的占位符
        # GoalEnv 要求实现 compute_reward(achieved_goal, desired_goal, info)
        # robosuite 的奖励在 env.reward() 中实现，这里直接转发
        # 占位参数用于模仿 Wrapper 接口
        return self.env.reward()

    def close(self):
        """
        调用底层 env close 函数的包装器。
        """
        self.env.close()
