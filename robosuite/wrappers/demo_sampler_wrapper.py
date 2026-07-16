"""
This file contains a wrapper for sampling environment states
from a set of demonstrations on every reset. The main use case is for
altering the start state distribution of training episodes for
learning RL policies.
"""
# =============================================================================
# RL 进阶技术：演示初始化强化学习（Demo-initialized RL / Demonstration-guided RL）
# =============================================================================
# 解决什么问题？
#   纯 RL 从环境的初始状态开始探索，对于操作任务（如抓取、装配）来说：
#     - 初始状态离目标很远，随机探索几乎不可能碰到成功
#     - 稀疏奖励下，策略永远看不到成功信号，无法学习
#
# 核心思想：用人类演示（demonstrations）构造更好的初始状态分布
#   - 从演示轨迹中采样某个中间状态作为 episode 起点
#   - 让策略从"接近成功"的状态开始学习，逐步回退到更难的状态
#   - 相当于"从答案附近开始练习，慢慢学会从头解题"
#
# 这本质上是**课程学习（Curriculum Learning）**的一种形式：
#   - forward scheme: 从演示开头开始，窗口逐步扩大（从简单到难）
#   - reverse scheme: 从演示结尾开始，窗口向前扩大（先学最后一步，再学前置）
#   - uniform scheme: 完全随机采样演示中的状态
#
# 经典应用：
#   - OpenAI Dactyl: 用 reverse curriculum 学会单手转魔方
#   - HER (Hindsight Experience Replay): 类似思想，但用失败轨迹的终点作为伪目标
#   - DDPGfD / DQfD: 把演示数据混入 replay buffer 加速学习
# =============================================================================

import os
import random
import time

import h5py
import numpy as np

from robosuite.wrappers import Wrapper


class DemoSamplerWrapper(Wrapper):
    """
    初始化一个包装器，支持将环境状态重置为演示中的某个状态。
    它还支持课程学习（curriculum），用于调整从演示采样 vs 从环境采样的频率。

    参数:
        env (MujocoEnv): 要包装的环境。

        demo_path (str): 包含演示的文件夹路径。
            该文件夹下应有 `demo.hdf5` 文件和名为 `models` 的文件夹，
            后者包含所有从演示中保存的模型 xml 文件。

        need_xml (bool): 若为 True，从演示采样状态时需要重新加载 mujoco 模型。
            这可能是因为每个演示是在不同的物体属性下采集的。
            此情况下，每个采样的状态都附带一个对应的 xml 用于环境重置。

        num_traj (int): 若提供，从演示集合中子采样指定数量的演示，而非使用全部。

        sampling_schemes (list of str): 要使用的采样方案列表。以下是有效的方案字符串:

                `'random'`: 直接从被包装环境采样一个 reset 状态

                `'uniform'`: 从演示中均匀随机采样一个状态

                `'forward'`: 从演示开头开始，窗口逐步增长的范围内采样

                `'reverse'`: 从演示末尾开始，窗口逐步增长的范围内采样

        scheme_ratios (list of float --> np.array): 分配给 @sampling_schemes 中
            每个成员的概率值列表。必须非负且和为 1。

        open_loop_increment_freq (int): 在开环方案（"forward" 和 "reverse"）中
            增加窗口大小的频率。窗口大小每 @open_loop_increment_freq 次采样
            增加 @open_loop_window_increment。只有开环方案生成的采样才计入此计数。

        open_loop_initial_window_width (int): 开环方案中初始采样窗口的宽度，
            以演示时间步数表示。

        open_loop_window_increment (int): 窗口大小每 @open_loop_increment_freq
            次采样增加 @open_loop_window_increment。此数字以演示时间步数为单位。

    抛出:
        AssertionError: [环境不兼容]
        AssertionError: [无效采样方案]
        AssertionError: [无效方案比例]
    """

    # =========================================================================
    # 【RL 进阶】演示初始化 RL 的完整工作流程
    # =========================================================================
    #
    # 步骤 1：收集人类演示
    #   python robosuite/scripts/collect_human_demonstrations.py \
    #       --environment Lift --robots Panda --controller OSC_POSE \
    #       --episodes 50 --hz 20
    #   这会生成 demo.hdf5 文件，包含 50 条人类操作轨迹
    #
    # 步骤 2：用 DemoSamplerWrapper 包装环境
    #   env = suite.make("Lift", robots="Panda", reward_shaping=True, ...)
    #   env = DemoSamplerWrapper(
    #       env,
    #       demo_path="path/to/demos",
    #       sampling_schemes=("reverse", "random"),  # 反向课程 + 环境采样
    #       scheme_ratios=(0.9, 0.1),                # 90% 演示 + 10% 环境
    #       open_loop_initial_window_width=25,        # 初始窗口 25 步
    #       open_loop_window_increment=25,            # 每次扩大 25 步
    #       open_loop_increment_freq=100,             # 每 100 次采样扩大一次
    #   )
    #
    # 步骤 3：训练 RL 策略
    #   env = GymWrapper(env)
    #   model = SAC("MlpPolicy", env)
    #   model.learn(total_timesteps=500000)
    #
    # 步骤 4：评估
    #   训练完成后，关闭 DemoSamplerWrapper，用纯环境 reset 测试泛化能力
    # =========================================================================

    def __init__(
        self,
        env,
        demo_path,
        need_xml=False,
        num_traj=-1,
        sampling_schemes=("uniform", "random"),
        scheme_ratios=(0.9, 0.1),
        open_loop_increment_freq=100,
        open_loop_initial_window_width=25,
        open_loop_window_increment=25,
    ):
        super().__init__(env)

        # =========================================================================
        # 加载演示数据
        # =========================================================================
        # demo.hdf5 文件结构:
        #   /data/
        #     attrs/env          -> 环境名（用于校验）
        #     attrs/robot        -> 机器人型号
        #     demo_0/            -> 第 0 条演示
        #       attrs/model_file -> 对应的 xml 文件名
        #       states           -> 所有时间步的 MuJoCo 状态 (N, state_dim)
        #       actions          -> 所有时间步的动作 (N, action_dim)
        #     demo_1/
        #     ...
        # =========================================================================
        self.demo_path = demo_path
        hdf5_path = os.path.join(self.demo_path, "demo.hdf5")
        self.demo_file = h5py.File(hdf5_path, "r")

        # 确保被包装的环境与采集演示时的环境一致
        # 不同环境的观测空间、动作空间不同，混用会导致状态不匹配
        # ensure that wrapped env matches the env on which demonstrations were collected
        env_name = self.demo_file["data"].attrs["env"]
        assert (
            env_name == self.unwrapped.__class__.__name__
        ), "Wrapped env {} does not match env on which demos were collected ({})".format(
            env.__class__.__name__, env_name
        )

        # 获取所有演示 episode 的列表
        # list of all demonstrations episodes
        self.demo_list = list(self.demo_file["data"].keys())

        # 若请求，子采样指定数量的演示
        # subsample a selection of demonstrations if requested
        # 用固定种子保证每次运行采样相同的演示子集（可复现性）
        if num_traj > 0:
            random.seed(3141)  # ensure that the same set is sampled every time
            self.demo_list = random.sample(self.demo_list, num_traj)

        self.need_xml = need_xml
        # 计数器：记录已采样的次数，用于开环方案的窗口扩大逻辑
        self.demo_sampled = 0

        # =========================================================================
        # 采样方案到方法名的映射
        # =========================================================================
        # 这种设计模式（策略模式）便于扩展新的采样方案：
        #   - 添加新方案只需实现一个 _xxx_sample 方法
        #   - 在此字典中注册方案名 -> 方法名
        #   - 不需要修改 sample() 方法的核心逻辑
        # =========================================================================
        self.sample_method_dict = {
            "random": "_random_sample",
            "uniform": "_uniform_sample",
            "forward": "_forward_sample_open_loop",
            "reverse": "_reverse_sample_open_loop",
        }

        self.sampling_schemes = sampling_schemes
        self.scheme_ratios = np.asarray(scheme_ratios)

        # 校验：所有方案名必须有效
        # make sure the list of schemes is valid
        schemes = self.sample_method_dict.keys()
        assert np.all([(s in schemes) for s in self.sampling_schemes])

        # 校验：方案列表与比例列表长度一致
        # make sure the distribution is the correct size
        assert len(self.sampling_schemes) == len(self.scheme_ratios)

        # 校验：比例必须非负且和为 1（概率单纯形约束）
        # make sure the distribution lies in the probability simplex
        assert np.all(self.scheme_ratios > 0.0)
        assert sum(self.scheme_ratios) == 1.0

        # 开环方案的配置参数
        # open loop configuration
        self.open_loop_increment_freq = open_loop_increment_freq
        self.open_loop_window_increment = open_loop_window_increment

        # 当前窗口大小（会随训练逐步扩大）
        # keep track of window size
        self.open_loop_window_size = open_loop_initial_window_width

    def reset(self):
        """
        从演示中采样状态并重置仿真到该状态的逻辑。

        返回:
            OrderedDict: reset 后的环境观测空间
        """
        # =========================================================================
        # 【RL 核心】reset 重写：从演示中采样初始状态而非用环境默认初始化
        # =========================================================================
        # 这一步是"演示初始化 RL"的核心——改变了 episode 的起点分布
        #
        # RL 理论基础：
        #   策略性能高度依赖初始状态分布 ρ₀(s₀)
        #   - 若 ρ₀ 总在目标附近，策略容易获得成功信号 -> 学得快
        #   - 若 ρ₀ 总在起点，稀疏奖励下几乎学不到 -> 学得慢
        #
        # 通过演示采样 ρ₀，就是在"作弊式"地构造有利的学习分布：
        #   - 相当于让智能体"站在巨人的肩膀上"开始学习
        #   - 随着课程推进，起点逐步回退到真正的初始状态
        #
        # 这与"逆强化学习"和"Hindsight Experience Replay"有思想上的联系：
        #   - 都是利用现有信息（演示/失败轨迹）来构造更好的学习信号
        # =========================================================================
        state = self.sample()
        if state is None:
            # None 表示应直接从环境采样（_random_sample 方案）
            # None indicates that a normal env reset should occur
            # _random_sample 返回 None，表示用环境默认的初始状态
            return self.env.reset()
        else:
            # 若需要重新加载模型（因为演示可能在不同物体属性下采集）
            if self.need_xml:
                # reset the simulation from the model if necessary
                state, xml = state
                self.env.reset_from_xml_string(xml)

            if isinstance(state, tuple):
                state = state[0]

            # -------------------------------------------------------------------------
            # 直接设置 MuJoCo 仿真器的内部状态为演示中的某个时刻
            # -------------------------------------------------------------------------
            # set_state_from_flattened:
            #   - 输入是扁平化的状态向量（包含所有关节位置、速度等）
            #   - MuJoCo 会据此恢复完整的仿真状态
            #
            # forward:
            #   - 重新计算所有派生量（如笛卡尔位姿、接触点、雅可比等）
            #   - 必须调用，否则观测可能与状态不一致
            #   - 类似于"正向传播一次"以更新缓存
            #
            # 为什么这样能工作？
            #   - MuJoCo 的状态是完全确定性的（给定状态+动作，下一步完全确定）
            #   - 只要恢复状态向量，仿真器就"穿越"到那个时刻
            #   - 然后策略从那个时刻开始决策，相当于"从中途开始 episode"
            # -------------------------------------------------------------------------
            self.sim.set_state_from_flattened(state)
            self.sim.forward()

            # 返回新状态对应的观测
            return self.env._get_observation()

    def sample(self):
        """
        核心采样方法。根据配置从演示中采样一个状态。

        返回:
            None or np.array or 2-tuple:
                - None: 表示用环境默认 reset
                - np.array: 从演示文件采样的状态
                - 2-tuple: 状态 + 模型 xml 文件
        """

        # =========================================================================
        # 【课程学习关键】按比例混合不同采样方案
        # =========================================================================
        # 例如 scheme_ratios=(0.9, 0.1) 表示 90% 从演示采样 + 10% 从环境采样
        #
        # 为什么需要混合？
        #   - 纯演示初始化会导致策略过拟合于演示附近的状态
        #     （策略只学会从"接近成功"的状态开始，遇到真正初始状态就失败）
        #   - 混合一定比例的环境 reset 能提升泛化能力
        #   - 类似于"训练时既做简单题也做难题"
        #
        # RL 经验：
        #   - 入门阶段：高比例演示（0.9 演示 + 0.1 环境），快速建立基础行为
        #   - 进阶阶段：逐步降低演示比例（0.5 + 0.5），提升泛化能力
        #   - 最终评估：完全关闭演示（0.0 + 1.0），测试真实性能
        #
        # 实现原理（累积分布函数法）:
        #   假设 scheme_ratios = (0.6, 0.3, 0.1)
        #   累积分布:          (0.6, 0.9, 1.0)
        #   生成随机数 seed ∈ [0, 1)
        #   找到第一个 > seed 的累积值，对应即为选中的方案
        # =========================================================================
        # chooses a sampling scheme randomly based on the mixing ratios
        seed = random.uniform(0, 1)
        ratio = np.cumsum(self.scheme_ratios)
        ratio = ratio > seed
        for i, v in enumerate(ratio):
            if v:
                break

        # 根据选中的方案名，调用对应的采样方法
        # getattr 动态获取方法，体现了策略模式的灵活性
        sample_method = getattr(self, self.sample_method_dict[self.sampling_schemes[i]])
        return sample_method()

    def _random_sample(self):
        """
        采样方法。

        返回 None 表示应直接从环境采样状态。
        """
        # =========================================================================
        # 方案 1：纯环境 reset（不使用演示）
        # =========================================================================
        # 用于让策略接触"真实"的初始状态分布，防止过拟合
        #
        # 在混合方案中扮演"现实检验"的角色：
        #   - 即使大部分时间从演示采样，也要保留一定比例的真实 reset
        #   - 否则策略可能只在演示附近状态有效，泛化能力差
        #
        # 返回 None 是一个约定：
        #   - reset() 方法检测到 None 就调用 env.reset()
        #   - 这种"空对象模式"避免了返回一个特殊状态值
        # =========================================================================
        return None

    def _uniform_sample(self):
        """
        采样方法。

        首先从演示集合中均匀采样一条演示。
        然后从选中的演示中均匀采样一个状态。

        返回:
            np.array or 2-tuple:
                - np.array: 从演示文件采样的状态
                - 2-tuple: 状态 + 模型 xml 文件
        """
        # =========================================================================
        # 方案 2：均匀采样演示中的任意状态
        # =========================================================================
        # 难度居中：可能采到简单的（接近成功），也可能采到难的（远离成功）
        # 适合任务本身不太难、演示质量参差不齐的情况
        #
        # 与 reverse/forward 的区别：
        #   - uniform: 所有状态等概率被采样，没有课程结构
        #   - reverse: 从末尾开始，由易到难
        #   - forward: 从开头开始，由简到繁
        #
        # 适用场景：
        #   - 任务没有明显的"由易到难"结构
        #   - 演示质量不一，想充分利用所有数据
        #   - 作为 baseline 对比课程学习的效果
        # =========================================================================

        # 随机选一条演示 episode
        # get a random episode index
        ep_ind = random.choice(self.demo_list)

        # 从该 episode 中均匀随机选一个时间步的状态
        # select a flattened mujoco state uniformly from this episode
        states = self.demo_file["data/{}/states".format(ep_ind)][()]
        state = random.choice(states)

        # 若需要 xml（演示在不同模型参数下采集），一并返回
        if self.need_xml:
            model_xml = self._xml_for_episode_index(ep_ind)
            xml = self.env.edit_model_xml(model_xml)
            return state, xml
        return state

    def _reverse_sample_open_loop(self):
        """
        采样方法。

        从演示中开环反向采样。开始时从演示末尾附近的状态采样。
        随着调用次数增加，窗口以固定速率向后扩大。

        返回:
            np.array or 2-tuple:
                - np.array: 从演示文件采样的状态
                - 2-tuple: 状态 + 模型 xml 文件
        """
        # =========================================================================
        # 方案 3：反向课程学习（reverse curriculum）——最经典的 demo-initialized RL
        # =========================================================================
        #
        # ┌─────────────────────────────────────────────────────────────────────┐
        # │ 学习顺序图示（演示轨迹的时间轴）:                                       │
        # │                                                                     │
        # │   时间步:  0 ---- 25 ---- 50 ---- 75 ---- 100 (成功)                 │
        # │           |---------|---------|---------|---------|                  │
        # │                                                                     │
        # │   阶段1:                   [===========|====]  窗口=[75,100]         │
        # │   阶段2:          [====================|========]  窗口=[50,100]      │
        # │   阶段3:  [==============================================]  窗口=[0,100]│
        # │                                                                     │
        # │   起点从末尾逐步向前扩展，难度从易到难                                 │
        # └─────────────────────────────────────────────────────────────────────┘
        #
        # 学习顺序：先从演示末尾（接近成功）开始 → 逐步扩大窗口向前 → 最终学会从起点开始
        #
        # 直觉类比：学游泳时先在浅水区（接近终点），熟练后逐步到深水区（远离终点）
        #
        # 为什么有效？
        #   - 末尾状态离成功最近，策略容易获得正奖励，快速建立基础行为
        #   - 窗口逐步扩大，策略必须学会处理越来越早的状态
        #   - 这是一种"由易到难"的课程，符合循序渐进的学习规律
        #
        # open_loop 含义：
        #   - 窗口扩大不依赖策略性能，而是按固定频率扩大
        #   - 优点：实现简单，无需监控策略性能
        #   - 缺点：可能窗口扩大太快（策略还没学会简单状态就遇到难的）
        #
        # 对比 closed_loop：
        #   - 根据策略成功率动态调整难度
        #   - 成功率高 -> 扩大窗口；成功率低 -> 缩小窗口
        #   - 更复杂但更自适应，参见论文 "Reverse Curriculum Generation for RL"
        # =========================================================================

        # 随机选一条演示 episode
        # get a random episode index
        ep_ind = random.choice(self.demo_list)

        # -------------------------------------------------------------------------
        # 从演示末尾开始，在 [eps_len - window_size, eps_len] 范围内均匀采样
        # -------------------------------------------------------------------------
        # sample uniformly in a window that grows backwards from the end of the demos
        # 窗口范围：[eps_len - window_size, eps_len]
        # window_size 从 open_loop_initial_window_width 开始，逐步增大到 eps_len
        #
        # 例如 eps_len=100, window_size=25:
        #   采样范围 = [75, 100]，即从演示最后 25 步中随机选一个状态
        #   这些状态接近成功，策略容易获得正奖励
        # -------------------------------------------------------------------------
        states = self.demo_file["data/{}/states".format(ep_ind)][()]
        eps_len = states.shape[0]
        index = np.random.randint(max(eps_len - self.open_loop_window_size, 0), eps_len)
        state = states[index]

        # -------------------------------------------------------------------------
        # 开环窗口扩大逻辑
        # -------------------------------------------------------------------------
        # increase window size at a fixed frequency (open loop)
        # 每采样 open_loop_increment_freq 次，窗口扩大 open_loop_window_increment 步
        #
        # 例如 increment_freq=100, window_increment=25:
        #   - 前 100 次采样：窗口=25（简单）
        #   - 第 101-200 次：窗口=50（中等）
        #   - 第 201-300 次：窗口=75（较难）
        #   - 第 301 次起：  窗口=100（完整难度）
        #
        # 这种"阶梯式"课程让策略有充分时间在简单阶段学习
        # -------------------------------------------------------------------------
        self.demo_sampled += 1
        if self.demo_sampled >= self.open_loop_increment_freq:
            if self.open_loop_window_size < eps_len:
                self.open_loop_window_size += self.open_loop_window_increment
            self.demo_sampled = 0

        if self.need_xml:
            model_xml = self._xml_for_episode_index(ep_ind)
            xml = self.env.edit_model_xml(model_xml)
            return state, xml

        return state

    def _forward_sample_open_loop(self):
        """
        采样方法。

        从演示中开环正向采样。开始时从演示开头附近的状态采样。
        随着调用次数增加，窗口以固定速率向前扩大。

        返回:
            np.array or 2-tuple:
                - np.array: 从演示文件采样的状态
                - 2-tuple: 状态 + 模型 xml 文件
        """
        # =========================================================================
        # 方案 4：正向课程学习（forward curriculum）
        # =========================================================================
        #
        # ┌─────────────────────────────────────────────────────────────────────┐
        # │ 学习顺序图示（演示轨迹的时间轴）:                                       │
        # │                                                                     │
        # │   时间步:  0 ---- 25 ---- 50 ---- 75 ---- 100 (成功)                 │
        # │           |---------|---------|---------|---------|                  │
        # │                                                                     │
        # │   阶段1:  [====|=======]                                       窗口=[0,25]   │
        # │   阶段2:  [================|================]                  窗口=[0,50]   │
        # │   阶段3:  [================================|==============]   窗口=[0,75]   │
        # │   阶段4:  [==================================================] 窗口=[0,100]  │
        # │                                                                     │
        # │   终点从未尾逐步向前扩展，从开头开始的任务范围越来越大                   │
        # └─────────────────────────────────────────────────────────────────────┘
        #
        # 与 reverse 相反：从演示开头开始，窗口向后扩大
        #
        # 适用场景：
        #   - 任务前半部分简单（如接近物体），后半部分难（如精细装配）
        #   - 先掌握前半段，再逐步学习更靠后的状态
        #
        # 对比 reverse:
        #   - reverse: 从成功状态倒推，适合"结果导向"的任务（如抬起、放置）
        #   - forward:  从初始状态顺推，适合"过程导向"的任务（如轨迹跟踪、装配）
        #
        # 实际中使用频率：reverse > forward > uniform
        # =========================================================================

        # 随机选一条演示 episode
        # get a random episode index
        ep_ind = random.choice(self.demo_list)

        # 从演示开头开始，在 [0, window_size] 范围内均匀采样
        # sample uniformly in a window that grows forwards from the beginning of the demos
        # 窗口范围：[0, window_size]
        states = self.demo_file["data/{}/states".format(ep_ind)][()]
        eps_len = states.shape[0]
        index = np.random.randint(0, min(self.open_loop_window_size, eps_len))
        state = states[index]

        # 开环窗口扩大逻辑（与 reverse 相同）
        # increase window size at a fixed frequency (open loop)
        self.demo_sampled += 1
        if self.demo_sampled >= self.open_loop_increment_freq:
            if self.open_loop_window_size < eps_len:
                self.open_loop_window_size += self.open_loop_window_increment
            self.demo_sampled = 0

        if self.need_xml:
            model_xml = self._xml_for_episode_index(ep_ind)
            xml = self.env.edit_model_xml(model_xml)
            return state, xml

        return state

    def _xml_for_episode_index(self, ep_ind):
        """
        辅助方法：根据 episode 索引获取对应的模型 xml 字符串。

        参数:
            ep_ind (int): 从演示文件中提取的 episode 索引

        返回:
            str: 模型 xml 字符串
        """
        # 读取模型 xml，使用该 episode 属性中存储的元数据
        # 每个 demo 可能对应不同的模型 xml（如不同的物体属性、桌面尺寸等）
        # read the model xml, using the metadata stored in the attribute for this episode
        model_file = self.demo_file["data/{}".format(ep_ind)].attrs["model_file"]
        model_path = os.path.join(self.demo_path, "models", model_file)
        with open(model_path, "r") as model_f:
            model_xml = model_f.read()
        return model_xml
