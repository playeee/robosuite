"""
本脚本演示如何将 robosuite 环境适配为兼容 Gymnasium API 的形式。

=============================================================================
【背景】为什么需要 GymWrapper（Gym 适配器）？
=============================================================================

Gymnasium（前身为 OpenAI Gym）是强化学习领域事实上的"环境接口标准"。
绝大多数主流 RL 训练库（如 Stable-Baselines3、RLlib、CleanRL、OpenAI Baselines
等）都默认只接受符合 Gymnasium API 的环境。

robosuite 原生 API 与 Gymnasium API 的关键差异：

    ┌──────────────────┬─────────────────────────┬────────────────────────────┐
    │ 项目              │ robosuite 原生 API       │ Gymnasium API               │
    ├──────────────────┼─────────────────────────┼────────────────────────────┤
    │ step() 返回值     │ (obs, reward, done, info)│ (obs, reward, terminated,   │
    │                  │  4 元组                  │  truncated, info) 5 元组     │
    │ reset() 返回值    │ obs                      │ (obs, info) 2 元组           │
    │ 动作空间          │ robosuite 自定义         │ gym.spaces.Box              │
    │ 观测空间          │ OrderedDict              │ gym.spaces.Box / Dict       │
    │ 随机数种子        │ env.seed()               │ env.reset(seed=...)         │
    └──────────────────┴─────────────────────────┴────────────────────────────┘

terminated vs truncated 的区别（Gymnasium 的重要概念）：
    - terminated: 任务自然终止（成功或失败），不应再 bootstrap 未来价值
    - truncated:  因时间限制等外部原因被截断，任务本身未结束，可继续 bootstrap

GymWrapper 的工作原理：装饰器模式（Decorator Pattern）
    在不修改 robosuite 环境源码的前提下，包一层"适配器"，把调用结果
    转换成 Gymnasium 期望的格式。这样就能无缝接入任何 Gym 兼容的训练管线。

=============================================================================
参考代码：以下片段来自 Gymnasium 官方文档"Basic Usage"章节，用于演示
标准 Gym 环境的基本用法。

    import gymnasium as gym
    env = gym.make("LunarLander-v2", render_mode="human")
    observation, info = env.reset()

    for _ in range(1000):
        # 智能体策略：根据 observation 和 info 决定动作
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()
            env.close()

本脚本演示如何通过 GymWrapper 轻松地将 robosuite 的 API 适配为
OpenAI Gym 风格，使其能被上述标准训练流程直接使用。
"""

import robosuite as suite
from robosuite.wrappers import GymWrapper

if __name__ == "__main__":

    # =========================================================================
    # 创建环境：先用 robosuite 原生 suite.make() 构造 Lift 任务环境，
    # 再用 GymWrapper 把它包成 Gymnasium 兼容接口。
    #
    # 注意这种"先 make 再 wrap"的嵌套调用方式——这是适配器模式的典型写法：
    #   内层 suite.make(...)  返回 robosuite 原生 Env
    #   外层 GymWrapper(...)   把原生 Env 适配为 Gym Env
    # =========================================================================
    env = GymWrapper(
        suite.make(
            "Lift",  # 任务名：单臂抬起立方体（robosuite 入门级任务）
            robots="Sawyer",  # 使用 Sawyer 机器人（Rethink Robotics 单臂）
            use_camera_obs=False,  # 不使用像素观测（即不用图像作为 state）
            has_offscreen_renderer=False,  # 无需离屏渲染（因为不用像素观测，可省显存/提速）
            has_renderer=True,  # 启用屏幕渲染，便于人眼观察智能体行为
            reward_shaping=True,  # 使用稠密奖励（dense reward），更利于随机策略探索
            control_freq=20,  # 控制频率 20Hz：每秒 20 次动作输入，仿真看起来流畅
        )
    )

    # 设置随机种子，保证环境初始化（物体放置等）可复现
    env.reset(seed=0)

    # =========================================================================
    # 训练/评估的主循环：跑 20 个 episode
    #
    # 这里用 env.action_space.sample() 随机采样动作（即"随机策略"），
    # 仅用于演示环境 API 是否工作，并非真正的 RL 训练。
    # 真实训练时，应替换为学习算法（如 PPO/SAC）产生的动作。
    # =========================================================================
    for i_episode in range(20):
        # 每个 episode 开始时 reset 环境，返回初始观测
        observation = env.reset()
        for t in range(500):  # 每个 episode 最多 500 步（horizon）
            # 渲染当前帧到屏幕（仅展示用，不参与决策）
            env.render()
            # 从动作空间随机采样一个动作（随机策略）
            action = env.action_space.sample()
            # 执行动作，获取 Gymnasium 标准 5 元组返回值
            observation, reward, terminated, truncated, info = env.step(action)
            # 若任务终止（成功/失败）或被截断（达到 horizon），则结束本 episode
            if terminated or truncated:
                print("Episode finished after {} timesteps".format(t + 1))
                observation, info = env.reset()
                break
        env.close()
