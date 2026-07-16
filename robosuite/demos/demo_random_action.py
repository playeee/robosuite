import time

from robosuite.robots import MobileRobot
from robosuite.utils.input_utils import *

MAX_FR = 25  # 最大帧率，用于限制仿真渲染速度（让肉眼能看清动作）

# =============================================================================
# 【RL 入门第一课】用随机策略与环境交互——理解 RL 的基本循环
# =============================================================================
#
# 什么是强化学习（RL）？
#   RL 研究的是"智能体（agent）通过与环境（environment）交互来学习最优策略"。
#   智能体在每一步观察环境状态（observation），选择一个动作（action），
#   环境根据动作返回奖励（reward）和新状态。智能体的目标是最大化累计奖励。
#
# RL 的核心循环可以概括为以下五步（本脚本完整演示了这个循环）：
#
#   ┌──────────────────────────────────────────────────────────┐
#   │  1. env.reset()         -> 获取初始观测 (observation)      │
#   │  2. policy(obs)         -> 根据观测选择动作 (action)        │
#   │  3. env.step(action)    -> 得到 next_obs, reward, done     │
#   │  4. env.render()        -> 可视化（仅观察用，训练可关）       │
#   │  5. 若 done 则 reset，否则回到第 2 步                        │
#   └──────────────────────────────────────────────────────────┘
#
# 本脚本的特殊之处：
#   - 第 2 步用"随机高斯噪声"代替策略网络，是 RL 中最弱的 baseline（随机策略）
#   - 真实训练时只需把第 2 步替换为：action = policy_network(obs)
#   - 并在循环中累积 (obs, action, reward, next_obs) 用于更新策略网络
#
# 学习建议：
#   先跑这个脚本观察机器人"乱动"的样子，再思考——
#   如果策略能根据 obs 智能地选择 action，机器人就能学会完成任务。
#   这就是 RL 要解决的核心问题。
# =============================================================================

if __name__ == "__main__":

    # 创建字典，保存将传递给环境创建函数的所有选项
    # Create dict to hold options that will be passed to env creation call
    options = {}

    # 打印欢迎信息
    # print welcome info
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    # 选择环境（任务），并添加到选项字典
    # Choose environment and add it to options
    options["env_name"] = choose_environment()

    # 若选择了双臂环境，需要选择配置和相应的机器人
    # If a multi-arm environment has been chosen, choose configuration and appropriate robot(s)
    if "TwoArm" in options["env_name"]:
        # 选择环境配置（并行/串联/双臂单机等）并添加到选项
        # Choose env config and add it to options
        options["env_configuration"] = choose_multi_arm_config()

        # 若配置为"单机器人双臂"（bimanual），必须使用 Baxter 机器人
        # 否则让用户分别选择两个机器人
        if options["env_configuration"] == "single-robot":
            options["robots"] = choose_robots(exclude_bimanual=False, use_humanoids=True, exclude_single_arm=True)
        else:
            options["robots"] = []

            # 让用户依次选择两个机器人
            # Have user choose two robots
            for i in range(2):
                print("Please choose Robot {}...\n".format(i))
                options["robots"].append(choose_robots(exclude_bimanual=False, use_humanoids=True))

    # 若选择了人形机器人环境，选择对应的人形机器人
    # If a humanoid environment has been chosen, choose humanoid robots
    elif "Humanoid" in options["env_name"]:
        options["robots"] = choose_robots(use_humanoids=True)
    else:
        # 单臂环境，让用户选择任意机器人
        options["robots"] = choose_robots(exclude_bimanual=False, use_humanoids=True)

    # =========================================================================
    # 【关键】初始化 RL 环境——这里决定了 RL 训练的"游戏规则"
    # =========================================================================
    # RL 环境初始化的关键参数说明：
    #
    #   has_renderer=True            -> 开启可视化窗口
    #      训练时通常设为 False，因为渲染会消耗大量计算资源
    #      调试和演示时设为 True 以观察机器人行为
    #
    #   has_offscreen_renderer=False -> 关闭离屏渲染
    #      离屏渲染用于生成图像观测（像素输入的 RL）
    #      不使用图像观测时关闭可大幅提速（10-100倍）
    #
    #   ignore_done=True             -> 忽略 done 信号
    #      让 episode 一直跑下去（本演示用，避免频繁 reset）
    #      真实训练时通常设为 False，让 episode 在成功/失败时结束
    #
    #   use_camera_obs=False         -> 不使用图像作为观测
    #      使用低维物理状态（关节角度、物体位姿等），RL 入门强烈推荐
    #      图像观测会让问题变成"视觉强化学习"，样本效率极低
    #
    #   control_freq=20              -> 控制频率 20Hz
    #      即每秒 20 个动作，每个动作间隔 50ms
    #      这决定了 RL 的离散时间步：一个 episode 有多少个决策点
    #      频率太高：动作精细但训练样本爆炸
    #      频率太低：动作粗糙，难以完成精细任务
    # =========================================================================
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )

    # RL 循环第 1 步：重置环境，获取初始观测
    # reset() 会把机器人归位、物体放回初始位置、计数器清零
    # 返回值是初始观测（本例不接收，因为后面用随机动作不需要观测）
    env.reset()
    env.viewer.set_camera(camera_id=0)

    # 对于移动机器人（如 PandaOmron），关闭腿部和底座控制
    # 这样脚本只控制机械臂部分，简化演示
    for robot in env.robots:
        if isinstance(robot, MobileRobot):
            robot.enable_parts(legs=False, base=False)

    # =========================================================================
    # 【核心】RL 主循环——这里就是 RL 训练的骨架
    # =========================================================================
    # 循环 10000 步，相当于一个超长 episode
    # 真实训练中，每 episode 长度由 horizon 参数控制（通常 200-1000 步）
    # do visualization
    for i in range(10000):
        start = time.time()

        # -------------------------------------------------------------------------
        # RL 循环第 2 步：选择动作（这是 RL 的核心——策略决策）
        # -------------------------------------------------------------------------
        # 这里用 np.random.randn 生成标准正态分布的随机动作
        #   - 这是 RL 中最弱的 baseline（随机策略），用于验证环境是否正常
        #   - 真实训练时替换为: action = policy_network(obs).sample()
        #
        # env.action_spec() 返回 (low, high) 元组，描述动作空间的上下界：
        #   - low  : 动作下界数组（通常是 -1）
        #   - high : 动作上界数组（通常是 +1）
        #   - low.shape 即为动作维度
        #
        # 动作维度由控制器决定（见 robosuite/controllers/）：
        #   - OSC_POSE:      6 (3平移+3旋转) + 1 (夹爪) = 7 维
        #   - OSC_POSITION:  3 (3平移)       + 1 (夹爪) = 4 维
        #   - JOINT_VELOCITY: 关节数         + 1 (夹爪)
        #   - JOINT_TORQUE:   关节数         + 1 (夹爪)
        #
        # RL 中连续动作用高斯策略输出（均值+方差），再用 action_space 归一化
        # -------------------------------------------------------------------------
        action = np.random.randn(*env.action_spec[0].shape)

        # -------------------------------------------------------------------------
        # RL 循环第 3 步：执行动作，环境返回四元组
        # -------------------------------------------------------------------------
        # 这是 RL 中最重要的一步——智能体与环境交互的瞬间
        #
        # 返回值含义：
        #   obs    : 下一时刻的观测（本体感受 + 物体状态等）
        #            例：robot0_joint_pos（关节角度）, object-state（物体位姿）
        #
        #   reward : 标量奖励，衡量这一步的好坏
        #            参见 lift.py 中的 reward 函数设计（稀疏 vs 稠密）
        #
        #   done   : episode 是否结束
        #            本例 ignore_done=True 故恒为 False
        #            真实训练中，done=True 时需要 reset 环境开始新 episode
        #
        #   info   : 额外信息字典（调试用，不参与训练）
        #            例：{"grasped": True, "cube_pos": [x,y,z]}
        # -------------------------------------------------------------------------
        obs, reward, done, _ = env.step(action)

        # -------------------------------------------------------------------------
        # RL 循环第 4 步：渲染可视化（训练时关闭以加速）
        # -------------------------------------------------------------------------
        # 渲染只是为了让人类看到机器人在做什么，对 RL 训练本身没有帮助
        # 训练时设 has_renderer=False 可提速 10-100 倍
        # -------------------------------------------------------------------------
        env.render()

        # 限制帧率，让动画不至于太快（人眼看起来舒服）
        # limit frame rate if necessary
        elapsed = time.time() - start
        diff = 1 / MAX_FR - elapsed
        if diff > 0:
            time.sleep(diff)

    # =========================================================================
    # 【扩展】如何把这个脚本改造成真正的 RL 训练？
    # =========================================================================
    # 1. 把随机动作替换为策略网络：
    #      action = policy_network(torch.from_numpy(obs)).sample().numpy()
    #
    # 2. 在循环中累积经验 (obs, action, reward, next_obs, done) 到 replay buffer
    #
    # 3. 每隔若干步，从 buffer 采样 batch 更新策略网络（用 SAC/PPO 等算法）
    #
    # 4. 当 done=True 时调用 env.reset() 开始新 episode
    #
    # 5. 训练完成后，保存策略网络权重，部署时加载即可
    #
    # 推荐使用 GymWrapper 包装环境，然后用 Stable-Baselines3 一行代码训练：
    #      from stable_baselines3 import SAC
    #      model = SAC("MlpPolicy", GymWrapper(env), verbose=1)
    #      model.learn(total_timesteps=100000)
    # =========================================================================
