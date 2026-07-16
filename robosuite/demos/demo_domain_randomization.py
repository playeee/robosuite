"""
用于展示域随机化（Domain Randomization，DR）功能的脚本。

=============================================================================
【核心概念】什么是域随机化？
=============================================================================

域随机化（Domain Randomization）是 sim2real（仿真到现实迁移）中最常用的技术之一。
核心思想：在仿真训练时，刻意把环境的各种视觉/物理参数在一个较大范围内随机变化，
        让策略对参数扰动"见怪不怪"，从而提高迁移到真实机器人时的鲁棒性。

可随机化的维度通常包括：
    - 视觉（Visual）：物体颜色、纹理、光照、相机位姿、相机内参
    - 物理（Dynamics）：摩擦、质量、阻尼、刚度、质心位置、环境介质密度/粘度

为什么有效？
    - 真实世界可视为仿真参数空间中的一个"点"；如果策略见过足够多样的仿真参数，
      它对真实世界参数就具有了泛化能力。
    - 相当于把"模型不确定性"显式地注入训练分布，正则化了策略。

本脚本重点演示 robosuite 提供的 DomainRandomizationWrapper，它会在每个 episode
或每个 step 自动调用底层的 Modder（LightingModder、TextureModder、CameraModder、
DynamicsModder 等）对环境进行随机化。

=============================================================================
注意：使用 instance randomization 时，会按"geom 组"（instance）整体随机化颜色/纹理，
      而不是把每个小 geom 单独随机化。这样更符合直觉：例如一整个机器人保持统一颜色。
"""

import time

import robosuite.macros as macros
import robosuite as suite
from robosuite.utils.input_utils import *
from robosuite.wrappers import DomainRandomizationWrapper

# 启用 instance 随机化：让整组 geom 一起随机化颜色/纹理
# 这样可以避免出现机器人上半身红色、下半身蓝色的"碎片化"随机效果
macros.USING_INSTANCE_RANDOMIZATION = True

if __name__ == "__main__":
    # 创建一个字典，保存将要传给 suite.make(...) 的环境创建参数
    options = {}

    # 打印欢迎信息与 robosuite 版本号
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    # 让用户交互式选择环境，并记录到 options 中
    options["env_name"] = choose_environment()

    # 如果选择的是双臂环境，需要额外选择配置方式以及合适的机器人
    if "TwoArm" in options["env_name"]:
        # 选择双臂配置（如双手协调 / 双单臂等）
        options["env_configuration"] = choose_multi_arm_config()

        # 若选择双手协调（bimanual）配置，则必须使用 Baxter 机器人
        # 否则让用户逐个选择两个单臂机器人
        if options["env_configuration"] == "bimanual":
            options["robots"] = "Baxter"
        else:
            options["robots"] = []

            # 提示用户选择两个机器人
            print("A multiple single-arm configuration was chosen.\n")

            for i in range(2):
                print("Please choose Robot {}...\n".format(i))
                options["robots"].append(choose_robots(exclude_bimanual=True))
    # 若选择的是人形机器人环境，则使用人形机器人选项
    elif "Humanoid" in options["env_name"]:
        options["robots"] = choose_robots(use_humanoids=True)
    # 否则为普通单臂环境，直接选择一个单臂机器人
    else:
        options["robots"] = choose_robots(exclude_bimanual=True)

    # 创建基础任务环境
    env = suite.make(
        **options,
        has_renderer=True,  # 启用屏幕渲染，便于观察随机化效果
        has_offscreen_renderer=False,  # 无需离屏渲染
        ignore_done=True,  # 不因为 horizon 结束而终止，方便持续观察
        use_camera_obs=False,  # 不使用图像观测
        control_freq=20,  # 控制频率 20Hz
        hard_reset=False,  # TODO: 在 macOS 上设为 True 可能导致 segfault，Linux 上可能出现 glfw 错误
    )

    # 用 DomainRandomizationWrapper 包裹环境，开启域随机化
    #
    # 当前示例为了可视化效果更平滑，关闭了大部分随机化：
    #   randomize_color=False   颜色随机化（当前仅 mujoco==3.1.1 支持）
    #   randomize_camera=False  相机随机化（视觉晃动较剧烈，关闭后更易观察）
    #   randomize_dynamics=False 动力学随机化
    # 若要让训练具备 sim2real 鲁棒性，通常应将这些设为 True。
    env = DomainRandomizationWrapper(
        env,
        randomize_color=False,  # 颜色随机化目前仅在 mujoco==3.1.1 下工作
        randomize_camera=False,  # 关闭相机随机化，避免可视化时画面过于晃动
        randomize_dynamics=False,
    )

    # 重置环境并设置相机视角
    env.reset()
    env.viewer.set_camera(camera_id=0)

    max_frame_rate = 20  # 设置期望的最大帧率（每秒 20 帧）

    # 获取动作空间的上下界
    low, high = env.action_spec

    # 可视化主循环：随机采样动作并执行，同时观察域随机化效果
    for i in range(100):
        # 在动作空间范围内均匀随机采样一个动作
        action = np.random.uniform(low, high)
        # 执行动作，返回观测、奖励、是否终止、额外信息
        obs, reward, done, _ = env.step(action)
        # 渲染当前帧到屏幕
        env.render()
        # 按目标帧率休眠，控制可视化速度
        time.sleep(1 / max_frame_rate)
