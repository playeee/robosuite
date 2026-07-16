"""传感器损坏演示（Sensor Corruption Demo）。

=============================================================================
【核心概念】为什么仿真中需要模拟传感器损坏？
=============================================================================

真实机器人上的传感器从来不是完美的：
    - 相机图像有噪声、运动模糊、延迟
    - 关节编码器有读数误差
    - 控制指令下发到执行器存在通信延迟

如果只在"干净"的仿真环境中训练，策略学到的是理想观测到理想动作的映射。
一旦部署到真实机器人，面对带噪声、有延迟的观测，策略性能会急剧下降。

robosuite 的 Observable 机制允许我们为任意传感器添加：
    - corrupter（损坏器）：例如高斯噪声
    - delayer（延迟器）：例如均匀采样的随机延迟
    - sampling_rate（采样率）：降低某些传感器的更新频率

本脚本在 demo_device_control（遥操作脚本）基础上修改而来，演示如何给
图像观测和本体感觉（proprioception，即关节位置）同时添加噪声与延迟。

=============================================================================
示例运行方式：
    $ python demo_sensor_corruption.py --environment Stack --robots Panda \
        --delay 0.05 --corruption 5.0 --toggle-corruption-on-grasp
"""

import argparse
import sys
from copy import deepcopy

import cv2
import numpy as np

import robosuite as suite
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.utils.observables import Observable, create_gaussian_noise_corrupter, create_uniform_sampled_delayer
from robosuite.wrappers import VisualizationWrapper

if __name__ == "__main__":
    # =========================================================================
    # 命令行参数解析：用户可自定义环境、机器人、控制设备、噪声与延迟强度等
    # =========================================================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Lift", help="要使用的任务环境名称")
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default="Panda",
        help="环境中使用的机器人（一个或多个）",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="必要时指定的环境配置（如双臂任务的配置方式）",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="在双手协调（bimanual）场景下控制哪只手臂：'right' 或 'left'",
    )
    parser.add_argument(
        "--switch-on-grasp",
        action="store_true",
        help="在夹爪动作发生时切换控制的手臂",
    )
    parser.add_argument(
        "--toggle-corruption-on-grasp",
        action="store_true",
        help="在夹爪动作发生时切换传感器损坏的开启/关闭状态",
    )
    parser.add_argument("--device", type=str, default="keyboard", help="输入设备：keyboard / spacemouse / dualsense")
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=1.0,
        help="位置输入的缩放系数（越大相同输入使末端移动越快）",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=1.0,
        help="旋转输入的缩放系数（越大相同输入使末端旋转越快）",
    )
    parser.add_argument("--delay", type=float, default=0.04, help="平均延迟时间（秒）")
    parser.add_argument(
        "--corruption",
        type=float,
        default=20.0,
        help="损坏强度（高斯噪声的标准差）",
    )
    parser.add_argument("--camera", type=str, default="agentview", help="要渲染的相机名称")
    parser.add_argument("--width", type=int, default=512, help="相机图像宽度")
    parser.add_argument("--height", type=int, default=384, help="相机图像高度")
    parser.add_argument(
        "--reverse_xy",
        type=bool,
        default=False,
        help="（仅 DualSense）反转摇杆 x/y 轴的效果。用于处理画面中左右/前后方向与摇杆方向相反的情况",
    )

    args = parser.parse_args()

    # 根据命令行参数构造环境配置字典
    config = {
        "env_name": args.environment,
        "robots": args.robots,
    }

    # 若使用双臂环境，需要额外传入 env_configuration 参数；否则无需该参数
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config
    else:
        args.config = None

    # =========================================================================
    # 创建环境：启用离屏渲染以获取相机图像，同时使用图像观测和物体真实状态观测
    # =========================================================================
    env = suite.make(
        **config,
        has_renderer=False,  # 不启用屏幕渲染，改为用 cv2 显示离屏图像
        has_offscreen_renderer=True,  # 启用离屏渲染以获取相机图像
        ignore_done=True,  # 不因 horizon 结束而终止，便于持续操作
        camera_names=args.camera,
        camera_heights=args.height,
        camera_widths=args.width,
        use_camera_obs=True,  # 将相机图像加入观测
        use_object_obs=True,  # 将物体真实状态加入观测
        hard_reset=False,
    )

    # 用 VisualizationWrapper 包裹环境，用于可视化控制指示器（indicator）
    # indicator_configs=None 表示不额外配置指示器
    env = VisualizationWrapper(env, indicator_configs=None)

    # =========================================================================
    # 配置 Observables：给图像和本体感觉传感器添加 corrupter（噪声）和 delayer（延迟）
    #
    # attributes 列表表示 Observable 的三个可修改属性：
    #   - corrupter:  损坏器（加噪声）
    #   - delayer:    延迟器
    #   - sampling_rate: 采样频率
    # =========================================================================
    attributes = ["corrupter", "delayer", "sampling_rate"]
    corruption_mode = 1  # 1 表示损坏开启，0 表示损坏关闭
    obs_settings = {}

    # 辅助函数：批量修改某个 observable 的多个属性
    def modify_obs(obs_name, attrs, mods):
        for attr, mod in zip(attrs, mods):
            env.modify_observable(
                observable_name=obs_name,
                attribute=attr,
                modifier=mod,
            )

    # =========================================================================
    # 1. 给图像观测添加高斯噪声与随机延迟
    # =========================================================================
    image_sampling_rate = 10.0  # 图像采样率 10Hz，低于控制频率，模拟低帧率相机
    image_obs_name = f"{args.camera}_image"
    # 高斯噪声损坏器：均值为 0，标准差为 corruption，像素值裁剪到 [0, 255]
    image_corrupter = create_gaussian_noise_corrupter(mean=0.0, std=args.corruption, low=0, high=255)
    # 均匀采样延迟器：延迟在 [delay-0.025, delay+0.025] 秒之间随机
    image_delayer = create_uniform_sampled_delayer(min_delay=max(0, args.delay - 0.025), max_delay=args.delay + 0.025)
    image_modifiers = [image_corrupter, image_delayer, image_sampling_rate]

    # 初始化图像 observable：应用噪声、延迟、采样率
    modify_obs(obs_name=image_obs_name, attrs=attributes, mods=image_modifiers)

    # 记录图像观测的损坏/延迟设置，方便后续按 corruption_mode 动态切换
    obs_settings[image_obs_name] = {
        "attrs": attributes[:2],
        "mods": lambda: image_modifiers[:2] if corruption_mode else [None, None],
    }

    # =========================================================================
    # 2. 给本体感觉（关节位置）观测添加高斯噪声与随机延迟
    # =========================================================================
    proprio_sampling_rate = 20.0  # 本体感觉采样率 20Hz，与控制频率一致
    # 构造关节位置观测名称：机器人命名前缀 + "joint_pos"
    proprio_obs_name = f"{env.robots[0].robot_model.naming_prefix}joint_pos"
    # 获取关节限位，用于根据关节行程确定噪声尺度
    joint_limits = env.sim.model.jnt_range[env.robots[0]._ref_joint_indexes]
    joint_range = joint_limits[:, 1] - joint_limits[:, 0]
    # 高斯噪声：标准差为关节行程的 1/50（模拟编码器噪声）
    proprio_corrupter = create_gaussian_noise_corrupter(mean=0.0, std=joint_range / 50.0)
    # 当前 proprioception 延迟，初始为 0
    curr_proprio_delay = 0.0
    # 临时延迟器：延迟范围约为图像延迟的一半
    tmp_delayer = create_uniform_sampled_delayer(
        min_delay=max(0, (args.delay - 0.025) / 2), max_delay=(args.delay + 0.025) / 2
    )

    # 自定义延迟函数：同步"损坏传感器"与"真实值延迟传感器"的延迟量
    # 这样我们才能准确计算"噪声带来的偏差"而不是"延迟带来的偏差"
    def proprio_delayer():
        global curr_proprio_delay
        curr_proprio_delay = tmp_delayer()
        return curr_proprio_delay

    # 将原始延迟时间（秒）转换为离散时间步下的采样延迟
    def calculate_proprio_delay():
        base = env.model_timestep  # 单个仿真步的时间（秒）
        return base * round(curr_proprio_delay / base) if corruption_mode else 0.0

    proprio_modifiers = [proprio_corrupter, proprio_delayer, proprio_sampling_rate]

    # 创建一个"ground truth" 但同样被延迟的本体感觉 observable，用于实时评估损坏程度
    # 通过比较 "corrupted delayed" 与 "ground truth delayed"，可单独量化噪声影响
    proprio_sensor = env._observables[proprio_obs_name]._sensor
    proprio_ground_truth_obs_name = f"{proprio_obs_name}_ground_truth"
    observable = Observable(
        name=proprio_ground_truth_obs_name,
        sensor=proprio_sensor,
        delayer=lambda: curr_proprio_delay,  # 使用与损坏传感器相同的延迟
        sampling_rate=proprio_sampling_rate,
    )

    # 将该 ground truth 延迟观测添加到环境
    env.add_observable(observable)

    # 默认情况下 joint_pos 这个 observable 可能不是 active 的，需要显式激活
    env.modify_observable(observable_name=proprio_obs_name, attribute="active", modifier=True)

    # 初始化 proprioception observable
    modify_obs(obs_name=proprio_obs_name, attrs=attributes, mods=proprio_modifiers)

    # 记录 proprioception 的损坏/延迟设置
    obs_settings[proprio_obs_name] = {
        "attrs": attributes[:2],
        "mods": lambda: proprio_modifiers[:2] if corruption_mode else [None, None],
    }
    obs_settings[proprio_ground_truth_obs_name] = {
        "attrs": [attributes[1]],  # 只有 delayer 需要动态切换
        "mods": lambda: [lambda: curr_proprio_delay] if corruption_mode else [None],
    }

    # 设置 NumPy 数组打印格式：保留 3 位小数、抑制科学计数法
    np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

    # =========================================================================
    # 初始化输入设备（键盘 / 3D 鼠标 / DualSense 手柄）
    # =========================================================================
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    elif args.device == "dualsense":
        from robosuite.devices import DualSense

        device = DualSense(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
            reverse_xy=args.reverse_xy,
        )
    else:
        raise Exception("Invalid device choice: choose either 'keyboard' or 'spacemouse' or 'dualsense'.")

    # =========================================================================
    # 主循环：重置环境 -> 读取设备输入 -> 执行动作 -> 显示带损坏的观测
    # =========================================================================
    while True:
        # 重置环境，获取初始观测
        obs = env.reset()

        # 重置损坏模式为开启状态
        corruption_mode = 1

        # 用于在 reset 之间保持状态的变量
        last_grasp = 0

        # 启动设备控制
        device.start_control()

        # 记录每个机器人每根手臂的上一次夹爪动作（基于位置控制，需要在手臂切换时保持）
        all_prev_gripper_actions = [
            {
                f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
                for robot_arm in robot.arms
                if robot.gripper[robot_arm].dof > 0
            }
            for robot in env.robots
        ]

        while True:
            # 设置当前活跃的机器人
            active_robot = env.robots[device.active_robot]

            # 从设备读取最新的动作（平移、旋转、夹爪等）
            input_ac_dict = device.input2action()

            # 若 input2action 返回 None，表示用户请求重置，跳出内层循环
            if input_ac_dict is None:
                break

            # 深拷贝输入动作，避免后续修改影响原始字典
            action_dict = deepcopy(input_ac_dict)

            # 根据控制器输入类型（delta / absolute）填充各手臂动作
            for arm in active_robot.arms:
                if isinstance(active_robot.composite_controller, WholeBody):
                    # WholeBody 复合控制器：从 joint_action_policy 获取输入类型
                    controller_input_type = active_robot.composite_controller.joint_action_policy.input_type
                else:
                    # 普通控制器：从对应手臂的 part_controller 获取输入类型
                    controller_input_type = active_robot.part_controllers[arm].input_type

                if controller_input_type == "delta":
                    action_dict[arm] = input_ac_dict[f"{arm}_delta"]
                elif controller_input_type == "absolute":
                    action_dict[arm] = input_ac_dict[f"{arm}_abs"]
                else:
                    raise ValueError

            # 为每个机器人创建完整动作向量：非活跃机器人保持上一次夹爪动作，
            # 活跃机器人使用当前动作字典
            env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
            env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
            env_action = np.concatenate(env_action)

            # 执行一步仿真并渲染
            obs, reward, done, info = env.step(env_action)

            # 计算并打印本体感觉的统计信息：观测值、损坏量（与 ground truth 延迟值之差）、延迟
            observed_value = obs[proprio_obs_name]
            ground_truth_delayed_value = obs[proprio_ground_truth_obs_name]
            print(
                f"Observed joint pos: {observed_value}, "
                f"Corruption: {observed_value - ground_truth_delayed_value}, "
                f"Delay: {calculate_proprio_delay():.3f} sec"
            )

            # 读取相机图像并转换为 OpenCV 可显示的格式（BGR、上下翻转）
            im = np.flip(obs[args.camera + "_image"][..., ::-1], 0).astype(np.uint8)

            cv2.imshow("offscreen render", im)
            cv2.waitKey(1)
