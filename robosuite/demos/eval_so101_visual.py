"""
SO101 训练策略的可视化评估脚本

功能：
    1. 加载训练好的 SAC 模型与 VecNormalize 参数
    2. 创建带渲染的 SO101 + Lift 环境
    3. 可视化运行若干 episode，观察策略行为

运行方式：
    conda run -n robosuite python robosuite/demos/eval_so101_visual.py

注意事项：
    - TRAIN_MODE 必须与训练时一致，否则观测维度不匹配会报错
    - 需要有显示器 / X11 / MuJoCo 渲染窗口
    - 默认加载 lift_so101_sac_realistic.zip 和 vec_normalize_so101.pkl
"""

import os

import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.wrappers.domain_randomization_wrapper import DomainRandomizationWrapper
from robosuite.utils.placement_samplers import UniformRandomSampler
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from so101_realistic import SO101LiftObservationWrapper, SO101LiftRewardShapingWrapper


# =============================================================================
# 配置：必须与训练时一致
# =============================================================================
# 默认模式。如果存在保存的 VecNormalize，脚本会尝试自动推断训练时使用的模式，
# 避免手动填写错误导致观测维度不匹配。
TRAIN_MODE = "realistic_state"   # 训练时用的模式："easy" / "realistic_state"
MODEL_PATH = "lift_so101_sac_realistic.zip"
VEC_NORMALIZE_PATH = "vec_normalize_so101.pkl"

NUM_EPISODES = 5                 # 可视化评估的 episode 数
MAX_STEPS_PER_EPISODE = 200      # 每回合最大步数


def _try_load_with_mode(mode):
    """尝试用指定 TRAIN_MODE 加载 VecNormalize，返回成功加载的 env 或 None。"""
    global TRAIN_MODE
    original_mode = TRAIN_MODE
    TRAIN_MODE = mode
    env = DummyVecEnv([make_env])
    try:
        env = VecNormalize.load(VEC_NORMALIZE_PATH, env)
        return env
    except AssertionError:
        env.close()
        return None
    finally:
        TRAIN_MODE = original_mode


def auto_select_train_mode():
    """根据保存的 VecNormalize 观测维度自动推断训练时使用的 TRAIN_MODE。

    如果默认 TRAIN_MODE 能直接加载成功，则保持原配置；否则会尝试另一种模式。
    两种模式都不匹配时抛出 RuntimeError。
    """
    if not os.path.exists(VEC_NORMALIZE_PATH):
        return None

    # 先尝试默认配置
    env = _try_load_with_mode(TRAIN_MODE)
    if env is not None:
        env.close()
        return TRAIN_MODE

    # 默认失败，尝试另一种模式
    other_mode = "easy" if TRAIN_MODE == "realistic_state" else "realistic_state"
    env = _try_load_with_mode(other_mode)
    if env is not None:
        env.close()
        return other_mode

    raise RuntimeError(
        f"无法为保存的 VecNormalize（{VEC_NORMALIZE_PATH}）匹配任何 TRAIN_MODE。"
        f"请检查该文件是否与当前环境配置兼容。"
    )


def make_env():
    """创建带渲染的 SO101 + Lift 环境，配置必须与训练时一致。"""

    # 与训练脚本保持一致：cube 放在 SO101 可达工作空间内
    placement_initializer = UniformRandomSampler(
        name="SO101ObjectSampler",
        x_range=[-0.28, -0.12],
        y_range=[-0.08, 0.08],
        rotation=None,
        ensure_object_boundary_in_range=False,
        ensure_valid_placement=True,
        reference_pos=(0, 0, 0.8),
        z_offset=0.01,
    )

    env = suite.make(
        "Lift",
        robots="SO101",

        # 渲染设置：开启可视化窗口
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        render_camera="frontview",   # 渲染相机视角

        # RL 时间设置（与训练时一致）
        control_freq=20,
        horizon=MAX_STEPS_PER_EPISODE,

        # 奖励与观测设置（与训练时一致）
        reward_shaping=(TRAIN_MODE == "easy"),
        reward_scale=1.0,
        use_object_obs=(TRAIN_MODE == "easy"),

        # 初始化噪声可以关闭或保留；评估时通常设为 None 更稳定
        initialization_noise=None,

        # 桌面参数（与训练时一致）
        table_friction=(1.0, 5e-3, 1e-4),
        table_full_size=(0.8, 0.8, 0.05),

        # cube 放置范围必须与训练时一致
        placement_initializer=placement_initializer,
    )

    # 评估时通常不需要域随机化；如果训练时开了，可以保留但幅度要小
    if TRAIN_MODE == "realistic_state":
        env = DomainRandomizationWrapper(
            env,
            seed=None,
            randomize_color=False,
            randomize_camera=False,
            randomize_lighting=False,
            randomize_dynamics=False,
            color_randomization_args={},
            camera_randomization_args={},
            lighting_randomization_args={},
            dynamics_randomization_args={},
        )

    env = GymWrapper(env)

    # 与训练脚本保持一致：realistic_state 下叠加观测增强与奖励塑形
    if TRAIN_MODE != "easy":
        env = SO101LiftObservationWrapper(env)
        env = SO101LiftRewardShapingWrapper(env, mode=TRAIN_MODE)

    return Monitor(env)


def main():
    global TRAIN_MODE

    print(f"加载模型：{MODEL_PATH}")
    print(f"加载归一化参数：{VEC_NORMALIZE_PATH}")

    # 自动推断训练时使用的 TRAIN_MODE（如果 VecNormalize 存在）
    detected_mode = auto_select_train_mode()
    if detected_mode is not None and detected_mode != TRAIN_MODE:
        TRAIN_MODE = detected_mode
        print(f"[自动推断] 根据 VecNormalize 维度，使用训练时的 TRAIN_MODE: {TRAIN_MODE}")
    elif detected_mode is not None:
        print(f"评估模式：{TRAIN_MODE}")
    else:
        print(f"评估模式：{TRAIN_MODE}（未找到 VecNormalize，使用默认配置）")
    print("=" * 60)

    # 创建单环境并包装为 VecNormalize（SB3 加载模型需要向量环境）
    env = DummyVecEnv([make_env])
    env = VecNormalize.load(VEC_NORMALIZE_PATH, env)

    # 评估模式：只归一化观测，不归一化奖励
    env.training = False
    env.norm_reward = False

    # 加载 SAC 模型
    model = SAC.load(MODEL_PATH, env=env)

    episode_rewards = []
    for ep in range(NUM_EPISODES):
        obs = env.reset()
        episode_reward = 0.0
        done = [False]
        step = 0

        print(f"\nEpisode {ep + 1}/{NUM_EPISODES} 开始")

        while not done[0] and step < MAX_STEPS_PER_EPISODE:
            # deterministic=True 使用策略均值，表现更稳定
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            step += 1

            # 渲染当前帧；render() 会阻塞并处理窗口事件
            env.render(mode="human")

        episode_rewards.append(episode_reward)
        print(f"Episode {ep + 1}: 步数={step}, 奖励={episode_reward:.2f}")

    env.close()

    print("\n" + "=" * 60)
    print(f"平均奖励：{np.mean(episode_rewards):.2f}")
    print(f"最高奖励：{np.max(episode_rewards):.2f}")
    print(f"最低奖励：{np.min(episode_rewards):.2f}")


if __name__ == "__main__":
    main()
