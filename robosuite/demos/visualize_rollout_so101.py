"""
SO101 策略 Rollout 可视化与收集脚本

功能：
    1. 加载训练好的 SAC 模型与 VecNormalize 参数
    2. 可视化运行若干 episode（3D 渲染窗口）
    3. 保存每条 rollout 轨迹为 .npz 文件
    4. 绘制奖励、末端高度、夹爪开合等时序曲线

运行方式：
    conda run -n robosuite python robosuite/demos/visualize_rollout_so101.py

输出：
    - ./logs/sac_lift_so101_realistic/visual_rollouts/rollout_vis_ep*.npz
    - ./logs/sac_lift_so101_realistic/visual_rollouts/rollout_vis_ep*.png
"""

import os
import time

import numpy as np

# 兼容 gymnasium 和 openai gym
try:
    import gymnasium as gym
except ImportError:
    import gym

import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.wrappers.domain_randomization_wrapper import DomainRandomizationWrapper
from robosuite.utils.placement_samplers import UniformRandomSampler
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# 从工具包复用包装器，确保评估时与训练时完全一致
from so101_realistic import (
    REWARD_COMPONENTS,
    SO101LiftObservationWrapper,
    SO101LiftRewardShapingWrapper,
)


# =============================================================================
# 配置：必须与训练时一致
# =============================================================================
TRAIN_MODE = "realistic_state"   # "easy" / "realistic_state"
MODEL_PATH = "lift_so101_sac_realistic.zip"
VEC_NORMALIZE_PATH = "vec_normalize_so101.pkl"

NUM_EPISODES = 3                 # 可视化运行的 episode 数
MAX_STEPS_PER_EPISODE = 200      # 每回合最大步数
RENDER_CAMERA = "frontview"      # 渲染相机视角
DETERMINISTIC = True             # True 使用策略均值，表现更稳定
SLEEP_PER_STEP = 0.01            # 每步渲染后暂停（秒），0 表示最快

USE_DOMAIN_RANDOMIZATION = False # 可视化时通常关闭域随机化
SAVE_DIR = "./logs/sac_lift_so101_realistic/visual_rollouts"


# =============================================================================
# 环境创建
# =============================================================================
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

        # 渲染设置
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        render_camera=RENDER_CAMERA,

        # RL 时间设置
        control_freq=20,
        horizon=MAX_STEPS_PER_EPISODE,

        # 奖励与观测设置
        reward_shaping=(TRAIN_MODE == "easy"),
        reward_scale=1.0,
        use_object_obs=(TRAIN_MODE == "easy"),

        # 可视化时建议关闭初始化噪声，便于复现
        initialization_noise=None,

        # 桌面参数
        table_friction=(1.0, 5e-3, 1e-4),
        table_full_size=(0.8, 0.8, 0.05),

        # cube 放置范围必须与训练时一致
        placement_initializer=placement_initializer,
    )

    if USE_DOMAIN_RANDOMIZATION:
        env = DomainRandomizationWrapper(
            env,
            seed=None,
            randomize_color=False,
            randomize_camera=False,
            randomize_lighting=False,
            randomize_dynamics=True,
            randomize_on_reset=True,
            randomize_every_n_steps=0,
            color_randomization_args={},
            camera_randomization_args={},
            lighting_randomization_args={},
            dynamics_randomization_args={},
        )

    env = GymWrapper(env)

    # 关键：与训练时保持一致，realistic_state 模式下需要叠加观测增强与奖励塑形
    if TRAIN_MODE != "easy":
        env = SO101LiftObservationWrapper(env)
        env = SO101LiftRewardShapingWrapper(env, mode=TRAIN_MODE)

    return Monitor(env)


# =============================================================================
# 获取 sim 内部状态（用于可视化分析）
# =============================================================================
def _get_base_env(env):
    """从 Monitor -> VecNormalize -> DummyVecEnv -> GymWrapper -> DomainRandomizationWrapper -> Lift 解包。"""
    inner = env.envs[0] if hasattr(env, "envs") else env
    while hasattr(inner, "env"):
        inner = inner.env
    return inner


def _get_eef_pos(env):
    """获取右臂末端执行器位置。"""
    base_env = _get_base_env(env)
    eef_site_id = base_env.robots[0].eef_site_id["right"]
    return np.array(base_env.sim.data.site_xpos[eef_site_id])


def _get_gripper_opening(env):
    """获取夹爪开合比例的近似值（简单归一化到 [0, 1]）。"""
    base_env = _get_base_env(env)
    qpos = base_env.sim.data.qpos[base_env.robots[0]._ref_gripper_joint_pos_indexes["right"]]
    actuator_ids = base_env.robots[0]._ref_joint_gripper_actuator_indexes["right"]
    ratios = []
    for i, act_id in enumerate(actuator_ids):
        lo, hi = base_env.sim.model.actuator_ctrlrange[act_id]
        if hi > lo:
            ratios.append(np.clip((qpos[i] - lo) / (hi - lo), 0.0, 1.0))
    return float(np.mean(ratios)) if ratios else 0.0


def _get_cube_height(env):
    """获取 cube 中心的高度（用于判断抬起）。"""
    base_env = _get_base_env(env)
    return float(base_env.sim.data.body_xpos[base_env.cube_body_id][2])


# =============================================================================
# 单条 rollout 收集与可视化
# =============================================================================
def collect_and_visualize_rollout(model, env, episode_idx=0):
    """
    运行一个 episode，实时渲染，并收集完整 rollout 数据。

    返回:
        dict: 包含 observations, actions, rewards, next_observations, dones,
              eef_heights, gripper_openings, cube_heights 以及统计信息
    """
    obs = env.reset()
    done = [False]
    step = 0
    episode_reward = 0.0

    rollout = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "next_observations": [],
        "dones": [],
        "eef_heights": [],
        "gripper_openings": [],
        "cube_heights": [],
        # 各奖励分量的逐步轨迹，用于绘制“奖励分解”堆叠图，判断策略奖励来源
        "reward_components": {name: [] for name in REWARD_COMPONENTS},
    }

    print(f"\nEpisode {episode_idx + 1} 开始可视化...")

    while not done[0] and step < MAX_STEPS_PER_EPISODE:
        action, _ = model.predict(obs, deterministic=DETERMINISTIC)
        next_obs, reward, done, info = env.step(action)

        # 记录 rollout 数据
        rollout["observations"].append(np.array(obs[0], dtype=np.float32))
        rollout["actions"].append(np.array(action[0], dtype=np.float32))
        rollout["rewards"].append(float(reward[0]))
        rollout["next_observations"].append(np.array(next_obs[0], dtype=np.float32))
        rollout["dones"].append(bool(done[0]))

        # 记录可视化分析用的额外状态
        rollout["eef_heights"].append(_get_eef_pos(env)[2])
        rollout["gripper_openings"].append(_get_gripper_opening(env))
        rollout["cube_heights"].append(_get_cube_height(env))

        # 记录各奖励分量（info[0] 是 SO101LiftRewardShapingWrapper 写入的分项）
        step_info = info[0] if info and len(info) > 0 and isinstance(info[0], dict) else {}
        for name in REWARD_COMPONENTS:
            rollout["reward_components"][name].append(float(step_info.get(name, 0.0)))

        obs = next_obs
        episode_reward += reward[0]
        step += 1

        # 渲染并控制播放速度
        env.render(mode="human")
        if SLEEP_PER_STEP > 0:
            time.sleep(SLEEP_PER_STEP)

    rollout["episode_reward"] = episode_reward
    rollout["episode_length"] = step
    rollout["success"] = episode_reward >= 2.0

    print(f"Episode {episode_idx + 1} 结束：步数={step}, 总奖励={episode_reward:.2f}, 成功={rollout['success']}")
    return rollout


# =============================================================================
# 保存 rollout 为 .npz
# =============================================================================
def save_rollout(rollout, episode_idx):
    """将 rollout 保存为 .npz 文件。"""
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f"rollout_vis_ep{episode_idx:03d}.npz")

    np.savez_compressed(
        save_path,
        observations=np.array(rollout["observations"]),
        actions=np.array(rollout["actions"]),
        rewards=np.array(rollout["rewards"], dtype=np.float32),
        next_observations=np.array(rollout["next_observations"]),
        dones=np.array(rollout["dones"], dtype=np.uint8),
        eef_heights=np.array(rollout["eef_heights"], dtype=np.float32),
        gripper_openings=np.array(rollout["gripper_openings"], dtype=np.float32),
        cube_heights=np.array(rollout["cube_heights"], dtype=np.float32),
        # 各奖励分量的逐步轨迹：(T,) per component，键名与 REWARD_COMPONENTS 一致
        **{name: np.array(rollout["reward_components"][name], dtype=np.float32)
           for name in REWARD_COMPONENTS},
        success=rollout["success"],
        total_reward=rollout["episode_reward"],
        length=rollout["episode_length"],
    )
    print(f"  轨迹已保存：{save_path}")
    return save_path


# =============================================================================
# 绘制 rollout 分析图
# =============================================================================
def plot_rollout(rollout, episode_idx):
    """
    绘制奖励、末端高度、夹爪开合、cube 高度的时序曲线。

    如果未安装 matplotlib，会跳过绘图并提示。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # 无头环境也能保存图片
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [提示] 未安装 matplotlib，跳过绘图。可运行：conda run -n robosuite pip install matplotlib")
        return None

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f"rollout_vis_ep{episode_idx:03d}.png")

    rewards = np.array(rollout["rewards"])
    eef_heights = np.array(rollout["eef_heights"])
    gripper = np.array(rollout["gripper_openings"])
    cube = np.array(rollout["cube_heights"])
    steps = np.arange(len(rewards))

    # 奖励分量：正信号（任务进度/先验）与负信号（惩罚）分别堆叠
    pos_names = ["reward_reach", "reward_grasp", "reward_lift", "reward_gripper_move"]
    neg_names = ["reward_smooth", "reward_vel", "reward_ee_vel", "reward_z_float"]
    pos_stack = np.array([rollout["reward_components"][n] for n in pos_names])  # (4, T)
    neg_stack = np.array([rollout["reward_components"][n] for n in neg_names])

    fig, axes = plt.subplots(5, 1, figsize=(11, 15), sharex=True)
    fig.suptitle(f"Episode {episode_idx + 1} Rollout Analysis\n"
                 f"Total Reward={rollout['episode_reward']:.2f} | Success={rollout['success']} | Length={rollout['episode_length']}")

    # 0) 奖励分量堆叠面积图（核心诊断图：一眼看出“奖励从哪来”）
    colors_pos = ["#4C9F70", "#2E86AB", "#F6AE2D", "#9D8DF1"]
    colors_neg = ["#888", "#AAA", "#CCC", "#E07A5F"]
    axes[0].stackplot(steps, pos_stack, labels=pos_names, colors=colors_pos, alpha=0.85)
    axes[0].stackplot(steps, neg_stack, labels=neg_names, colors=colors_neg, alpha=0.85)
    axes[0].axhline(0, color="black", linewidth=0.6)
    axes[0].set_ylabel("各奖励分量")
    axes[0].set_title("奖励分解（堆叠面积：上=正信号，下=惩罚）")
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    axes[0].grid(True, alpha=0.3)

    # 1) 单步总奖励
    axes[1].plot(steps, rewards, color="C0", linewidth=1.2)
    axes[1].axhline(2.25, color="red", linestyle="--", label="success reward (2.25)")
    axes[1].set_ylabel("Step Reward")
    axes[1].set_title("Reward per Step (合成后)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 2) 末端高度
    axes[2].plot(steps, eef_heights, color="C1", linewidth=1.2)
    axes[2].set_ylabel("EEF Z (m)")
    axes[2].set_title("End-Effector Height")
    axes[2].grid(True, alpha=0.3)

    # 3) 夹爪开合
    axes[3].plot(steps, gripper, color="C2", linewidth=1.2)
    axes[3].set_ylabel("Gripper Opening")
    axes[3].set_title("Gripper Opening Ratio (0=closed, 1=open)")
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].grid(True, alpha=0.3)

    # 4) cube 高度
    axes[4].plot(steps, cube, color="C3", linewidth=1.2)
    axes[4].set_ylabel("Cube Z (m)")
    axes[4].set_title("Cube Height")
    axes[4].set_xlabel("Step")
    axes[4].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  分析图已保存：{save_path}")
    return save_path


# =============================================================================
# 主流程
# =============================================================================
def main():
    print("=" * 60)
    print("SO101 Lift Rollout 可视化与收集")
    print("=" * 60)
    print(f"模型：{MODEL_PATH}")
    print(f"归一化参数：{VEC_NORMALIZE_PATH}")
    print(f"训练模式：{TRAIN_MODE}")
    print(f"评估 episode 数：{NUM_EPISODES}")
    print(f"保存目录：{SAVE_DIR}")
    print("=" * 60)

    # 检查文件
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"未找到模型文件：{MODEL_PATH}，请先完成训练。")
    if not os.path.exists(VEC_NORMALIZE_PATH):
        raise FileNotFoundError(f"未找到归一化参数：{VEC_NORMALIZE_PATH}，请先完成训练。")

    # 创建环境
    env = DummyVecEnv([make_env])
    env = VecNormalize.load(VEC_NORMALIZE_PATH, env)
    env.training = False
    env.norm_reward = False

    # 加载模型
    model = SAC.load(MODEL_PATH, env=env)
    print("\n模型加载成功，开始可视化...")
    print("提示：按关闭按钮或等待所有 episode 运行结束。")

    episode_rewards = []
    saved_paths = []

    for ep in range(NUM_EPISODES):
        rollout = collect_and_visualize_rollout(model, env, episode_idx=ep)
        save_rollout(rollout, episode_idx=ep)
        plot_rollout(rollout, episode_idx=ep)
        episode_rewards.append(rollout["episode_reward"])

    env.close()

    # 汇总统计
    print("\n" + "=" * 60)
    print("可视化 Rollout 汇总")
    print("=" * 60)
    print(f"Episode 奖励：{[f'{r:.2f}' for r in episode_rewards]}")
    print(f"平均奖励：{np.mean(episode_rewards):.2f}")
    print(f"最高奖励：{np.max(episode_rewards):.2f}")
    print(f"最低奖励：{np.min(episode_rewards):.2f}")
    print(f"文件保存目录：{SAVE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
