"""
robosuite + Stable-Baselines3 (SB3) 强化学习训练脚本 —— PPO 版本
SO101 真实机器人模型 + 更接近真实世界的 Lift 仿真环境

本脚本是 train_rl_sb3_so101_realistic.py 的 PPO 算法变体。
主要改动（相对 SAC 版本）：
    1. 算法从 SAC 改为 PPO（on-policy）
    2. 删除 SAC 专属参数：buffer_size、tau、use_sde
    3. 新增 PPO 专属参数：n_steps、n_epochs、gae_lambda、clip_range、ent_coef、
       vf_coef、max_grad_norm
    4. policy_kwargs.net_arch 改为 dict 分离结构（SB3 ≥ 1.6 推荐）
    5. NUM_ENVS 从 8 调到 16（PPO 强依赖并行环境数）
    6. TOTAL_TIMESTEPS 从 500K 调到 2M（PPO 样本效率低）
    7. MODEL_PATH / VEC_NORMALIZE_PATH / LOG_DIR 改为 PPO 专用，避免与 SAC 互相污染
    8. 其余（环境构造、Wrapper、Callback、测试逻辑）完全复用

运行方式：
    conda run -n robosuite python robosuite/demos/train_rl_sb3_so101_realistic_ppo.py
"""

import json
import os
import time

import numpy as np

# 兼容 gymnasium 和 openai gym
# SO101LiftRewardShapingWrapper 需要 gym.Wrapper 基类
try:
    import gymnasium as gym
except ImportError:
    import gym

import robosuite as suite

# 统一使用脚本所在目录推导项目根目录，避免相对路径随运行目录变化
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs", "ppo_lift_so101_realistic")
from robosuite.wrappers import GymWrapper
from robosuite.wrappers.domain_randomization_wrapper import DomainRandomizationWrapper
from robosuite.utils.placement_samplers import UniformRandomSampler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, CheckpointCallback

from so101_realistic import (
    SO101LiftObservationWrapper,
    SO101LiftRewardShapingWrapper,
    TrainingLoggerCallback,
    RolloutCollectorCallback,
    RewardBreakdownCallback,
    REWARD_COMPONENTS,
    analyze_rollouts,
)


# =============================================================================
# 配置开关
# =============================================================================
# SO101 配置
ROBOT_NAME = "SO101"           # 已在 robosuite 中注册
# 默认控制器会从 robosuite/controllers/config/robots/default_so101.json 自动加载

# 训练超参数
# PPO 是 on-policy 算法，强依赖并行环境数。建议 8~32，这里取 16 平衡 CPU 占用与稳定性。
NUM_ENVS = 16
# PPO 样本效率低于 SAC，建议 1M~3M。这里取 2M，足以在 realistic_state 下收敛。
TOTAL_TIMESTEPS = 2_000_000

# 环境模式（三选一）
#   "easy":              使用物体真实位姿 + 稠密奖励，最容易训练，用于快速验证 SO101
#   "realistic_state":   无 oracle 物体状态 + 稀疏奖励，更接近真实世界
#   "vision":            （待扩展）使用相机图像 + 稀疏奖励，最接近真实世界
# 注意：域随机化由 USE_DOMAIN_RANDOMIZATION 独立控制，不受 TRAIN_MODE 影响。
TRAIN_MODE = "easy"

# 是否启用 Domain Randomization（sim2real 域随机化）
#   True:  启用动力学域随机化，增强策略对真实机器人/真实环境的迁移能力
#   False: 关闭域随机化，训练更稳定、收敛更快，但 sim2real 能力可能下降
USE_DOMAIN_RANDOMIZATION = False

# 是否在训练过程中收集 Rollout 轨迹数据
#   True:  启用 RolloutCollectorCallback，保存训练时的 (s, a, r, s', done) 轨迹
#   False: 不收集训练轨迹，减少磁盘开销
COLLECT_TRAINING_ROLLOUTS = True

# 是否在测试阶段保存 Rollout 轨迹数据
#   True:  测试时保存确定性策略的完整 episode 轨迹
#   False: 仅打印测试统计，不保存轨迹
SAVE_TEST_ROLLOUTS = True

# 是否继续训练
# 注意：PPO 模型文件不能跨算法 load（不能加载 SAC 的 .zip）
# 第一次训练请设为 False，跑出 .zip 后再改回 True
RESUME = False
MODEL_PATH = "lift_so101_ppo_realistic.zip"
VEC_NORMALIZE_PATH = "vec_normalize_so101_ppo.pkl"


def make_env():
    """
    创建 SO101 + Lift 环境，并包装为 Gym API。

    根据 USE_DOMAIN_RANDOMIZATION 开关决定是否追加 Domain Randomization。

    返回:
        Monitor: 符合 Gym API 的环境实例
    """
    # =========================================================================
    # 创建 robosuite 环境
    # =========================================================================
    if TRAIN_MODE == "vision":
        raise NotImplementedError(
            "vision 模式需要自定义图像 Wrapper + CNN Policy，请参见脚本末尾注释。"
        )

    # =========================================================================
    # 自定义 cube 放置范围
    # =========================================================================
    # SO101 的可达工作空间经测试：末端最大前伸约 x=-0.09（机器人基座在 x=-0.56）。
    # 默认 UniformRandomSampler 把 cube 放在桌面中心（x≈0），导致 cube 完全不可达。
    # 这里把 cube 限制在机器人前方的可达区域内。
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
        robots=ROBOT_NAME,

        # 渲染设置（训练时关闭）
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,

        # RL 时间设置
        control_freq=20,
        horizon=200,

        # 奖励设置：easy 模式用稠密奖励，realistic_state 模式用稀疏奖励
        reward_shaping=(TRAIN_MODE == "easy"),
        reward_scale=1.0,

        # 观测设置
        use_object_obs=(TRAIN_MODE == "easy"),

        # 初始化噪声
        initialization_noise={
            "magnitude": 0.08,
            "type": "uniform",
        },

        # 桌面摩擦更接近真实材质
        table_friction=(1.0, 5e-3, 1e-4),
        table_full_size=(0.8, 0.8, 0.05),

        # 使用自定义的 cube 放置范围
        placement_initializer=placement_initializer,
    )

    # =========================================================================
    # Domain Randomization：可选的 sim2real 增强
    # =========================================================================
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
            dynamics_randomization_args={
                "randomize_position": False,
                "randomize_quaternion": False,
                "randomize_density": True,
                "randomize_viscosity": True,
                "density_perturbation_ratio": 0.05,
                "viscosity_perturbation_ratio": 0.05,
                "randomize_inertia": True,
                "randomize_mass": True,
                "inertia_perturbation_ratio": 0.03,
                "mass_perturbation_ratio": 0.03,
                "randomize_friction": True,
                "randomize_solref": False,
                "randomize_solimp": False,
                "friction_perturbation_ratio": 0.05,
                "randomize_stiffness": False,
                "randomize_frictionloss": True,
                "randomize_damping": True,
                "randomize_armature": True,
                "frictionloss_perturbation_size": 0.02,
                "damping_perturbation_size": 0.003,
                "armature_perturbation_size": 0.003,
            },
        )

    # 转换为 Gym API
    env = GymWrapper(env)

    # =========================================================================
    # 自定义观测增强：为 realistic_state 拼接 eef -> cube 的相对位置
    # =========================================================================
    if TRAIN_MODE != "easy":
        env = SO101LiftObservationWrapper(env)

    # =========================================================================
    # 自定义奖励 shaping / 监控：所有模式都挂载 wrapper
    # easy 模式下只把各奖励分项写入 info 用于可视化，不改变实际训练奖励
    # =========================================================================
    env = SO101LiftRewardShapingWrapper(env, mode=TRAIN_MODE)

    # Monitor 用于记录 episode 奖励和长度（包装在最外层）
    return Monitor(env)


def build_ppo_kwargs(n_envs):
    """
    构建 PPO 超参数字典。

    与 SAC 的关键区别：
        - 没有 buffer_size、tau、use_sde（off-policy / SAC 专属）
        - 新增 n_steps、n_epochs、gae_lambda、clip_range、ent_coef、vf_coef、max_grad_norm
        - net_arch 用 dict 分离 actor / critic

    整除校验：
        n_steps * n_envs = 2048 * 16 = 32768
        batch_size = 64，32768 / 64 = 512 ✓
        （若修改 NUM_ENVS 或 n_steps，请重新校验）

    Args:
        n_envs: 并行环境数，用于在日志中打印整除校验结果
    """
    return dict(
        verbose=0,
        tensorboard_log=LOG_DIR,
        learning_rate=3e-4,
        n_steps=2048,           # 每个 env 收集 2048 步后做一次策略更新
        batch_size=64,          # 从 n_steps*n_envs 中随机切片，必须能整除
        n_epochs=10,            # 每次 rollout 重复训练 10 轮
        gamma=0.99,
        gae_lambda=0.95,        # GAE 优势函数的 λ，0.95 是经典值
        clip_range=0.2,         # PPO clip 范围，0.2 是原论文值，不要轻易改
        ent_coef=0.01,          # 熵正则系数，稀疏奖励任务建议 0.01~0.02
        vf_coef=0.5,            # value function loss 权重
        max_grad_norm=0.5,      # 梯度裁剪，防止训练发散
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),  # actor/critic 分离
        ),
        device="auto",
    )


def main():
    """训练主流程：创建环境 -> 训练 -> 保存 -> 测试。"""

    # =========================================================================
    # 打印训练配置，方便确认当前运行参数
    # =========================================================================
    print("\n" + "=" * 60)
    print("SO101 Lift 训练配置（PPO 算法）")
    print("=" * 60)
    print(f"  训练模式 (TRAIN_MODE): {TRAIN_MODE}")
    print(f"  算法: PPO (on-policy)")
    print(f"  域随机化 (USE_DOMAIN_RANDOMIZATION): {USE_DOMAIN_RANDOMIZATION}")
    print(f"  收集训练轨迹 (COLLECT_TRAINING_ROLLOUTS): {COLLECT_TRAINING_ROLLOUTS}")
    print(f"  保存测试轨迹 (SAVE_TEST_ROLLOUTS): {SAVE_TEST_ROLLOUTS}")
    print(f"  并行环境数 (NUM_ENVS): {NUM_ENVS}")
    print(f"  总训练步数 (TOTAL_TIMESTEPS): {TOTAL_TIMESTEPS:,}")
    print(f"  是否继续训练 (RESUME): {RESUME}")
    if RESUME:
        print(f"  加载模型路径: {MODEL_PATH}")
        print(f"  加载归一化路径: {VEC_NORMALIZE_PATH}")
    print("=" * 60 + "\n")

    # =========================================================================
    # PPO 整除校验：n_steps * n_envs 必须能被 batch_size 整除
    # =========================================================================
    # 提前校验，避免训练到第一次更新时才报错，浪费时间
    n_steps = 2048
    batch_size = 64
    total_rollout = n_steps * NUM_ENVS
    if total_rollout % batch_size != 0:
        raise ValueError(
            f"PPO 整除校验失败：n_steps({n_steps}) * n_envs({NUM_ENVS}) = "
            f"{total_rollout} 不能被 batch_size({batch_size}) 整除。"
            f"请调整 n_steps / NUM_ENVS / batch_size。"
        )
    print(f"PPO 整除校验通过：{total_rollout} / {batch_size} = {total_rollout // batch_size} batches/rollout\n")

    print(f"正在创建 {NUM_ENVS} 个并行 SO101 Lift 环境...")
    env = SubprocVecEnv([make_env for _ in range(NUM_ENVS)])

    # 观测与奖励归一化
    # PPO 对 reward scale 敏感，norm_reward=True 必须开
    if RESUME and os.path.exists(VEC_NORMALIZE_PATH):
        print(f"加载已有归一化参数：{VEC_NORMALIZE_PATH}")
        env = VecNormalize.load(VEC_NORMALIZE_PATH, env)
    else:
        if RESUME:
            print(f"[警告] 未找到归一化参数 {VEC_NORMALIZE_PATH}，将重新初始化")
        env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # PPO 超参数
    # verbose=0 关闭 SB3 默认的进度表格输出，避免终端被大量重复行刷屏。
    # 训练进度由 TrainingLoggerCallback 控制，评估信息由 EvalCallback 输出。
    ppo_kwargs = build_ppo_kwargs(n_envs=NUM_ENVS)

    if RESUME and os.path.exists(MODEL_PATH):
        print(f"加载已有模型：{MODEL_PATH}")
        model = PPO.load(MODEL_PATH, env=env, **ppo_kwargs)
    else:
        if RESUME:
            print(f"[警告] 未找到模型 {MODEL_PATH}，将从头训练")
        model = PPO("MlpPolicy", env, **ppo_kwargs)

    # =========================================================================
    # 创建评估环境（用于定期评估策略）
    # =========================================================================
    print("\n创建评估环境...")
    eval_env = SubprocVecEnv([make_env for _ in range(1)])
    if RESUME and os.path.exists(VEC_NORMALIZE_PATH):
        eval_env = VecNormalize.load(VEC_NORMALIZE_PATH, eval_env)
    else:
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

    # =========================================================================
    # 组合 Callback：训练日志 + 定期评估 + 定期保存检查点
    # =========================================================================
    log_callback = TrainingLoggerCallback(log_interval=10000, total_timesteps=TOTAL_TIMESTEPS)
    reward_breakdown_callback = RewardBreakdownCallback(log_interval=10000)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(LOG_DIR, "best_model"),
        log_path=os.path.join(LOG_DIR, "eval"),
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(LOG_DIR, "checkpoints"),
        name_prefix="lift_so101_ppo",
        verbose=0,
    )

    callbacks = [log_callback, reward_breakdown_callback, eval_callback, checkpoint_callback]
    if COLLECT_TRAINING_ROLLOUTS:
        rollout_callback = RolloutCollectorCallback(
            save_dir=os.path.join(LOG_DIR, "rollouts"),
            save_freq=50000,
            max_episodes_per_save=100,
        )
        callbacks.append(rollout_callback)
    callback = CallbackList(callbacks)

    print(f"\n开始训练 SO101 + 真实感 Lift（PPO），总目标步数：{TOTAL_TIMESTEPS}")
    print("=" * 60)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

    # =========================================================================
    # 保存最终模型与归一化参数
    # =========================================================================
    final_model_path = "lift_so101_ppo_realistic"
    model.save(final_model_path)
    env.save(VEC_NORMALIZE_PATH)
    print("\n训练完成！已保存：")
    print(f"  - 最终模型：{final_model_path}.zip")
    print(f"  - 最佳模型：{os.path.join(LOG_DIR, 'best_model', 'best_model.zip')}")
    print(f"  - 归一化参数：{VEC_NORMALIZE_PATH}")
    print(f"  - 检查点：{os.path.join(LOG_DIR, 'checkpoints')}/")

    # =========================================================================
    # 快速测试：输出更详细的统计指标，并可选保存 rollout 轨迹
    # =========================================================================
    print("\n开始测试训练好的策略...")
    env.training = False
    env.norm_reward = False

    episode_rewards = []
    episode_lengths = []
    success_count = 0
    n_test_episodes = 10

    test_rollout_dir = os.path.join(LOG_DIR, "test_rollouts")
    if SAVE_TEST_ROLLOUTS:
        os.makedirs(test_rollout_dir, exist_ok=True)

    for i in range(n_test_episodes):
        obs = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = [False]

        # 用于保存当前 episode 的 rollout 数据
        rollout = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "next_observations": [],
            "dones": [],
            "infos": [],
            # 各奖励分量的逐步轨迹（与 REWARD_COMPONENTS 一致），
            # 供后续 analyze_rollouts / diagnose 工具做"奖励从哪来"分析
            "reward_components": {name: [] for name in REWARD_COMPONENTS},
        }

        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, done, info = env.step(action)

            # 收集当前 step 的 (s, a, r, s', done, info)
            rollout["observations"].append(np.array(obs[0], dtype=np.float32))
            rollout["actions"].append(np.array(action[0], dtype=np.float32))
            rollout["rewards"].append(float(reward[0]))
            rollout["next_observations"].append(np.array(next_obs[0], dtype=np.float32))
            rollout["dones"].append(bool(done[0]))
            step_info = info[0] if info and len(info) > 0 else {}
            rollout["infos"].append(step_info)
            for name in REWARD_COMPONENTS:
                rollout["reward_components"][name].append(float(step_info.get(name, 0.0)))

            obs = next_obs
            episode_reward += reward[0]
            episode_length += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # VecEnv 返回的 info 是 list of dicts
        is_success = (
            info and len(info) > 0 and isinstance(info[0], dict)
            and info[0].get("is_success", False)
        )
        if is_success:
            success_count += 1

        # 保存测试 rollout
        if SAVE_TEST_ROLLOUTS:
            save_path = os.path.join(test_rollout_dir, f"rollout_test_ep{i:03d}.npz")
            np.savez_compressed(
                save_path,
                observations=np.array(rollout["observations"]),
                actions=np.array(rollout["actions"]),
                rewards=np.array(rollout["rewards"], dtype=np.float32),
                next_observations=np.array(rollout["next_observations"]),
                dones=np.array(rollout["dones"], dtype=np.uint8),
                # 各奖励分量逐步轨迹：(T,) per component，键名与 REWARD_COMPONENTS 一致
                **{name: np.array(rollout["reward_components"][name], dtype=np.float32)
                   for name in REWARD_COMPONENTS},
                success=is_success,
                total_reward=float(episode_reward),
                length=int(episode_length),
            )
            print(
                f"Episode {i+1:>2}: 奖励 = {episode_reward:>8.2f} | "
                f"步数 = {episode_length:>3} | 成功 = {is_success} | 已保存 {save_path}"
            )
        else:
            print(
                f"Episode {i+1:>2}: 奖励 = {episode_reward:>8.2f} | "
                f"步数 = {episode_length:>3} | 成功 = {is_success}"
            )

    print(f"\n测试统计（{n_test_episodes} 个 episode）：")
    print(f"  平均奖励：{np.mean(episode_rewards):.2f}")
    print(f"  最高奖励：{np.max(episode_rewards):.2f}")
    print(f"  最低奖励：{np.min(episode_rewards):.2f}")
    print(f"  奖励标准差：{np.std(episode_rewards):.2f}")
    print(f"  平均步数：{np.mean(episode_lengths):.1f}")
    print(f"  成功率：{success_count / n_test_episodes * 100:.1f}% ({success_count}/{n_test_episodes})")

    # 对保存的测试轨迹进行详细统计分析
    if SAVE_TEST_ROLLOUTS and os.path.isdir(test_rollout_dir):
        print(f"\n测试 Rollout 已保存至：{test_rollout_dir}")
        analyze_rollouts(test_rollout_dir, success_reward_threshold=2.0, save_report=True)


if __name__ == "__main__":
    main()


# =============================================================================
# 扩展：如何开启 vision 模式（最接近真实世界的设置）
# =============================================================================
# 1. 在 suite.make() 中设置：
#      use_camera_obs=True,
#      has_offscreen_renderer=True,
#      camera_names="agentview",
#      camera_heights=84,
#      camera_widths=84,
#      use_object_obs=False,
#      reward_shaping=False,
#
# 2. 自定义 Wrapper 把 GymWrapper 输出的图像提取为 (H, W, C) Box 观测：
#      class ImageWrapper(gym.ObservationWrapper):
#          def __init__(self, env, image_key="agentview_image"):
#              super().__init__(env)
#              self.image_key = image_key
#              obs = self.env.reset()[0]
#              h, w, c = obs[image_key].shape
#              self.observation_space = gym.spaces.Box(0, 255, (h, w, c), dtype=np.uint8)
#
#          def observation(self, obs):
#              return obs[self.image_key]
#
# 3. 在 make_env() 中返回 ImageWrapper(Monitor(env))。
#
# 4. SB3 使用 PPO("CnnPolicy", env, ...) 训练。
#    注意：PPO 训练 CNN 比 SAC 慢，建议 n_steps 加大到 4096，n_envs 至少 16。
#
# 5. 开启视觉域随机化：
#      randomize_color=True,
#      randomize_camera=True,
#      randomize_lighting=True,
# =============================================================================
