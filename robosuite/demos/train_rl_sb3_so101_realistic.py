"""
robosuite + Stable-Baselines3 (SB3) 强化学习训练脚本
SO101 真实机器人模型 + 更接近真实世界的 Lift 仿真环境

主要改进：
    1. 机器人改为 SO101（5-DOF 舵机机械臂 + 单指夹爪）
    2. 使用 JOINT_POSITION 控制器，更符合真实 STS3215 舵机控制方式
    3. 引入初始化噪声 + DomainRandomizationWrapper，缩小 sim2real 差距
    4. 使用稀疏奖励（reward_shaping=False），更接近真实任务评价
    5. 关闭 oracle 物体状态（use_object_obs=False），但拼接末端到 cube 的
       相对位置作为观测，使策略在 realistic_state 下仍能感知目标方向
    6. cube 放置范围限制在 SO101 可达工作空间内，避免任务不可达
    7. 优化初始姿态与控制器动作幅度，避免末端悬浮、无法下降
    8. 保留向量化环境、VecNormalize、SAC 等训练优化

运行方式：
    conda run -n robosuite python robosuite/demos/train_rl_sb3_so101_realistic.py
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
LOG_DIR = os.path.join(PROJECT_ROOT, "logs", "sac_lift_so101_realistic")
from robosuite.wrappers import GymWrapper
from robosuite.wrappers.domain_randomization_wrapper import DomainRandomizationWrapper
from robosuite.utils.placement_samplers import UniformRandomSampler
from stable_baselines3 import SAC
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
NUM_ENVS = 16                   # 并行环境数
TOTAL_TIMESTEPS = 1000_000      # 总训练步数

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
RESUME = True
MODEL_PATH = "lift_so101_sac_realistic.zip"
VEC_NORMALIZE_PATH = "vec_normalize_so101.pkl"


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
    # 关键参数说明：
    #   - reward_shaping=False: 稀疏奖励，只有成功抬起才给奖励，更接近真实任务
    #   - use_object_obs=False: 不给出 cube 的真实位姿，策略需从本体感受推断
    #   - initialization_noise: 给机器人初始关节位置加噪声，增加初始状态多样性
    #   - table_friction: 使用更接近真实木桌/橡胶面的摩擦参数
    # =========================================================================
    if TRAIN_MODE == "vision":
        raise NotImplementedError(
            "vision 模式需要自定义图像 Wrapper + CNN Policy，请参见脚本末尾注释。"
        )

    # =========================================================================
    # 自定义 cube 放置范围
    # =========================================================================
    # SO101 的可达工作空间经测试：末端最大前伸约 x=-0.09（机器人基座在 x=-0.56）。
    # 默认 UniformRandomSampler 把 cube 放在桌面中心（x≈0），导致 cube 完全不可达，
    # 策略只能学会“抬起来悬停”这种失败行为。这里把 cube 限制在机器人前方
    # 的可达区域内（x∈[-0.28, -0.12], y∈[-0.08, 0.08]），更贴近真实摆放。
    # =========================================================================
    placement_initializer = UniformRandomSampler(
        name="SO101ObjectSampler",
        x_range=[-0.28, -0.12],
        y_range=[-0.08, 0.08],
        # rotation=None 会在 z 轴上施加 0~2π 均匀随机旋转，使 cube 的有效水平
        # 宽度从边长(44mm)变为面对角线(62mm)。SO101 hinge 夹爪的最大有效内间隙
        # 仅 ~55mm，无法容纳 62mm 的对角线 → pad 穿透 cube → "穿模"。
        # 设为 0（轴对齐）使 cube 宽度固定为 44mm，夹爪间隙 46~55mm 足以容纳。
        rotation=0,
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

        # 观测设置：easy 模式给出 cube 真实位姿；realistic_state 模式关闭 oracle
        # 物体状态，但后续通过 SO101LiftObservationWrapper 拼接 eef->cube 相对位置
        use_object_obs=(TRAIN_MODE == "easy"),

        # 初始化噪声：在优化后的 init_qpos 附近小幅扰动，既保证多样性又避免
        # 末端悬浮过高或进入奇异姿态
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
    # 通过 USE_DOMAIN_RANDOMIZATION 开关控制是否启用。
    # 启用时在每个 episode reset 时随机化动力学参数（质量、摩擦、阻尼等），
    # 帮助策略对真实机器人参数不确定性更鲁棒。
    #
    # 关键修复：
    #   1. randomize_every_n_steps=0：关闭每步随机化，只在 reset 时随机化。
    #      默认值为 1 时每步都会改 body 位姿，导致桌子、机器人底座剧烈抖动。
    #   2. 关闭 body position/quaternion 随机化：桌子、地板等静态物体不应被移动。
    #   3. 关闭 solref/solimp 随机化：接触求解器参数对稳定性敏感，先保持固定。
    #   4. 扰动幅度整体调小，优先保证训练稳定，再逐步增强 sim2real 难度。
    # =========================================================================
    if USE_DOMAIN_RANDOMIZATION:
        env = DomainRandomizationWrapper(
            env,
            seed=None,
            randomize_color=False,     # 无图像观测时视觉随机化无效
            randomize_camera=False,
            randomize_lighting=False,
            randomize_dynamics=True,   # 动力学随机化是核心
            randomize_on_reset=True,   # 每个 episode 开始时随机化一次
            randomize_every_n_steps=0, # 0 表示不在 step 中随机化
            color_randomization_args={},
            camera_randomization_args={},
            lighting_randomization_args={},
            dynamics_randomization_args={
                # 不随机化 body 位姿，避免桌子/底座抖动
                "randomize_position": False,
                "randomize_quaternion": False,
                # 全局介质参数
                "randomize_density": True,
                "randomize_viscosity": True,
                "density_perturbation_ratio": 0.05,
                "viscosity_perturbation_ratio": 0.05,
                # 刚体质量/惯量（只随机化非零质量 body）
                "randomize_inertia": True,
                "randomize_mass": True,
                "inertia_perturbation_ratio": 0.03,
                "mass_perturbation_ratio": 0.03,
                # 接触参数
                "randomize_friction": True,
                "randomize_solref": False,
                "randomize_solimp": False,
                "friction_perturbation_ratio": 0.05,
                # 关节参数
                "randomize_stiffness": False,  # 位置控制关节刚度通常为 0
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
    # use_object_obs=False 时策略看不到 cube 位姿，但真实机器人可通过相机/标定
    # 估计该相对位置。这里把它加入观测，既保留了 realistic_state 的设定，
    # 又让策略有目标方向可学，避免盲目悬停。
    # =========================================================================
    if TRAIN_MODE != "easy":
        env = SO101LiftObservationWrapper(env)

    # =========================================================================
    # 自定义奖励 shaping / 监控：所有模式都挂载 wrapper
    # easy 模式下只把 lift.py 的各奖励分项写入 info 用于可视化，不改变实际训练奖励
    # =========================================================================
    env = SO101LiftRewardShapingWrapper(env, mode=TRAIN_MODE)

    # Monitor 用于记录 episode 奖励和长度（包装在最外层）
    return Monitor(env)


def main():
    """训练主流程：创建环境 -> 训练 -> 保存 -> 测试。"""

    # =========================================================================
    # 打印训练配置，方便确认当前运行参数
    # =========================================================================
    print("\n" + "=" * 60)
    print("SO101 Lift 训练配置")
    print("=" * 60)
    print(f"  训练模式 (TRAIN_MODE): {TRAIN_MODE}")
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

    print(f"正在创建 {NUM_ENVS} 个并行 SO101 Lift 环境...")
    env = SubprocVecEnv([make_env for _ in range(NUM_ENVS)])

    # 观测与奖励归一化
    if RESUME and os.path.exists(VEC_NORMALIZE_PATH):
        print(f"加载已有归一化参数：{VEC_NORMALIZE_PATH}")
        env = VecNormalize.load(VEC_NORMALIZE_PATH, env)
    else:
        if RESUME:
            print(f"[警告] 未找到归一化参数 {VEC_NORMALIZE_PATH}，将重新初始化")
        env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # SAC 超参数
    # verbose=0 关闭 SB3 默认的进度表格输出，避免终端被大量重复行刷屏。
    # 训练进度由 TrainingLoggerCallback 控制，评估信息由 EvalCallback 输出。
    sac_kwargs = dict(
        verbose=0,
        tensorboard_log=LOG_DIR,
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        # 熵系数：防止策略过早收敛到确定性策略。
        # SAC_19 中 ent_coef 从 0.94 降到 0.0007，策略在 500K 步后失去探索能力，
        # 收敛到"接近但不抓取"的局部最优。设 ent_coef="auto" 但降低 target_entropy
        # 使策略需要维持更高的熵，从而延长探索期。
        ent_coef="auto",
        target_entropy=-1.5,  # 默认 -dim(A)/2 = -3，改为 -1.5 增加探索
        policy_kwargs=dict(net_arch=[256, 256], use_sde=False),
        device="auto",
    )

    if RESUME and os.path.exists(MODEL_PATH):
        print(f"加载已有模型：{MODEL_PATH}")
        model = SAC.load(MODEL_PATH, env=env, **sac_kwargs)
    else:
        if RESUME:
            print(f"[警告] 未找到模型 {MODEL_PATH}，将从头训练")
        model = SAC("MlpPolicy", env, **sac_kwargs)

    # =========================================================================
    # 创建评估环境（用于定期评估策略）
    # =========================================================================
    # 评估环境独立创建，避免干扰训练环境的 VecNormalize 统计量。
    # EvalCallback 会自动把训练环境的归一化参数同步到评估环境。
    print("\n创建评估环境...")
    eval_env = SubprocVecEnv([make_env for _ in range(1)])
    if RESUME and os.path.exists(VEC_NORMALIZE_PATH):
        eval_env = VecNormalize.load(VEC_NORMALIZE_PATH, eval_env)
    else:
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

    # =========================================================================
    # 组合 Callback：训练日志 + 定期评估 + 定期保存检查点
    # =========================================================================
    # 训练进度日志：每 10000 步输出一次，避免刷屏
    log_callback = TrainingLoggerCallback(log_interval=10000, total_timesteps=TOTAL_TIMESTEPS)

    # 奖励分量监控：每 10000 步打印各 reward 分量均值，及早发现 reward hacking。
    # 已支持自动识别 info 中的奖励键名：easy 模式显示 lift.py 原始分量，
    # realistic_state 模式显示自定义 shaping 分量。
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

    # 检查点保存：verbose=0 避免每次保存都打印路径
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(LOG_DIR, "checkpoints"),
        name_prefix="lift_so101_sac",
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

    print(f"\n开始训练 SO101 + 真实感 Lift，总目标步数：{TOTAL_TIMESTEPS}")
    print("=" * 60)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

    # =========================================================================
    # 保存最终模型与归一化参数
    # =========================================================================
    final_model_path = "lift_so101_sac_realistic"
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

    # 测试轨迹保存目录
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
            # 供后续 analyze_rollouts / diagnose 工具做“奖励从哪来”分析
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
#              # 从原始观测字典确定图像形状
#              obs = self.env.reset()[0]
#              h, w, c = obs[image_key].shape
#              self.observation_space = gym.spaces.Box(0, 255, (h, w, c), dtype=np.uint8)
#
#          def observation(self, obs):
#              return obs[self.image_key]
#
# 3. 在 make_env() 中返回 ImageWrapper(Monitor(env))。
#
# 4. SB3 使用 SAC("CnnPolicy", env, ...) 训练。
#
# 5. 开启视觉域随机化：
#      randomize_color=True,
#      randomize_camera=True,
#      randomize_lighting=True,
#      color_randomization_args={},
#      camera_randomization_args={},
#      lighting_randomization_args={},
# =============================================================================
