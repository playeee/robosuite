"""
robosuite + SB3 可视化评估脚本

加载训练好的 SAC 模型，在可视化窗口中观看机器人执行任务。

前置条件：
    先运行 train_rl_sb3.py 完成训练，生成以下文件：
    - lift_panda_sac_optimized.zip  （模型权重）
    - vec_normalize.pkl              （归一化参数）

运行方式：
    python robosuite/demos/evaluate_rl.py
"""

import sys
import threading
import time

import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor


# =============================================================================
# 辅助函数
# =============================================================================
def _unwrap_env(vec_env, idx=0):
    """逐层 unwrap VecEnv + Wrapper，拿到最底层的 robosuite 环境。"""
    e = vec_env.envs[idx]
    while hasattr(e, "env"):
        e = e.env
    return e


def safe_input(prompt, timeout=5.0):
    """
    带超时的 input。

    返回:
        (str, bool): (用户输入内容, 是否因超时返回)
                     超时或遇到 EOF 时返回 ("", True/False)。
    """
    if not sys.stdin.isatty():
        print("  [非交互式终端，自动进入下一个 episode]")
        return "", False

    result = [None]

    def _read():
        try:
            result[0] = input(prompt)
        except EOFError:
            result[0] = ""

    t = threading.Thread(target=_read, daemon=True)
    t.start()
    t.join(timeout)

    if t.is_alive():
        print("  [等待超时，自动重播当前 episode]")
        return "", True

    return (result[0] if result[0] is not None else ""), False


# =============================================================================
# 配置：与训练时保持一致（除了渲染设置）
# =============================================================================
MODEL_PATH = "lift_panda_sac_optimized.zip"
VEC_NORMALIZE_PATH = "vec_normalize.pkl"
NUM_EPISODES = 5          # 评估的 episode 数量
RENDER_MODE = "human"     # human = 弹出可视化窗口


def main():
    # =========================================================================
    # 第一步：创建带可视化窗口的环境
    # =========================================================================
    # 评估时开启渲染（has_renderer=True），其余参数必须与训练时完全一致
    # =========================================================================
    def make_eval_env():
        env = suite.make(
            "Lift",
            robots="Panda",
            has_renderer=True,             # 开可视化窗口（与训练时的区别）
            has_offscreen_renderer=False,
            use_camera_obs=False,
            control_freq=20,
            horizon=200,
            reward_shaping=True,
        )
        env = GymWrapper(env)
        env = Monitor(env)
        return env

    # 用 DummyVecEnv 包装（VecNormalize 需要 VecEnv 接口）
    env = DummyVecEnv([make_eval_env])

    # =========================================================================
    # 第二步：加载训练时的归一化参数
    # =========================================================================
    # 必须加载！否则策略会"看不懂"未归一化的观测
    # =========================================================================
    env = VecNormalize.load(VEC_NORMALIZE_PATH, env)

    # 测试模式：关闭训练更新和奖励归一化
    env.training = False
    env.norm_reward = False

    # =========================================================================
    # 第三步：加载训练好的模型
    # =========================================================================
    model = SAC.load(MODEL_PATH, env=env)

    # =========================================================================
    # 第四步：运行评估并渲染
    # =========================================================================
    print("=" * 60)
    print(f"加载模型：{MODEL_PATH}")
    print(f"开始可视化评估，共 {NUM_EPISODES} 个 episode")
    print("=" * 60)

    episode_rewards = []
    episode_successes = []

    for ep in range(NUM_EPISODES):
        # =====================================================================
        # 运行一次原始 episode，并记录动作序列
        # =====================================================================
        obs = env.reset()
        episode_reward = 0.0
        done = [False]
        actions = []          # 保存动作序列，用于重播

        while not done[0]:
            # deterministic=True：用策略均值，表现最稳定
            action, _ = model.predict(obs, deterministic=True)
            actions.append(action)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward[0]
            env.render()  # 渲染画面

        # 直接调用底层 robosuite 环境的 _check_success() 判断任务是否成功
        # （奖励累加值受 reward_shaping 影响，不适合作为成功标准）
        success = _unwrap_env(env)._check_success()
        episode_successes.append(success)
        episode_rewards.append(episode_reward)

        status = "成功" if success else "失败"
        print(f"Episode {ep+1}: 奖励 = {episode_reward:.2f}  [{status}]")

        # =====================================================================
        #  episode 结束后隔一段时间自动重播，直到用户确认进入下一个 episode
        # =====================================================================
        while True:
            prompt = (
                "按 Enter 进入下一个 episode，"
                "输入 r 立即重播（5 秒无输入自动重播当前 episode）："
            )
            user_input, is_timeout = safe_input(prompt, timeout=5.0)
            user_input = user_input.strip().lower()

            # 用户按 Enter 或输入非 r 内容：确认进入下一个 episode
            # 超时或输入 r：重播当前 episode
            if not is_timeout and user_input != "r":
                break

            print("  [重播当前 episode...]")
            time.sleep(0.5)   # 短暂停顿后开始重播
            env.reset()
            for action in actions:
                _, _, done, _ = env.step(action)
                env.render()
                if done[0]:
                    break

    # =========================================================================
    # 第五步：汇总统计
    # =========================================================================
    print("\n" + "=" * 60)
    print("评估汇总")
    print("=" * 60)
    print(f"  平均奖励：  {np.mean(episode_rewards):.2f}")
    print(f"  最高奖励：  {np.max(episode_rewards):.2f}")
    print(f"  最低奖励：  {np.min(episode_rewards):.2f}")
    print(f"  成功率：    {sum(episode_successes)}/{NUM_EPISODES} "
          f"({100 * sum(episode_successes) / NUM_EPISODES:.0f}%)")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
