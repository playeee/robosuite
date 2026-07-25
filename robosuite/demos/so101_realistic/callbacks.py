"""
SB3 训练回调：训练日志与 Rollout 收集。
"""

import os
import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from .wrappers import REWARD_COMPONENTS

# =============================================================================
# 训练过程日志 Callback
# =============================================================================
class TrainingLoggerCallback(BaseCallback):
    """
    定期在终端输出训练进度、FPS、已用时间和剩余时间估计。

    输出格式简洁，方便在长时间训练时实时监控。
    """

    def __init__(self, log_interval=1000, total_timesteps=None, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.total_timesteps = total_timesteps
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.time()
        print(f"\n[TrainingLogger] 训练开始，每 {self.log_interval} 步输出一次进度")
        return True

    def _on_step(self):
        if self.num_timesteps % self.log_interval == 0:
            elapsed = time.time() - self.start_time
            fps = int(self.num_timesteps / elapsed) if elapsed > 0 else 0
            total = self.locals.get("total_timesteps", self.total_timesteps)
            if total is None:
                total = self.num_timesteps
            remaining_steps = max(0, total - self.num_timesteps)
            eta_seconds = remaining_steps / fps if fps > 0 else 0.0
            print(
                f"[Train] Step {self.num_timesteps:>8,}/{total:,} | "
                f"FPS {fps:>4} | 已用 {elapsed/60:>6.1f} min | 预计剩余 {eta_seconds/60:>6.1f} min"
            )
        return True

    def _on_training_end(self):
        elapsed = time.time() - self.start_time
        print(f"\n[TrainingLogger] 训练结束，总耗时 {elapsed/60:.1f} min")
        return True


# =============================================================================
# 奖励分量监控 Callback
# =============================================================================
class RewardBreakdownCallback(BaseCallback):
    """
    在训练过程中定期统计并打印各奖励分量的均值，用于及早发现 reward hacking。

    为什么需要它：
        只看 SB3 的 ep_reward 总量无法判断策略“为什么”拿到这个奖励——
        可能是真正完成任务，也可能是某个分量在白送分（例如旧版 r_lift 用桌面
        高度当基线，cube 不动也持续拿 +0.5/步）。本 callback 在每个统计窗口
        内累加 REWARD_COMPONENTS 中各分量的单步值，打印其均值，并写入
        tensorboard（若可用），让你在训练早期就看到“哪个分量在主导奖励”。

    输出示例（每 log_interval 步一次）：
        [RewardBreakdown] step 50000 | 窗口 2000 步
          reward_reach   :  +0.0312   reward_grasp  :  +0.0000
          reward_lift    :  +0.5421  ← 持续高但 original_reward=0 → 可疑
          reward_smooth  :  -0.0061   reward_z_float:  -0.0812
          original_reward:  +0.0000   shaped_reward :  +0.4860

    判读经验：
      - original_reward 恒为 0 但 reward_lift 很高 → 抬升奖励基线写错（reward hacking）
      - reward_reach 始终接近 0 → reach_tanh_scale 过大，梯度信号丢失
      - reward_z_float 很负且持续 → 末端悬浮被罚，说明策略在悬停
      - reward_grasp 始终为 0 → 策略从未接触 cube，需检查可达性/探索
    """

    def __init__(self, log_interval=10000, window=None, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        # 统计窗口大小（步数）。None 时使用 log_interval，保证统计的是“最近一段”而非全局
        self.window = window if window is not None else log_interval
        self._reward_keys = None
        self._reset_accumulators()
        self._last_report_step = 0

    def _detect_reward_keys(self, info):
        """从 info 字典中自动识别奖励分量键名。

        这样不同训练模式（easy / realistic_state）可以显示各自真正相关的曲线，
        避免 easy 模式下出现与训练无关的自定义 shaping 分项。
        """
        if not isinstance(info, dict):
            return list(REWARD_COMPONENTS)
        keys = [
            k for k in info.keys()
            if k.startswith(("reward_", "lift_")) or k in ("original_reward", "shaped_reward")
        ]
        return sorted(keys) if keys else list(REWARD_COMPONENTS)

    def _reset_accumulators(self):
        keys = self._reward_keys if self._reward_keys is not None else REWARD_COMPONENTS
        self._acc = {name: 0.0 for name in keys}
        self._count = 0

    def _on_training_start(self):
        print(
            f"\n[RewardBreakdown] 已启用，每 {self.log_interval} 步输出一次各奖励分量均值"
            f"（统计窗口 {self.window} 步）"
        )
        return True

    def _on_step(self):
        infos = self.locals.get("infos")
        if infos is None or len(infos) == 0:
            return True

        # 延迟检测奖励分量键名：根据第一个包含奖励信息的 info 决定
        if self._reward_keys is None:
            for info in infos:
                if isinstance(info, dict):
                    keys = self._detect_reward_keys(info)
                    if keys:
                        self._reward_keys = keys
                        self._reset_accumulators()
                        break
            if self._reward_keys is None:
                return True

        # 累加每个并行环境的 info（取所有 env 的分量均值更稳健）
        for info in infos:
            if not isinstance(info, dict):
                continue
            for name in self._reward_keys:
                if name in info:
                    self._acc[name] += float(info[name])
            self._count += 1

        if self.num_timesteps - self._last_report_step >= self.log_interval and self._count > 0:
            self._report()
            self._reset_accumulators()
            self._last_report_step = self.num_timesteps
        return True

    def _report(self):
        n = max(self._count, 1)
        names = self._reward_keys if self._reward_keys is not None else REWARD_COMPONENTS
        means = {name: self._acc[name] / n for name in names}

        print(
            f"\n[RewardBreakdown] step {self.num_timesteps:,} | "
            f"统计 {self._count} 个 env-step"
        )
        # 单列排版，兼容动态键名数量
        for name in names:
            print(f"  {name:22s}: {means[name]:+8.4f}")

        # 写入 tensorboard（如果 logger 存在）
        try:
            for name in names:
                self.logger.record(f"reward_breakdown/{name}", means[name])
            self.logger.dump(self.num_timesteps)
        except Exception:
            # logger 不可用时静默跳过，不影响训练
            pass

        # 一句话健康度提示（仅对自定义 shaping 模式的键名有效）
        orig = means.get("original_reward", 0.0)
        lift = means.get("reward_lift", 0.0)
        reach = means.get("reward_reach", 0.0)
        if orig < 1e-6 and lift > 0.2:
            print("  ⚠ 警告：original_reward≈0 但 reward_lift 偏高 → 疑似抬升奖励基线 reward hacking")
        if reach < 1e-3 and orig < 1e-6:
            print("  ⚠ 警告：reward_reach≈0 → 接近奖励梯度丢失，检查 reach_tanh_scale")


# =============================================================================
# Rollout 轨迹收集 Callback
# =============================================================================
class RolloutCollectorCallback(BaseCallback):
    """
    在训练过程中实时收集完整 episode 的 (s, a, r, s', done) 轨迹数据。

    每条轨迹保存为一个 .npz 文件，包含：
        observations:      (T, obs_dim)      当前状态 s
        actions:           (T, act_dim)      动作 a
        rewards:           (T,)              单步奖励 r（包含 shaping 后的奖励）
        next_observations: (T, obs_dim)      下一状态 s'
        dones:             (T,)              是否终止
        infos:             list of dict      每步的 info（含原始奖励、各奖励分项等）

    注意：
      - 默认只收集第一个并行环境（env_id=0）的轨迹，避免数据冗余。
      - 轨迹只在 episode 结束时保存，保证每条轨迹都是完整的。
      - 收集频率由 save_freq 控制，每达到 save_freq 步数时打印一次汇总。
      - max_episodes_per_save=0 表示不自动清理旧轨迹；设为 N>0 时只保留最新的 N 条。
    """

    def __init__(
        self,
        save_dir="./logs/sac_lift_so101_realistic/rollouts",
        save_freq=50000,
        max_episodes_per_save=0,
        verbose=0,
    ):
        super().__init__(verbose)
        # 使用绝对路径，避免工作目录变化导致保存位置不一致
        self.save_dir = os.path.abspath(save_dir)
        self.save_freq = save_freq
        # max_episodes_per_save=0 表示不自动清理，保留所有轨迹
        self.max_episodes_per_save = max_episodes_per_save

        self.current_trajectory = []
        self.collected_count = 0
        self.last_report_step = 0
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_training_start(self):
        print(f"\n[RolloutCollector] Rollout 收集已启用，保存目录：{self.save_dir}")
        return True

    def _on_step(self):
        # SB3 collect_rollouts 在 self.locals 中暴露的变量名。
        # 注意：不同 SB3 版本不一定提供 obs，因此当前状态优先从 model._last_obs 获取。
        action = self.locals.get("actions")
        reward = self.locals.get("rewards")
        done = self.locals.get("dones")
        new_obs = self.locals.get("new_obs")
        info = self.locals.get("infos")
        obs = self.locals.get("obs")

        # 防御性检查：action/reward/done 是必须的核心变量
        if action is None or reward is None or done is None:
            return True

        # 优先使用 locals 中的 obs；若不存在（SB3 >= 2.9 的 Off-PolicyAlgorithm），
        # 则使用 model._last_obs，它正是在执行当前动作之前的状态 s。
        if obs is None and self.model is not None:
            obs = getattr(self.model, "_last_obs", None)
        if obs is None:
            return True

        # 只收集第一个并行环境的轨迹
        next_obs = (
            np.asarray(new_obs[0], dtype=np.float32)
            if new_obs is not None
            else np.zeros_like(obs[0])
        )
        is_done = bool(done[0])

        # 对于终止步，SB3 的 info 中会包含真正的 terminal_observation；
        # 使用它可以让 s' 更准确地表示 episode 结束前的状态。
        if is_done and info is not None and len(info) > 0 and isinstance(info[0], dict):
            terminal_obs = info[0].get("terminal_observation")
            if terminal_obs is not None:
                next_obs = np.asarray(terminal_obs, dtype=np.float32)

        step_data = {
            "obs": np.asarray(obs[0], dtype=np.float32),
            "action": np.asarray(action[0], dtype=np.float32),
            "reward": float(reward[0]),
            "next_obs": next_obs,
            "done": is_done,
            "info": info[0] if info is not None and len(info) > 0 else {},
        }
        self.current_trajectory.append(step_data)

        # episode 结束则保存
        if is_done:
            self._save_current_trajectory()

        # 定期报告
        if self.num_timesteps - self.last_report_step >= self.save_freq:
            print(
                f"[RolloutCollector] Step {self.num_timesteps:,} | "
                f"已累计保存 {self.collected_count} 条完整轨迹"
            )
            self.last_report_step = self.num_timesteps

        return True

    def _save_current_trajectory(self):
        """将当前 episode 的轨迹保存为 .npz 文件，并附带元数据。"""
        if len(self.current_trajectory) == 0:
            return

        total_reward = float(sum(step["reward"] for step in self.current_trajectory))
        length = len(self.current_trajectory)
        # 稀疏成功奖励为 2.25，这里用 2.0 作为成功阈值（兼容 shaping 奖励叠加）
        success = total_reward >= 2.0

        traj = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "next_observations": [],
            "dones": [],
        }
        for step in self.current_trajectory:
            traj["observations"].append(step["obs"])
            traj["actions"].append(step["action"])
            traj["rewards"].append(step["reward"])
            traj["next_observations"].append(step["next_obs"])
            traj["dones"].append(step["done"])

        save_path = os.path.join(
            self.save_dir,
            f"rollout_train_step{self.num_timesteps:08d}_ep{self.collected_count:04d}.npz",
        )
        np.savez_compressed(
            save_path,
            observations=np.array(traj["observations"]),
            actions=np.array(traj["actions"]),
            rewards=np.array(traj["rewards"], dtype=np.float32),
            next_observations=np.array(traj["next_observations"]),
            dones=np.array(traj["dones"], dtype=np.uint8),
            success=success,
            total_reward=total_reward,
            length=length,
        )

        self.collected_count += 1
        self.current_trajectory = []

        # 首次保存或每 10 条轨迹打印一次确认信息，便于排查是否正常工作
        if self.collected_count == 1 or self.collected_count % 10 == 0:
            print(
                f"[RolloutCollector] 已保存第 {self.collected_count} 条训练轨迹："
                f"{save_path}（奖励 {total_reward:.2f}，长度 {length}）"
            )

        # 避免保存过多文件，超过上限时清理最旧的；0 表示不清理
        self._cleanup_old_rollouts()

    def _cleanup_old_rollouts(self):
        """保留最新的 max_episodes_per_save 条轨迹，删除旧文件；0 表示不清理。"""
        if self.max_episodes_per_save <= 0:
            return
        files = sorted([
            f for f in os.listdir(self.save_dir)
            if f.startswith("rollout_train_") and f.endswith(".npz")
        ])
        while len(files) > self.max_episodes_per_save:
            old_file = os.path.join(self.save_dir, files.pop(0))
            try:
                os.remove(old_file)
            except OSError:
                pass

    def _on_training_end(self):
        # 训练结束时若还有未保存的轨迹（未遇到 done），也保存下来
        if len(self.current_trajectory) > 0:
            self._save_current_trajectory()
        print(f"\n[RolloutCollector] 训练结束，共保存 {self.collected_count} 条轨迹")
        return True
