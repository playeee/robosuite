#!/usr/bin/env python3
"""
通用 SAC 训练分析工具。

适用于任意 SAC 训练运行（不限于 SAC_19），支持：

1. TensorBoard 分析：读取事件文件，提取所有可用 tag，打印汇总表，绘制多面板图。
2. Rollout 轨迹分析：加载 .npz 文件，自动探测可用键，绘制聚合/逐 episode 图。
3. 关键诊断：成功率、抓取率、EEF-cube 距离统计、动作统计。

使用方式：
    python tools/analyze_training.py --logdir ./logs/sac_lift_so101_realistic/ --run SAC_21
    python tools/analyze_training.py --logdir ./logs/sac_lift_so101_realistic/
    python tools/analyze_training.py --logdir ./logs/sac_lift_so101_realistic/ \
        --rollout-dir ./logs/sac_lift_so101_realistic/test_rollouts \
        --output-dir ./tools/my_analysis
"""

import argparse
import os
import sys
import glob
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    HAS_TB = True
except ImportError:
    HAS_TB = False


# ══════════════════════════════════════════════════════════════════════════════
# 路径/运行 自动探测
# ══════════════════════════════════════════════════════════════════════════════

def find_most_recent_run(logdir):
    """在 logdir 中查找最近的 SAC_* 目录，返回其 basename（如 'SAC_21'）。"""
    pattern = os.path.join(logdir, "SAC_*")
    runs = glob.glob(pattern)
    if not runs:
        return None
    runs.sort(key=lambda r: os.path.getmtime(r), reverse=True)
    return os.path.basename(runs[0])


def auto_detect_rollout_dir(logdir):
    """自动探测 rollout 目录。优先 test_rollouts/，其次 rollouts/，最后 visual_rollouts/。"""
    candidates = ["test_rollouts", "rollouts", "visual_rollouts"]
    for name in candidates:
        path = os.path.join(logdir, name)
        if os.path.isdir(path):
            npz_files = glob.glob(os.path.join(path, "*.npz"))
            if npz_files:
                return path
    return None


# ══════════════════════════════════════════════════════════════════════════════
# npz 加载与键自动探测
# ══════════════════════════════════════════════════════════════════════════════

def load_npz_rollouts(rollout_dir, n_max=20):
    """加载 rollout_dir 中所有 .npz 文件，自动探测所有键。"""
    npz_files = sorted(glob.glob(os.path.join(rollout_dir, "*.npz")))
    if not npz_files:
        return [], set()
    if n_max and len(npz_files) > n_max:
        # 取最近的 n_max 个
        npz_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
        npz_files = npz_files[:n_max]
        npz_files.sort()

    rollouts = []
    all_keys = set()
    for f in npz_files:
        data = np.load(f, allow_pickle=True)
        rollout = {"file": os.path.basename(f), "path": f}
        for k in data.keys():
            rollout[k] = data[k]
            all_keys.add(k)
        rollouts.append(rollout)
    return rollouts, all_keys


def find_key(available_keys, patterns):
    """在 available_keys 中查找第一个匹配任一 pattern（子串，大小写不敏感）的键。"""
    for pat in patterns:
        for k in available_keys:
            if pat.lower() in k.lower():
                return k
    return None


def find_reward_component_keys(available_keys):
    """
    识别各类奖励分量键，兼容不同命名：
      - reward_reach / lift_reaching_reward / reward_reaching
      - reward_grasp / lift_grasping_reward / reward_grasping
      - reward_lift / lift_success_reward / reward_success
    返回 dict: {'reach': key_or_None, 'grasp': ..., 'lift': ..., 'original': ..., 'shaped': ...}
    """
    mapping = {
        "reach": ["reward_reach", "lift_reaching_reward", "reward_reaching", "reach_reward"],
        "grasp": ["reward_grasp", "lift_grasping_reward", "reward_grasping", "grasp_reward",
                  "reward_grip_close", "reward_gripper_close"],
        "lift": ["reward_lift", "lift_success_reward", "reward_success", "lift_reward"],
        "original": ["original_reward"],
        "shaped": ["shaped_reward"],
    }
    result = {}
    for comp, pats in mapping.items():
        result[comp] = find_key(available_keys, pats)
    return result


def find_eef_keys(available_keys):
    """探测 EEF/cube 位置相关键（若 npz 中直接存在）。"""
    return {
        "eef_pos": find_key(available_keys, ["eef_pos", "eef_position", "robot0_eef_pos"]),
        "cube_pos": find_key(available_keys, ["cube_pos", "object_pos", "obj_pos"]),
        "eef_cube_dist": find_key(available_keys, ["eef_cube_dist", "eef_obj_dist", "dist"]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 1. TensorBoard 分析
# ══════════════════════════════════════════════════════════════════════════════

def analyze_tensorboard(run_dir, output_dir, run_name):
    """读取 TensorBoard 事件文件，打印汇总表并绘制多面板图。"""
    print("\n" + "=" * 80)
    print("【1】TensorBoard 分析")
    print("=" * 80)

    if not HAS_TB:
        print("  [警告] 未安装 tensorboard，跳过 TensorBoard 分析。")
        return None

    event_files = glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))
    if not event_files:
        print(f"  [警告] 在 {run_dir} 中未找到 TensorBoard 事件文件。")
        return None

    print(f"  事件文件: {os.path.basename(event_files[0])}")
    ea = EventAccumulator(run_dir)
    ea.Reload()
    all_tags = ea.Tags()

    # 打印所有 tag 分类
    print(f"\n  可用 tag 分类:")
    for category, tags in all_tags.items():
        if tags:
            print(f"    {category}: {tags}")

    scalar_tags = all_tags.get("scalars", [])
    if not scalar_tags:
        print("  [警告] 未找到 scalar tag。")
        return None

    # 提取每个 tag 的统计信息
    tag_stats = []
    print(f"\n  {'─' * 76}")
    print(f"  {'指标':<45} {'初始':>10} {'最终':>10} {'最大':>10} {'@step':>8}")
    print(f"  {'─' * 76}")

    for tag in scalar_tags:
        events = ea.Scalars(tag)
        if not events:
            continue
        steps = [e.step for e in events]
        values = [e.value for e in events]
        initial_val = values[0]
        final_val = values[-1]
        max_val = max(values)
        max_step = steps[values.index(max_val)]
        min_val = min(values)

        tag_stats.append({
            "tag": tag,
            "steps": steps,
            "values": values,
            "initial": initial_val,
            "final": final_val,
            "max": max_val,
            "max_step": max_step,
            "min": min_val,
            "n": len(values),
        })
        print(f"  {tag:<45} {initial_val:>10.4f} {final_val:>10.4f} {max_val:>10.4f} {max_step:>8}")
    print(f"  {'─' * 76}")

    # 绘制多面板图
    plot_tensorboard_metrics(tag_stats, output_dir, run_name)
    return tag_stats


def plot_tensorboard_metrics(tag_stats, output_dir, run_name):
    """绘制 TensorBoard 关键指标多面板图。"""
    if not tag_stats:
        return

    # 选择优先绘制的指标分组
    priority_groups = [
        ("rollout", ["rollout/ep_rew_mean", "rollout/ep_len_mean"]),
        ("eval", ["eval/mean_reward", "eval/mean_ep_length"]),
        ("train", ["train/actor_loss", "train/critic_loss", "train/ent_coef", "train/learning_rate"]),
        ("reward_breakdown", None),  # 全部 reward_breakdown/* tag
    ]

    tag_by_name = {s["tag"]: s for s in tag_stats}
    plotted_tags = set()
    panels = []  # list of (title, [tag_stats...])

    for group_name, specific_tags in priority_groups:
        if specific_tags is not None:
            tags_in_group = [tag_by_name[t] for t in specific_tags if t in tag_by_name]
        else:
            prefix = group_name + "/"
            tags_in_group = [s for s in tag_stats if s["tag"].startswith(prefix)]
        if tags_in_group:
            panels.append((group_name, tags_in_group))
            for s in tags_in_group:
                plotted_tags.add(s["tag"])

    # 把剩余未分组的也加入
    remaining = [s for s in tag_stats if s["tag"] not in plotted_tags]
    if remaining:
        panels.append(("other", remaining))

    if not panels:
        return

    n_panels = len(panels)
    n_cols = 2
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), squeeze=False)

    for idx, (title, stats_list) in enumerate(panels):
        ax = axes[idx // n_cols][idx % n_cols]
        for s in stats_list:
            short_name = s["tag"].split("/")[-1]
            ax.plot(s["steps"], s["values"], label=short_name, alpha=0.8)
        ax.set_title(f"{title} (n={len(stats_list)})")
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # 隐藏空余子图
    for idx in range(n_panels, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.suptitle(f"{run_name} TensorBoard Metrics", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out_path = os.path.join(output_dir, "tensorboard_metrics.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n  已保存: {os.path.basename(out_path)}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Rollout 轨迹分析
# ══════════════════════════════════════════════════════════════════════════════

def get_rollout_length(rollout):
    """获取 rollout 的有效长度。"""
    if "length" in rollout:
        return int(rollout["length"])
    # 回退：从 rewards / actions 推断
    for k in ["rewards", "actions", "observations"]:
        if k in rollout and hasattr(rollout[k], "shape") and rollout[k].shape:
            return rollout[k].shape[0]
    return 0


def plot_aggregate_trajectories(rollouts, reward_keys, eef_keys, output_dir, run_name):
    """绘制聚合轨迹（mean ± std）。"""
    if not rollouts:
        return

    # 收集要绘制的轨迹
    panels = []  # list of (title, ylabel, [(label, array(n_rollouts, min_len)), ...])

    # 1. 奖励分量
    reward_components = []
    for comp, key in reward_keys.items():
        if key is None:
            continue
        arrs = []
        for r in rollouts:
            T = get_rollout_length(r)
            if key in r and T > 0:
                arrs.append(np.asarray(r[key][:T]).reshape(-1))
        if arrs:
            min_len = min(len(a) for a in arrs)
            mat = np.array([a[:min_len] for a in arrs])
            reward_components.append((key, mat))
    if reward_components:
        panels.append(("Reward Components", "Reward", reward_components))

    # 2. EEF 相关（若可用）
    eef_panels = []
    if eef_keys.get("eef_cube_dist"):
        key = eef_keys["eef_cube_dist"]
        arrs = []
        for r in rollouts:
            T = get_rollout_length(r)
            if key in r and T > 0:
                arrs.append(np.asarray(r[key][:T]).reshape(-1))
        if arrs:
            min_len = min(len(a) for a in arrs)
            mat = np.array([a[:min_len] for a in arrs])
            eef_panels.append((key, mat))
    if eef_keys.get("eef_pos"):
        key = eef_keys["eef_pos"]
        arrs = []
        for r in rollouts:
            T = get_rollout_length(r)
            if key in r and T > 0:
                arr = np.asarray(r[key][:T])
                if arr.ndim == 2 and arr.shape[1] >= 3:
                    arrs.append(arr[:, 2])  # z 分量
        if arrs:
            min_len = min(len(a) for a in arrs)
            mat = np.array([a[:min_len] for a in arrs])
            eef_panels.append(("EEF z", mat))
    if eef_panels:
        panels.append(("EEF Trajectory", "Value", eef_panels))

    # 3. 动作各维度
    action_panels = []
    for r in rollouts:
        if "actions" not in r:
            break
    else:
        action_dim = None
        action_arrs_by_dim = {}
        for r in rollouts:
            T = get_rollout_length(r)
            if "actions" in r and T > 0:
                acts = np.asarray(r["actions"][:T])
                if acts.ndim == 2:
                    if action_dim is None:
                        action_dim = acts.shape[1]
                        action_arrs_by_dim = {d: [] for d in range(action_dim)}
                    for d in range(min(action_dim, acts.shape[1])):
                        action_arrs_by_dim[d].append(acts[:, d])
        if action_dim:
            for d in range(action_dim):
                arrs = action_arrs_by_dim.get(d, [])
                if not arrs:
                    continue
                min_len = min(len(a) for a in arrs)
                mat = np.array([a[:min_len] for a in arrs])
                label = f"gripper" if d == action_dim - 1 and action_dim == 6 else f"dim{d}"
                action_panels.append((label, mat))
    if action_panels:
        panels.append(("Actions per Dim", "Action", action_panels))

    if not panels:
        print("  [警告] 没有可绘制的聚合轨迹数据。")
        return

    # 绘制
    n_panels = len(panels)
    n_cols = 2
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), squeeze=False)

    for idx, (title, ylabel, items) in enumerate(panels):
        ax = axes[idx // n_cols][idx % n_cols]
        for label, mat in items:
            steps = np.arange(mat.shape[1])
            mean = mat.mean(axis=0)
            std = mat.std(axis=0)
            ax.plot(steps, mean, label=label, alpha=0.8)
            ax.fill_between(steps, mean - std, mean + std, alpha=0.2)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for idx in range(n_panels, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.suptitle(f"{run_name} Aggregate Trajectories (mean ± std, n={len(rollouts)})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out_path = os.path.join(output_dir, "aggregate_trajectories.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  已保存: {os.path.basename(out_path)}")


def plot_per_episode_detail(rollouts, reward_keys, eef_keys, output_dir, run_name, n_max=10):
    """为每个 episode 绘制详细多面板图。"""
    if not rollouts:
        return

    n = min(len(rollouts), n_max)
    for i in range(n):
        r = rollouts[i]
        T = get_rollout_length(r)
        if T == 0:
            continue
        steps = np.arange(T)

        # 决定子图数量
        has_actions = "actions" in r and np.asarray(r["actions"]).ndim == 2
        has_rewards = any(k and k in r for k in reward_keys.values()) or "rewards" in r
        has_eef = eef_keys.get("eef_pos") and eef_keys["eef_pos"] in r

        n_rows = 1
        if has_actions:
            n_rows += 1
        if has_rewards:
            n_rows += 1
        if has_eef:
            n_rows += 1

        fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3.5 * n_rows), squeeze=False)
        row = 0

        # 奖励分量
        if has_rewards:
            ax = axes[row][0]
            row += 1
            plotted = False
            for comp, key in reward_keys.items():
                if key and key in r:
                    ax.plot(steps, np.asarray(r[key][:T]).reshape(-1), label=key, alpha=0.8)
                    plotted = True
            if "rewards" in r and not plotted:
                ax.plot(steps, np.asarray(r["rewards"][:T]), label="rewards", alpha=0.8, color="b")
                plotted = True
            elif "rewards" in r:
                ax.plot(steps, np.asarray(r["rewards"][:T]), label="rewards(total)",
                        alpha=0.5, color="k", linestyle="--")
            ax.set_title("Reward Components")
            ax.set_ylabel("Reward")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        # 动作
        if has_actions:
            ax = axes[row][0]
            row += 1
            acts = np.asarray(r["actions"][:T])
            action_dim = acts.shape[1]
            for d in range(action_dim):
                label = f"gripper" if d == action_dim - 1 and action_dim == 6 else f"dim{d}"
                ax.plot(steps, acts[:, d], label=label, alpha=0.7)
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
            ax.set_title("Actions")
            ax.set_ylabel("Action")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        # EEF 位置
        if has_eef:
            ax = axes[row][0]
            row += 1
            eef = np.asarray(r[eef_keys["eef_pos"]][:T])
            if eef.ndim == 2 and eef.shape[1] >= 3:
                ax.plot(steps, eef[:, 0], label="EEF x", alpha=0.7)
                ax.plot(steps, eef[:, 1], label="EEF y", alpha=0.7)
                ax.plot(steps, eef[:, 2], label="EEF z", alpha=0.7)
            if eef_keys.get("cube_pos") and eef_keys["cube_pos"] in r:
                cube = np.asarray(r[eef_keys["cube_pos"]][:T])
                if cube.ndim == 2 and cube.shape[1] >= 3:
                    ax.plot(steps, cube[:, 2], "r--", label="Cube z", alpha=0.8)
            ax.set_title("EEF / Cube Position")
            ax.set_ylabel("Position (m)")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        # 最后一个子图：success/total_reward 文本信息
        ax = axes[row][0]
        row += 1
        ax.axis("off")
        success = bool(r["success"]) if "success" in r else "N/A"
        total_reward = float(r["total_reward"]) if "total_reward" in r else float("nan")
        length = T
        info_lines = [
            f"Episode {i}: {r['file']}",
            f"  success: {success}",
            f"  total_reward: {total_reward:.2f}",
            f"  length: {length}",
        ]
        ax.text(0.05, 0.5, "\n".join(info_lines), transform=ax.transAxes,
                fontsize=11, verticalalignment="center", fontfamily="monospace")

        axes[-1][0].set_xlabel("Step")
        fig.suptitle(f"{run_name} Episode {i} Detail "
                     f"(reward={total_reward:.1f}, success={success})",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"episode_{i:02d}_detail.png"), dpi=150)
        plt.close(fig)

    print(f"  已保存: episode_*_detail.png (共 {n} 个)")


# ══════════════════════════════════════════════════════════════════════════════
# 3. 关键诊断
# ══════════════════════════════════════════════════════════════════════════════

def compute_diagnostics(rollouts, reward_keys, eef_keys):
    """计算关键诊断指标。返回 dict。"""
    n_rollouts = len(rollouts)
    if n_rollouts == 0:
        return {}

    diag = {}

    # ── 成功率 ──
    success_count = 0
    for r in rollouts:
        if "success" in r and bool(r["success"]):
            success_count += 1
    diag["success_rate"] = success_count / n_rollouts
    diag["success_count"] = success_count
    diag["n_rollouts"] = n_rollouts

    # ── 总奖励统计 ──
    total_rewards = [float(r["total_reward"]) for r in rollouts if "total_reward" in r]
    if total_rewards:
        diag["total_reward_mean"] = float(np.mean(total_rewards))
        diag["total_reward_max"] = float(np.max(total_rewards))
        diag["total_reward_min"] = float(np.min(total_rewards))

    # ── 长度统计 ──
    lengths = [get_rollout_length(r) for r in rollouts]
    diag["length_mean"] = float(np.mean(lengths))
    diag["length_max"] = int(np.max(lengths))
    diag["length_min"] = int(np.min(lengths))
    total_steps = sum(lengths)
    diag["total_steps"] = total_steps

    # ── 抓取率（grasp reward > 0 的步数比例）──
    grasp_key = reward_keys.get("grasp")
    if grasp_key:
        grasp_steps = 0
        for r in rollouts:
            T = get_rollout_length(r)
            if grasp_key in r and T > 0:
                grasp_arr = np.asarray(r[grasp_key][:T]).reshape(-1)
                grasp_steps += int((grasp_arr > 0).sum())
        diag["grasp_key"] = grasp_key
        diag["grasp_steps"] = grasp_steps
        diag["grasp_rate"] = grasp_steps / max(total_steps, 1)
        # 有抓取的 episode 数
        grasp_eps = 0
        for r in rollouts:
            T = get_rollout_length(r)
            if grasp_key in r and T > 0:
                if (np.asarray(r[grasp_key][:T]).reshape(-1) > 0).any():
                    grasp_eps += 1
        diag["grasp_episode_count"] = grasp_eps

    # ── reach 奖励统计 ──
    reach_key = reward_keys.get("reach")
    if reach_key:
        reach_vals = []
        for r in rollouts:
            T = get_rollout_length(r)
            if reach_key in r and T > 0:
                reach_vals.append(np.asarray(r[reach_key][:T]).reshape(-1))
        if reach_vals:
            all_reach = np.concatenate(reach_vals)
            diag["reach_key"] = reach_key
            diag["reach_mean"] = float(all_reach.mean())
            diag["reach_max"] = float(all_reach.max())
            reach_steps = int((all_reach > 0.01).sum())
            diag["reach_steps"] = reach_steps
            diag["reach_rate"] = reach_steps / max(total_steps, 1)

    # ── lift/success 奖励统计 ──
    lift_key = reward_keys.get("lift")
    if lift_key:
        lift_vals = []
        for r in rollouts:
            T = get_rollout_length(r)
            if lift_key in r and T > 0:
                lift_vals.append(np.asarray(r[lift_key][:T]).reshape(-1))
        if lift_vals:
            all_lift = np.concatenate(lift_vals)
            diag["lift_key"] = lift_key
            diag["lift_mean"] = float(all_lift.mean())
            diag["lift_max"] = float(all_lift.max())
            lift_steps = int((all_lift > 0).sum())
            diag["lift_steps"] = lift_steps

    # ── original_reward 统计 ──
    orig_key = reward_keys.get("original")
    if orig_key:
        orig_vals = []
        for r in rollouts:
            T = get_rollout_length(r)
            if orig_key in r and T > 0:
                orig_vals.append(np.asarray(r[orig_key][:T]).reshape(-1))
        if orig_vals:
            all_orig = np.concatenate(orig_vals)
            diag["orig_key"] = orig_key
            diag["orig_mean"] = float(all_orig.mean())
            diag["orig_max"] = float(all_orig.max())
            diag["orig_min"] = float(all_orig.min())

    # ── EEF-cube 距离统计（若可观测）──
    dist_key = eef_keys.get("eef_cube_dist")
    if dist_key:
        min_dists = []
        all_dists = []
        for r in rollouts:
            T = get_rollout_length(r)
            if dist_key in r and T > 0:
                d = np.asarray(r[dist_key][:T]).reshape(-1)
                min_dists.append(float(d.min()))
                all_dists.append(d)
        if min_dists:
            diag["dist_key"] = dist_key
            diag["min_dist_mean"] = float(np.mean(min_dists))
            diag["min_dist_min"] = float(np.min(min_dists))
            diag["min_dist_per_ep"] = min_dists
            all_d = np.concatenate(all_dists)
            diag["dist_mean"] = float(all_d.mean())
            diag["close_5cm_steps"] = int((all_d < 0.05).sum())
            diag["close_2cm_steps"] = int((all_d < 0.02).sum())
    elif eef_keys.get("eef_pos") and eef_keys.get("cube_pos"):
        # 从 EEF 和 cube 位置计算距离
        eef_k = eef_keys["eef_pos"]
        cube_k = eef_keys["cube_pos"]
        min_dists = []
        all_dists = []
        for r in rollouts:
            T = get_rollout_length(r)
            if eef_k in r and cube_k in r and T > 0:
                eef = np.asarray(r[eef_k][:T])
                cube = np.asarray(r[cube_k][:T])
                if eef.ndim == 2 and cube.ndim == 2 and eef.shape[0] == cube.shape[0]:
                    d = np.linalg.norm(eef - cube, axis=1)
                    min_dists.append(float(d.min()))
                    all_dists.append(d)
        if min_dists:
            diag["dist_computed"] = True
            diag["min_dist_mean"] = float(np.mean(min_dists))
            diag["min_dist_min"] = float(np.min(min_dists))
            diag["min_dist_per_ep"] = min_dists
            all_d = np.concatenate(all_dists)
            diag["dist_mean"] = float(all_d.mean())
            diag["close_5cm_steps"] = int((all_d < 0.05).sum())
            diag["close_2cm_steps"] = int((all_d < 0.02).sum())

    # ── 动作统计（每维度）──
    action_stats = []
    action_dim = None
    for r in rollouts:
        if "actions" not in r:
            continue
        acts = np.asarray(r["actions"])
        if acts.ndim == 2:
            if action_dim is None:
                action_dim = acts.shape[1]
            if acts.shape[1] == action_dim:
                action_stats.append(acts)
    if action_stats:
        all_actions = np.concatenate(action_stats, axis=0)
        diag["action_dim"] = action_dim
        diag["action_stats"] = []
        for d in range(action_dim):
            col = all_actions[:, d]
            label = f"gripper" if d == action_dim - 1 and action_dim == 6 else f"dim{d}"
            diag["action_stats"].append({
                "dim": d,
                "label": label,
                "min": float(col.min()),
                "max": float(col.max()),
                "mean": float(col.mean()),
                "std": float(col.std()),
            })

    return diag


def print_diagnostics(diag):
    """打印关键诊断结果。"""
    print("\n" + "=" * 80)
    print("【3】关键诊断")
    print("=" * 80)

    if not diag:
        print("  无诊断数据。")
        return

    print(f"\n  Rollout 数量: {diag.get('n_rollouts', 0)}")
    print(f"  总步数:       {diag.get('total_steps', 0)}")
    print(f"  长度:         mean={diag.get('length_mean', 0):.1f}, "
          f"min={diag.get('length_min', 0)}, max={diag.get('length_max', 0)}")

    print(f"\n  -- 成功率 --")
    print(f"  success: {diag.get('success_count', 0)}/{diag.get('n_rollouts', 0)} "
          f"= {diag.get('success_rate', 0)*100:.1f}%")

    if "total_reward_mean" in diag:
        print(f"\n  -- 总奖励 --")
        print(f"  mean={diag['total_reward_mean']:.2f}, "
              f"min={diag['total_reward_min']:.2f}, max={diag['total_reward_max']:.2f}")

    if "grasp_key" in diag:
        print(f"\n  -- 抓取率 -- (key={diag['grasp_key']})")
        print(f"  grasp steps: {diag['grasp_steps']}/{diag['total_steps']} "
              f"= {diag['grasp_rate']*100:.1f}%")
        print(f"  有抓取的 episode: {diag['grasp_episode_count']}/{diag['n_rollouts']}")

    if "reach_key" in diag:
        print(f"\n  -- Reach 奖励 -- (key={diag['reach_key']})")
        print(f"  mean={diag['reach_mean']:.4f}, max={diag['reach_max']:.4f}")
        print(f"  reach (>0.01) steps: {diag['reach_steps']}/{diag['total_steps']} "
              f"= {diag['reach_rate']*100:.1f}%")

    if "lift_key" in diag:
        print(f"\n  -- Lift 奖励 -- (key={diag['lift_key']})")
        print(f"  mean={diag['lift_mean']:.4f}, max={diag['lift_max']:.4f}")
        print(f"  lift (>0) steps: {diag['lift_steps']}/{diag['total_steps']}")

    if "orig_key" in diag:
        print(f"\n  -- original_reward -- (key={diag['orig_key']})")
        print(f"  mean={diag['orig_mean']:.4f}, min={diag['orig_min']:.4f}, "
              f"max={diag['orig_max']:.4f}")

    if "min_dist_mean" in diag:
        src = diag.get("dist_key", "computed from eef_pos & cube_pos")
        print(f"\n  -- EEF-Cube 距离 -- (source: {src})")
        print(f"  每 episode 最小距离均值: {diag['min_dist_mean']*100:.2f} cm")
        print(f"  全局最小距离:           {diag['min_dist_min']*100:.2f} cm")
        if "min_dist_per_ep" in diag:
            for i, d in enumerate(diag["min_dist_per_ep"]):
                print(f"    ep{i}: {d*100:.2f} cm")
        if "close_5cm_steps" in diag:
            print(f"  <5cm 步数: {diag['close_5cm_steps']}/{diag['total_steps']}")
            print(f"  <2cm 步数: {diag['close_2cm_steps']}/{diag['total_steps']}")
    else:
        print(f"\n  -- EEF-Cube 距离 --")
        print(f"  [不可观测] npz 中未找到 EEF/cube 位置键。")

    if "action_stats" in diag:
        print(f"\n  -- 动作统计 (dim={diag['action_dim']}) --")
        print(f"  {'dim':<4} {'label':<10} {'min':>8} {'max':>8} {'mean':>8} {'std':>8}")
        for a in diag["action_stats"]:
            print(f"  {a['dim']:<4} {a['label']:<10} {a['min']:>8.3f} {a['max']:>8.3f} "
                  f"{a['mean']:>8.3f} {a['std']:>8.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 诊断报告生成
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(rollouts, all_keys, reward_keys, eef_keys, diag, run_name,
                    logdir, rollout_dir, output_dir):
    """生成文字诊断报告。"""
    report = []
    report.append("=" * 80)
    report.append(f"SAC 训练分析报告 (通用工具)")
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Log 目录: {logdir}")
    report.append(f"运行:     {run_name}")
    report.append(f"Rollout 目录: {rollout_dir}")
    report.append("=" * 80)

    # 基本信息
    report.append("\n## 1. 基本信息")
    report.append(f"  Rollout 数量: {diag.get('n_rollouts', 0)}")
    report.append(f"  总步数: {diag.get('total_steps', 0)}")
    if "total_reward_mean" in diag:
        report.append(f"  平均总奖励: {diag['total_reward_mean']:.2f} "
                      f"(min={diag['total_reward_min']:.2f}, max={diag['total_reward_max']:.2f})")
    report.append(f"  平均长度: {diag.get('length_mean', 0):.1f}")

    # npz 键
    report.append("\n## 2. npz 文件键自动探测")
    report.append(f"  检测到的所有键: {sorted(all_keys)}")
    report.append(f"  奖励分量映射:")
    for comp, key in reward_keys.items():
        report.append(f"    {comp}: {key}")
    report.append(f"  EEF/位置键映射:")
    for comp, key in eef_keys.items():
        report.append(f"    {comp}: {key}")

    # 关键诊断
    report.append("\n## 3. 关键诊断")
    report.append(f"  成功率: {diag.get('success_count', 0)}/{diag.get('n_rollouts', 0)} "
                  f"= {diag.get('success_rate', 0)*100:.1f}%")
    if "grasp_key" in diag:
        report.append(f"  抓取率: {diag['grasp_steps']}/{diag['total_steps']} "
                      f"= {diag['grasp_rate']*100:.1f}% (key={diag['grasp_key']})")
        report.append(f"  有抓取的 episode: {diag['grasp_episode_count']}/{diag['n_rollouts']}")
    if "reach_key" in diag:
        report.append(f"  Reach 率: {diag['reach_steps']}/{diag['total_steps']} "
                      f"= {diag['reach_rate']*100:.1f}%")
    if "min_dist_mean" in diag:
        report.append(f"  EEF-Cube 最小距离均值: {diag['min_dist_mean']*100:.2f} cm")
        report.append(f"  EEF-Cube 全局最小距离: {diag['min_dist_min']*100:.2f} cm")
    else:
        report.append(f"  EEF-Cube 距离: [不可观测]")

    # 动作统计
    if "action_stats" in diag:
        report.append(f"\n## 4. 动作统计 (dim={diag['action_dim']})")
        for a in diag["action_stats"]:
            report.append(f"  dim {a['dim']} ({a['label']}): "
                          f"min={a['min']:.3f}, max={a['max']:.3f}, "
                          f"mean={a['mean']:.3f}, std={a['std']:.3f}")

    # 结论
    report.append("\n## 5. 结论")
    sr = diag.get("success_rate", 0)
    if sr > 0:
        report.append(f"  ✅ 策略达到 {sr*100:.1f}% 成功率。")
    else:
        report.append(f"  ❌ 策略在 {diag.get('n_rollouts', 0)} 个 rollout 中未成功。")
        if "grasp_rate" in diag:
            if diag["grasp_rate"] == 0:
                report.append(f"  → 策略从未抓取（{diag['grasp_key']} 始终为 0）。")
                if "min_dist_mean" in diag:
                    if diag["min_dist_min"] > 0.05:
                        report.append(f"  → 根因: EEF 从未足够靠近 cube（最小距离 "
                                      f"{diag['min_dist_min']*100:.2f}cm > 5cm）。")
                    else:
                        report.append(f"  → 根因: EEF 靠近 cube 但夹爪未闭合。")
            else:
                report.append(f"  → 策略有抓取信号（{diag['grasp_rate']*100:.1f}% 步）但未能完成任务。")

    report_text = "\n".join(report)
    report_path = os.path.join(output_dir, "diagnostic_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\n{report_text}")
    print(f"\n报告已保存至: {report_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="通用 SAC 训练分析工具（适用于任意 SAC 运行）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--logdir", required=True,
                        help="训练日志目录（如 ./logs/sac_lift_so101_realistic/）")
    parser.add_argument("--run", default=None,
                        help="指定 SAC 运行名（如 SAC_21）。未指定则使用最近的。")
    parser.add_argument("--rollout-dir", default=None,
                        help="Rollout 目录。未指定则自动探测 test_rollouts/ 或 rollouts/。")
    parser.add_argument("--output-dir", default="./tools/analysis_output/",
                        help="分析图保存目录。默认: ./tools/analysis_output/")
    parser.add_argument("--n-rollouts", type=int, default=20,
                        help="最多分析的 rollout 数量（取最近）。默认: 20")
    return parser.parse_args()


def main():
    args = parse_args()

    logdir = os.path.abspath(args.logdir)
    if not os.path.isdir(logdir):
        print(f"[错误] logdir 不存在: {logdir}")
        sys.exit(1)

    # 确定运行名
    run_name = args.run
    if run_name is None:
        run_name = find_most_recent_run(logdir)
        if run_name is None:
            print(f"[错误] 在 {logdir} 中未找到 SAC_* 运行目录。")
            sys.exit(1)
        print(f"[信息] 未指定 --run，使用最近的运行: {run_name}")

    run_dir = os.path.join(logdir, run_name)
    if not os.path.isdir(run_dir):
        print(f"[错误] 运行目录不存在: {run_dir}")
        sys.exit(1)

    # 确定 rollout 目录
    rollout_dir = args.rollout_dir
    if rollout_dir is None:
        rollout_dir = auto_detect_rollout_dir(logdir)
        if rollout_dir is None:
            print(f"[警告] 未自动探测到 rollout 目录（test_rollouts/、rollouts/ 等）。")
        else:
            print(f"[信息] 未指定 --rollout-dir，自动探测到: {rollout_dir}")
    else:
        rollout_dir = os.path.abspath(rollout_dir)
        if not os.path.isdir(rollout_dir):
            print(f"[警告] 指定的 rollout-dir 不存在: {rollout_dir}")
            rollout_dir = None

    # 输出目录
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("通用 SAC 训练分析")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log 目录:   {logdir}")
    print(f"运行:       {run_name}")
    print(f"Rollout 目录: {rollout_dir}")
    print(f"输出目录:   {output_dir}")
    print("=" * 80)

    # ── 1. TensorBoard 分析 ──
    analyze_tensorboard(run_dir, output_dir, run_name)

    # ── 2 & 3. Rollout 分析 + 诊断 ──
    if rollout_dir is None:
        print("\n[警告] 无 rollout 目录，跳过 rollout 分析。")
        print(f"\n完成。结果保存至: {output_dir}/")
        return

    print("\n" + "=" * 80)
    print("【2】Rollout 轨迹分析")
    print("=" * 80)

    rollouts, all_keys = load_npz_rollouts(rollout_dir, n_max=args.n_rollouts)
    print(f"  加载 {len(rollouts)} 个 .npz 文件")
    print(f"  检测到的键: {sorted(all_keys)}")

    if not rollouts:
        print("[警告] 未加载到任何 rollout 数据。")
        return

    # 自动探测奖励分量键和 EEF 键
    reward_keys = find_reward_component_keys(all_keys)
    eef_keys = find_eef_keys(all_keys)
    print(f"\n  奖励分量映射:")
    for comp, key in reward_keys.items():
        print(f"    {comp}: {key}")
    print(f"  EEF/位置键映射:")
    for comp, key in eef_keys.items():
        print(f"    {comp}: {key}")

    # 绘制聚合轨迹
    plot_aggregate_trajectories(rollouts, reward_keys, eef_keys, output_dir, run_name)

    # 绘制逐 episode 详细图
    plot_per_episode_detail(rollouts, reward_keys, eef_keys, output_dir, run_name)

    # ── 3. 关键诊断 ──
    diag = compute_diagnostics(rollouts, reward_keys, eef_keys)
    print_diagnostics(diag)

    # ── 生成报告 ──
    generate_report(rollouts, all_keys, reward_keys, eef_keys, diag, run_name,
                    logdir, rollout_dir, output_dir)

    print("\n" + "=" * 80)
    print("分析完成！所有结果已保存至:")
    print(f"  {output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
