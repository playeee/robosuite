"""
离线 Rollout 可视化脚本：无需模型，直接从 .npz 文件绘制诊断图。

支持两种 rollout 格式：
  1. 训练 rollout（RolloutCollectorCallback 保存的）
  2. 测试 rollout（训练脚本测试阶段保存的）
  3. 可视化 rollout（visualize_rollout_so101.py 保存的）

功能：
  - 绘制奖励分量堆叠图
  - 绘制 EEF / cube 高度曲线（从观测中提取或从额外字段读取）
  - 绘制动作曲线
  - 输出关键统计指标

运行方式：
    conda run -n robosuite python robosuite/demos/visualize_rollout_offline.py \
        ./logs/sac_lift_so101_realistic/test_rollouts/rollout_test_ep000.npz

    # 或批量分析整个目录
    conda run -n robosuite python robosuite/demos/visualize_rollout_offline.py \
        ./logs/sac_lift_so101_realistic/test_rollouts/

    # 分析训练 rollout
    conda run -n robosuite python robosuite/demos/visualize_rollout_offline.py \
        ./logs/sac_lift_so101_realistic/rollouts/rollout_train_step01499200_ep0936.npz
"""

import argparse
import os
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("需要 matplotlib：conda run -n robosuite pip install matplotlib")
    sys.exit(1)


def load_rollout(path):
    """加载 .npz rollout 文件。"""
    data = np.load(path, allow_pickle=True)
    return data


def analyze_and_plot(data, path, output_dir=None):
    """分析并绘制单个 rollout 的诊断图。"""
    name = os.path.splitext(os.path.basename(path))[0]
    if output_dir is None:
        output_dir = os.path.dirname(path)

    # ---- 基本数据 ----
    rewards = data["rewards"] if "rewards" in data else None
    actions = data["actions"] if "actions" in data else None
    observations = data["observations"] if "observations" in data else None
    T = len(rewards) if rewards is not None else 0

    # ---- 额外字段（可视化 rollout 特有）----
    eef_heights = data["eef_heights"] if "eef_heights" in data else None
    gripper_openings = data["gripper_openings"] if "gripper_openings" in data else None
    cube_heights = data["cube_heights"] if "cube_heights" in data else None

    # ---- 奖励分量（自动检测所有含 "reward" 或 "lift_" 前缀的键）----
    # 区分两类：
    #   - 聚合键（aggregate）：original_reward, shaped_reward, rewards, total_reward
    #     这些是各分量的总和或原始任务奖励，量级较大（~2.25），单独画一个子图。
    #   - 分量键（component）：reward_reach, reward_grasp, reward_grip_close, lift_* 等
    #     这些是各 shaping 分量，量级较小（~0.01~0.5），单独画堆叠图 + 逐条曲线图。
    # 混在一起会导致大值压扁小值，小分量曲线看不清。
    AGGREGATE_KEYS = {"original_reward", "shaped_reward", "rewards", "total_reward",
                      "reward", "total", "shaped", "original"}
    reward_comps = {}      # 分量键
    reward_aggregates = {}  # 聚合键
    if T > 0:
        for key in data.files:
            lower = key.lower()
            if "reward" in lower or lower.startswith("lift_"):
                try:
                    arr = np.array(data[key], dtype=np.float64).flatten()
                    # 只接受与时间步长度匹配的一维时间序列
                    if arr.ndim == 1 and len(arr) == T:
                        if lower in AGGREGATE_KEYS:
                            reward_aggregates[key] = arr
                        else:
                            reward_comps[key] = arr
                except Exception:
                    pass

    # ---- 统计输出 ----
    print(f"\n{'='*60}")
    print(f"Rollout: {name}")
    print(f"{'='*60}")
    if rewards is not None:
        print(f"  长度: {T}, 总奖励: {rewards.sum():.2f}, 平均/步: {rewards.mean():.4f}")
    if "success" in data:
        print(f"  成功: {bool(data['success'])}")
    if "total_reward" in data:
        print(f"  total_reward: {float(data['total_reward']):.2f}")

    # 奖励分量统计
    if reward_aggregates:
        print("\n  聚合奖励统计:")
        for key, vals in reward_aggregates.items():
            arr = np.array(vals, dtype=np.float64).flatten()
            if len(arr) > 0:
                nonzero = np.count_nonzero(arr)
                print(f"    {key:28s}: sum={arr.sum():+8.3f}  mean={arr.mean():+8.5f}  "
                      f"max={arr.max():+8.5f}  nonzero={nonzero}/{len(arr)}")
    if reward_comps:
        print("\n  奖励分量统计:")
        for key, vals in reward_comps.items():
            arr = np.array(vals, dtype=np.float64).flatten()
            if len(arr) > 0:
                nonzero = np.count_nonzero(arr)
                print(f"    {key:28s}: sum={arr.sum():+8.3f}  mean={arr.mean():+8.5f}  "
                      f"max={arr.max():+8.5f}  nonzero={nonzero}/{len(arr)}")

    # ---- 绘图 ----
    n_plots = 1  # 动作
    if eef_heights is not None:
        n_plots += 2  # EEF 高度 + cube 高度
    if gripper_openings is not None:
        n_plots += 1

    # 聚合奖励曲线（original_reward / shaped_reward 等，量级大）
    has_aggs = len(reward_aggregates) > 0
    if has_aggs:
        n_plots += 1
    # 分量奖励：堆叠面积图 + 逐条曲线图（量级小，独立 y 轴）
    has_comps = len(reward_comps) > 0
    if has_comps:
        n_plots += 2

    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3.5 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]
    total_str = f"{rewards.sum():.2f}" if rewards is not None else "N/A"
    fig.suptitle(f"Rollout: {name}\n"
                 f"Total Reward={total_str} | "
                 f"Length={T}", fontsize=12)
    ax_idx = 0
    steps = np.arange(T)

    # 0) 聚合奖励曲线（original_reward / shaped_reward，量级 ~2.25）
    if has_aggs:
        ax = axes[ax_idx]
        ax_idx += 1
        for key, vals in reward_aggregates.items():
            ax.plot(steps, vals[:T], linewidth=1.0, label=key)
        ax.axhline(2.25, color="red", linestyle="--", alpha=0.5, label="success (2.25)")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("聚合奖励")
        ax.set_title("聚合奖励（original_reward / shaped_reward）")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    # 0.5) 奖励分量堆叠图（只含分量键，量级小，不会被聚合键压扁）
    if has_comps:
        ax = axes[ax_idx]
        ax_idx += 1
        # 按总和正负动态分组：正向贡献堆叠在上，负向惩罚堆叠在下
        comp_names = list(reward_comps.keys())
        pos_names = [n for n in comp_names if reward_comps[n].sum() >= 0]
        neg_names = [n for n in comp_names if reward_comps[n].sum() < 0]
        if pos_names:
            pos_stack = np.array([reward_comps[n][:T] for n in pos_names])
            ax.stackplot(steps, pos_stack, labels=pos_names, alpha=0.8)
        if neg_names:
            neg_stack = np.array([reward_comps[n][:T] for n in neg_names])
            ax.stackplot(steps, neg_stack, labels=neg_names, alpha=0.6)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("奖励分量")
        ax.set_title("奖励分量堆叠（shaping 分量，不含聚合键）")
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    # 0.7) 奖励分量逐条曲线图（每个分量一条线，独立 y 轴，便于精确对比）
    if has_comps:
        ax = axes[ax_idx]
        ax_idx += 1
        for key, vals in reward_comps.items():
            ax.plot(steps, vals[:T], linewidth=0.8, label=key)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("奖励分量值")
        ax.set_title("Reward Components（逐条曲线，独立 y 轴）")
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    # 1) 动作
    if actions is not None:
        ax = axes[ax_idx]
        ax_idx += 1
        for i in range(actions.shape[1]):
            ax.plot(steps, actions[:, i], linewidth=0.6, label=f"dim{i}")
        ax.set_ylabel("Action")
        ax.set_title("动作（各维度）")
        ax.legend(fontsize=7, ncol=actions.shape[1], loc="upper right")
        ax.grid(True, alpha=0.3)

    # 2) EEF 高度
    if eef_heights is not None:
        ax = axes[ax_idx]
        ax_idx += 1
        ax.plot(steps, eef_heights, color="C1", linewidth=1)
        if cube_heights is not None:
            ax.plot(steps, cube_heights, color="C3", linewidth=1, linestyle="--", label="cube")
            ax.legend(fontsize=8)
        ax.set_ylabel("Height (m)")
        ax.set_title("EEF 高度 (cube 高度虚线)")
        ax.grid(True, alpha=0.3)

    # 4) cube 高度
    if cube_heights is not None and eef_heights is None:
        ax = axes[ax_idx]
        ax_idx += 1
        ax.plot(steps, cube_heights, color="C3", linewidth=1)
        ax.set_ylabel("Cube Z (m)")
        ax.set_title("Cube 高度")
        ax.grid(True, alpha=0.3)

    # 5) 夹爪开合
    if gripper_openings is not None:
        ax = axes[ax_idx]
        ax_idx += 1
        ax.plot(steps, gripper_openings, color="C2", linewidth=1)
        ax.set_ylabel("Gripper")
        ax.set_title("夹爪开合比例 (0=闭合, 1=张开)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Step")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{name}_analysis.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  分析图已保存：{save_path}")
    return save_path


def main():
    parser = argparse.ArgumentParser(description="离线 Rollout 可视化")
    parser.add_argument("path", help=".npz 文件路径或目录路径")
    parser.add_argument("--output-dir", default=None, help="输出图片目录（默认与输入同目录）")
    parser.add_argument("--max-files", type=int, default=10, help="目录模式下最多处理多少个文件")
    args = parser.parse_args()

    path = args.path
    if os.path.isfile(path):
        data = load_rollout(path)
        analyze_and_plot(data, path, args.output_dir)
    elif os.path.isdir(path):
        npz_files = sorted([f for f in os.listdir(path) if f.endswith(".npz")])
        if args.max_files > 0:
            # 均匀采样
            if len(npz_files) > args.max_files:
                indices = np.linspace(0, len(npz_files) - 1, args.max_files, dtype=int)
                npz_files = [npz_files[i] for i in indices]
        print(f"目录模式下，处理 {len(npz_files)} 个文件...")
        for f in npz_files:
            fpath = os.path.join(path, f)
            try:
                data = load_rollout(fpath)
                analyze_and_plot(data, fpath, args.output_dir)
            except Exception as e:
                print(f"  [跳过] {f}: {e}")
    else:
        print(f"路径不存在：{path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
