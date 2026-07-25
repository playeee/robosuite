"""
Rollout 轨迹分析工具。
"""

import json
import os

import numpy as np

from .wrappers import REWARD_COMPONENTS


# =============================================================================
# Rollout 轨迹统计分析
# =============================================================================
def analyze_rollouts(rollout_dir, success_reward_threshold=2.0, save_report=True):
    """
    对 rollout 目录下的 .npz 轨迹文件进行统计分析。

    统计内容：
      - 轨迹数量、成功率
      - 总奖励的均值/标准差/极值
      - 轨迹长度的均值/标准差/极值
      - 单步奖励的分布（均值、标准差、分位数、零/正奖励比例）
      - 动作分布（各维度的均值、标准差、最小值、最大值）
      - 观测分布（各维度的均值、标准差、最小值、最大值）
      - 成功/失败轨迹的对比统计

    参数:
        rollout_dir (str): 轨迹文件目录
        success_reward_threshold (float): 判定为成功 episode 的总奖励阈值
        save_report (bool): 是否保存 JSON 报告

    返回:
        dict or None: 统计报告；未找到轨迹文件时返回 None
    """
    npz_files = sorted([
        f for f in os.listdir(rollout_dir)
        if f.endswith(".npz")
        and (f.startswith("rollout_train_") or f.startswith("rollout_test_"))
    ])
    if len(npz_files) == 0:
        print(f"[Analyze] 未找到轨迹文件：{rollout_dir}")
        return None

    total_rewards = []
    lengths = []
    success_flags = []
    all_rewards = []
    all_actions = []
    all_observations = []
    success_rewards = []
    fail_rewards = []
    # 各奖励分量的累计值（每条轨迹一个标量）；仅当 npz 含分量键时才统计
    component_totals = {name: [] for name in REWARD_COMPONENTS}
    has_components = False

    for fname in npz_files:
        path = os.path.join(rollout_dir, fname)
        data = np.load(path)
        rewards = data["rewards"]
        actions = data["actions"]
        obs = data["observations"]

        # 兼容旧轨迹：没有 success 字段时用奖励阈值判断
        if "success" in data:
            success = bool(data["success"])
        else:
            total_r = float(np.sum(rewards))
            success = total_r >= success_reward_threshold

        total_r = float(data["total_reward"]) if "total_reward" in data else float(np.sum(rewards))
        length = int(data["length"]) if "length" in data else int(len(rewards))

        total_rewards.append(total_r)
        lengths.append(length)
        success_flags.append(success)
        all_rewards.append(rewards)
        all_actions.append(actions)
        all_observations.append(obs)

        # 收集奖励分量累计值（诊断“奖励从哪来”）
        for name in REWARD_COMPONENTS:
            if name in data.files:
                component_totals[name].append(float(np.sum(data[name])))
                has_components = True

        if success:
            success_rewards.append(total_r)
        else:
            fail_rewards.append(total_r)

    all_rewards_arr = np.concatenate(all_rewards)
    all_actions_arr = np.concatenate(all_actions)
    all_obs_arr = np.concatenate(all_observations)

    report = {
        "rollout_dir": rollout_dir,
        "n_rollouts": len(npz_files),
        "success_reward_threshold": success_reward_threshold,
        "success_rate": float(np.mean(success_flags)),
        "total_reward": {
            "mean": float(np.mean(total_rewards)),
            "std": float(np.std(total_rewards)),
            "min": float(np.min(total_rewards)),
            "max": float(np.max(total_rewards)),
        },
        "length": {
            "mean": float(np.mean(lengths)),
            "std": float(np.std(lengths)),
            "min": int(np.min(lengths)),
            "max": int(np.max(lengths)),
        },
        "step_reward": {
            "mean": float(np.mean(all_rewards_arr)),
            "std": float(np.std(all_rewards_arr)),
            "min": float(np.min(all_rewards_arr)),
            "max": float(np.max(all_rewards_arr)),
            "percentiles": {
                "p25": float(np.percentile(all_rewards_arr, 25)),
                "p50": float(np.percentile(all_rewards_arr, 50)),
                "p75": float(np.percentile(all_rewards_arr, 75)),
                "p95": float(np.percentile(all_rewards_arr, 95)),
                "p99": float(np.percentile(all_rewards_arr, 99)),
            },
            "zero_ratio": float(np.mean(all_rewards_arr == 0.0)),
            "positive_ratio": float(np.mean(all_rewards_arr > 0.0)),
        },
        "action": {
            "mean": np.mean(all_actions_arr, axis=0).tolist(),
            "std": np.std(all_actions_arr, axis=0).tolist(),
            "min": np.min(all_actions_arr, axis=0).tolist(),
            "max": np.max(all_actions_arr, axis=0).tolist(),
        },
        "observation": {
            "mean": np.mean(all_obs_arr, axis=0).tolist(),
            "std": np.std(all_obs_arr, axis=0).tolist(),
            "min": np.min(all_obs_arr, axis=0).tolist(),
            "max": np.max(all_obs_arr, axis=0).tolist(),
        },
        "by_outcome": {
            "success": {
                "count": len(success_rewards),
                "mean_total_reward": float(np.mean(success_rewards)) if success_rewards else 0.0,
                "std_total_reward": float(np.std(success_rewards)) if success_rewards else 0.0,
            },
            "fail": {
                "count": len(fail_rewards),
                "mean_total_reward": float(np.mean(fail_rewards)) if fail_rewards else 0.0,
                "std_total_reward": float(np.std(fail_rewards)) if fail_rewards else 0.0,
            },
        },
    }

    # 奖励分量分解（仅当 npz 含分量键时才计算并加入报告）
    if has_components:
        breakdown = {}
        for name in REWARD_COMPONENTS:
            vals = component_totals[name]
            if len(vals) == 0:
                continue
            breakdown[name] = {
                "mean_total": float(np.mean(vals)),
                "std_total": float(np.std(vals)),
            }
        report["reward_breakdown"] = breakdown

    # 打印报告
    print("\n" + "=" * 60)
    print("Rollout 轨迹统计分析")
    print("=" * 60)
    print(f"分析目录：{rollout_dir}")
    print(f"轨迹数量：{report['n_rollouts']}")
    print(f"成功阈值（总奖励 ≥ {success_reward_threshold}）：")
    print(f"  成功率：{report['success_rate']*100:.1f}%")
    print(f"  成功轨迹数：{report['by_outcome']['success']['count']}")
    print(f"  失败轨迹数：{report['by_outcome']['fail']['count']}")
    print("总奖励统计：")
    print(f"  均值 ± 标准差：{report['total_reward']['mean']:.2f} ± {report['total_reward']['std']:.2f}")
    print(f"  最小值：{report['total_reward']['min']:.2f}")
    print(f"  最大值：{report['total_reward']['max']:.2f}")
    print(f"成功轨迹平均奖励：{report['by_outcome']['success']['mean_total_reward']:.2f} ± "
          f"{report['by_outcome']['success']['std_total_reward']:.2f}")
    print(f"失败轨迹平均奖励：{report['by_outcome']['fail']['mean_total_reward']:.2f} ± "
          f"{report['by_outcome']['fail']['std_total_reward']:.2f}")
    print("轨迹长度统计：")
    print(f"  均值 ± 标准差：{report['length']['mean']:.1f} ± {report['length']['std']:.1f}")
    print(f"  范围：[{report['length']['min']}, {report['length']['max']}]")
    print("单步奖励分布：")
    print(f"  均值 ± 标准差：{report['step_reward']['mean']:.4f} ± {report['step_reward']['std']:.4f}")
    print(f"  范围：[{report['step_reward']['min']:.4f}, {report['step_reward']['max']:.4f}]")
    print(f"  分位数 p25/p50/p75/p95/p99："
          f"{report['step_reward']['percentiles']['p25']:.4f}/"
          f"{report['step_reward']['percentiles']['p50']:.4f}/"
          f"{report['step_reward']['percentiles']['p75']:.4f}/"
          f"{report['step_reward']['percentiles']['p95']:.4f}/"
          f"{report['step_reward']['percentiles']['p99']:.4f}")
    print(f"  零奖励比例：{report['step_reward']['zero_ratio']*100:.1f}%")
    print(f"  正奖励比例：{report['step_reward']['positive_ratio']*100:.1f}%")
    print("动作统计（各维度）：")
    print(f"  均值：{report['action']['mean']}")
    print(f"  标准差：{report['action']['std']}")
    print(f"  最小值：{report['action']['min']}")
    print(f"  最大值：{report['action']['max']}")

    # 奖励分量分解（核心诊断：判断奖励来源是否合理）
    if has_components:
        print("奖励分量分解（每条轨迹累计值的均值 ± 标准差）：")
        orig_mean = report["reward_breakdown"].get("original_reward", {}).get("mean_total", 0.0)
        for name in REWARD_COMPONENTS:
            if name not in report["reward_breakdown"]:
                continue
            m = report["reward_breakdown"][name]["mean_total"]
            s = report["reward_breakdown"][name]["std_total"]
            flag = ""
            # 自动标注可疑模式，帮助快速定位 reward hacking
            if name == "reward_lift" and m > 1.0 and orig_mean < 1e-6:
                flag = "  ← ⚠ 白送分？cube 未抬起却拿抬升奖励（基线写错）"
            if name == "reward_z_float" and m < -5.0:
                flag = "  ← ⚠ 末端悬浮严重"
            if name == "reward_reach" and abs(m) < 0.5 and orig_mean < 1e-6:
                flag = "  ← 接近信号过弱"
            if name == "reward_grasp" and m < 1e-6:
                flag = "  ← 全程未抓取"
            print(f"  {name:20s}: {m:+8.3f} ± {s:.3f}{flag}")
    else:
        print("奖励分量分解：[跳过] 该批轨迹 npz 未保存 reward 分量"
              "（旧轨迹或未启用 SO101LiftRewardShapingWrapper）")
    print("=" * 60)

    if save_report:
        report_path = os.path.join(rollout_dir, "rollout_analysis_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n统计报告已保存：{report_path}")

    return report
