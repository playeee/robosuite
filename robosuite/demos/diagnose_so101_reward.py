"""
SO101 策略奖励分解与行为诊断脚本。

这是回答“策略为什么拿到这个奖励 / 为什么不工作”的核心工具。它加载训练好的
模型，跑若干个 episode，逐步记录：
  1. 各奖励分量（reward_reach / reward_grasp / reward_lift / 各惩罚项）
  2. 末端(eef)与 cube 的位置/高度/距离轨迹
  3. 任务成功与否、是否接触/抓取/抬起

然后打印一份“奖励分解 + 行为轨迹”的对照报告，并把结果保存为 JSON + PNG，
便于在改奖励函数/环境前后做对比。

典型用途：
  - 训练完/评估时跑一次，确认奖励来源是否合理（有没有 reward hacking）
  - 改了奖励函数后跑一次，对比“悬停”行为是否被纠正
  - 排查“机械臂抬起来就不下来”“悬浮在桌面上方”等问题

运行方式：
    conda run -n robosuite python robosuite/demos/diagnose_so101_reward.py

可选参数（环境变量）：
    MODEL_PATH=xxx.zip VEC_NORMALIZE_PATH=xxx.pkl N_EPISODES=5 \
        conda run -n robosuite python robosuite/demos/diagnose_so101_reward.py

输出：
    ./logs/sac_lift_so101_realistic/diagnose/reward_breakdown_report.json
    ./logs/sac_lift_so101_realistic/diagnose/reward_breakdown_ep*.png
"""

import os

import numpy as np

import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.utils.placement_samplers import UniformRandomSampler
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from so101_realistic import (
    REWARD_COMPONENTS,
    SO101LiftObservationWrapper,
    SO101LiftRewardShapingWrapper,
)


# =============================================================================
# 配置（与训练保持一致）
# =============================================================================
TRAIN_MODE = "realistic_state"
MODEL_PATH = os.environ.get("MODEL_PATH", "lift_so101_sac_realistic.zip")
VEC_NORMALIZE_PATH = os.environ.get("VEC_NORMALIZE_PATH", "vec_normalize_so101.pkl")
N_EPISODES = int(os.environ.get("N_EPISODES", "3"))
MAX_STEPS = 200
SAVE_DIR = "./logs/sac_lift_so101_realistic/diagnose"


# =============================================================================
# 环境创建（与 eval_so101_visual.py 一致，但关闭渲染以提速）
# =============================================================================
def make_env():
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
        "Lift", robots="SO101",
        has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
        control_freq=20, horizon=MAX_STEPS,
        reward_shaping=(TRAIN_MODE == "easy"), reward_scale=1.0,
        use_object_obs=(TRAIN_MODE == "easy"),
        initialization_noise=None,
        table_friction=(1.0, 5e-3, 1e-4), table_full_size=(0.8, 0.8, 0.05),
        placement_initializer=placement_initializer,
    )
    env = GymWrapper(env)
    if TRAIN_MODE != "easy":
        env = SO101LiftObservationWrapper(env)
        env = SO101LiftRewardShapingWrapper(env, mode=TRAIN_MODE)
    return Monitor(env)


def _get_base(env):
    """从 VecNormalize -> DummyVecEnv -> Monitor -> ... -> Lift 解包到底层。"""
    inner = env.venv.envs[0] if hasattr(env, "venv") else env
    while hasattr(inner, "env"):
        inner = inner.env
    return inner


# =============================================================================
# 单 episode 诊断
# =============================================================================
def diagnose_episode(model, env, base_env, eef_site_id, table_z, ep_idx):
    """跑一个 episode，记录奖励分量与位置轨迹，返回诊断结果 dict。"""
    obs = env.reset()
    # 初始状态
    eef0 = np.array(base_env.sim.data.site_xpos[eef_site_id])
    cube0 = np.array(base_env.sim.data.body_xpos[base_env.cube_body_id])

    comp_traj = {name: [] for name in REWARD_COMPONENTS}
    eef_z_traj, cube_z_traj, dist_traj = [], [], []
    grasp_steps = 0

    done = [False]
    step = 0
    while not done[0] and step < MAX_STEPS:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = env.step(action)
        i = info[0] if info and len(info) > 0 else {}

        eef = np.array(base_env.sim.data.site_xpos[eef_site_id])
        cube = np.array(base_env.sim.data.body_xpos[base_env.cube_body_id])
        eef_z_traj.append(eef[2])
        cube_z_traj.append(cube[2])
        dist_traj.append(float(np.linalg.norm(eef - cube)))

        for name in REWARD_COMPONENTS:
            comp_traj[name].append(float(i.get(name, 0.0)))
        if bool(i.get("reward_grasp", 0.0)) > 0:
            grasp_steps += 1
        step += 1

    # 统计
    totals = {name: float(np.sum(comp_traj[name])) for name in REWARD_COMPONENTS}
    eef_z = np.array(eef_z_traj)
    cube_z = np.array(cube_z_traj)
    dist = np.array(dist_traj)
    cube_lift_above_rest = float(max(0.0, cube_z.max() - cube0[2]))

    result = {
        "episode": ep_idx,
        "steps": step,
        "total_shaped_reward": totals["shaped_reward"],
        "original_reward_total": totals["original_reward"],
        "component_totals": totals,
        "init": {
            "eef_pos": eef0.tolist(),
            "cube_pos": cube0.tolist(),
            "table_z": float(table_z),
            "eef_above_table": float(eef0[2] - table_z),
            "cube_above_table": float(cube0[2] - table_z),
            "init_3d_dist": float(np.linalg.norm(eef0 - cube0)),
            "init_eef_above_cube": float(eef0[2] - cube0[2]),
        },
        "trajectory_stats": {
            "eef_z": {"start": float(eef_z[0]), "end": float(eef_z[-1]),
                      "mean": float(eef_z.mean()), "min": float(eef_z.min()),
                      "max": float(eef_z.max())},
            "cube_z": {"start": float(cube_z[0]), "end": float(cube_z[-1]),
                       "mean": float(cube_z.mean()), "min": float(cube_z.min()),
                       "max": float(cube_z.max())},
            "dist": {"start": float(dist[0]), "end": float(dist[-1]),
                     "mean": float(dist.mean()), "min": float(dist.min()),
                     "max": float(dist.max())},
            "mean_eef_above_cube": float((eef_z - cube_z).mean()),
            "mean_eef_above_table": float((eef_z - table_z).mean()),
            "steps_within_5cm": int((dist < 0.05).sum()),
            "steps_within_2cm": int((dist < 0.02).sum()),
            "grasp_steps": grasp_steps,
            "cube_lift_above_rest": cube_lift_above_rest,
        },
        "raw": {
            "eef_z": eef_z.tolist(),
            "cube_z": cube_z.tolist(),
            "dist": dist.tolist(),
            # 存全部分量，便于 plot_diagnosis 按需取用
            **{name: comp_traj[name] for name in REWARD_COMPONENTS},
        },
    }
    return result


# =============================================================================
# 打印报告
# =============================================================================
def print_report(result):
    ep = result["episode"]
    init = result["init"]
    ts = result["trajectory_stats"]
    tot = result["component_totals"]

    print("\n" + "=" * 72)
    print(f"Episode {ep} | 步数={result['steps']} | "
          f"shaped总奖励={result['total_shaped_reward']:.2f} | "
          f"original总奖励={result['original_reward_total']:.2f}")
    print("=" * 72)
    print("初始状态:")
    print(f"  桌面 z={init['table_z']:.3f}")
    print(f"  eef  z={init['eef_pos'][2]:.3f} (高出桌面 {init['eef_above_table']:.3f} m)")
    print(f"  cube z={init['cube_pos'][2]:.3f} (高出桌面 {init['cube_above_table']:.3f} m)")
    print(f"  初始 eef-cube 3D距离={init['init_3d_dist']:.3f} m, "
          f"eef比cube高={init['init_eef_above_cube']:.3f} m")

    print("\n奖励分量累计（整个 episode 求和）:")
    for name in REWARD_COMPONENTS:
        flag = ""
        if name == "reward_lift" and tot[name] > 1.0 and result["original_reward_total"] < 1e-6:
            flag = "  ← ⚠ 白送分？cube没被抬起却拿到抬升奖励（基线写错）"
        if name == "reward_z_float" and tot[name] < -5.0:
            flag = "  ← ⚠ 末端悬浮严重"
        if name == "reward_reach" and abs(tot[name]) < 0.5:
            flag = "  ← 接近信号过弱，eef没有靠近cube"
        if name == "reward_grasp" and tot[name] < 1e-6:
            flag = "  ← 全程未抓取"
        print(f"  {name:20s}: {tot[name]:+9.3f}{flag}")

    print("\n位置/距离轨迹:")
    print(f"  eef  z: 初={ts['eef_z']['start']:.3f} 末={ts['eef_z']['end']:.3f} "
          f"均={ts['eef_z']['mean']:.3f} 最低={ts['eef_z']['min']:.3f} 最高={ts['eef_z']['max']:.3f}")
    print(f"  cube z: 初={ts['cube_z']['start']:.3f} 末={ts['cube_z']['end']:.3f} "
          f"均={ts['cube_z']['mean']:.3f} 最低={ts['cube_z']['min']:.3f} 最高={ts['cube_z']['max']:.3f}")
    print(f"  3D距离: 初={ts['dist']['start']:.3f} 末={ts['dist']['end']:.3f} "
          f"均={ts['dist']['mean']:.3f} 最小={ts['dist']['min']:.3f}")
    print(f"  eef高出cube均值={ts['mean_eef_above_cube']:.3f} m | "
          f"eef高出桌面均值={ts['mean_eef_above_table']:.3f} m")
    print(f"  距离<5cm步数={ts['steps_within_5cm']}/{result['steps']} | "
          f"<2cm步数={ts['steps_within_2cm']}/{result['steps']} | "
          f"抓取步数={ts['grasp_steps']}/{result['steps']}")
    print(f"  cube 相对静止位置最大抬升={ts['cube_lift_above_rest']:.4f} m "
          f"(>0.04 才算成功抬起)")


# =============================================================================
# 绘图：奖励分解 + 位置轨迹
# =============================================================================
def plot_diagnosis(result, save_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [提示] 未安装 matplotlib，跳过绘图")
        return None

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"reward_breakdown_ep{result['episode']:02d}.png")
    raw = result["raw"]
    steps = np.arange(len(raw["eef_z"]))

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    fig.suptitle(
        f"Episode {result['episode']} 诊断 | shaped={result['total_shaped_reward']:.2f} "
        f"original={result['original_reward_total']:.2f} 步数={result['steps']}",
        fontsize=13,
    )

    # 1) 奖励分量堆叠面积图（正分量与负分量分开看）
    ax = axes[0]
    pos_names = ["reward_reach", "reward_grasp", "reward_lift", "reward_gripper_move"]
    neg_names = ["reward_smooth", "reward_vel", "reward_ee_vel", "reward_z_float"]
    pos_stack = np.array([raw[n] for n in pos_names]).T  # (T, 4)
    neg_stack = np.array([raw[n] for n in neg_names]).T
    colors_pos = ["#4C9F70", "#2E86AB", "#F6AE2D", "#9D8DF1"]
    colors_neg = ["#888", "#AAA", "#CCC", "#E07A5F"]
    ax.stackplot(steps, pos_stack.T, labels=pos_names, colors=colors_pos, alpha=0.85)
    ax.stackplot(steps, neg_stack.T, labels=neg_names, colors=colors_neg, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("单步各奖励分量")
    ax.set_title("奖励分解（堆叠面积：上=正信号，下=惩罚）")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # 2) eef 与 cube 高度
    ax = axes[1]
    ax.plot(steps, raw["eef_z"], color="#F6AE2D", label="eef z", linewidth=1.4)
    ax.plot(steps, raw["cube_z"], color="#E07A5F", label="cube z", linewidth=1.4)
    ax.axhline(result["init"]["table_z"], color="gray", linestyle="--",
               label=f"桌面 z={result['init']['table_z']:.2f}")
    ax.axhline(result["init"]["table_z"] + 0.04, color="green", linestyle=":",
               label="成功阈值 +4cm")
    ax.set_ylabel("高度 (m)")
    ax.set_title("末端与 cube 高度轨迹")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3) eef-cube 3D 距离 + 抓取标记
    ax = axes[2]
    ax.plot(steps, raw["dist"], color="#2E86AB", label="eef-cube 3D 距离", linewidth=1.4)
    ax.axhline(0.05, color="orange", linestyle="--", label="5cm（接近）")
    ax.axhline(0.02, color="red", linestyle=":", label="2cm（抓取范围）")
    ax.set_ylabel("距离 (m)")
    ax.set_xlabel("step")
    ax.set_title("末端到 cube 的 3D 距离")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close(fig)
    print(f"  分析图已保存：{save_path}")
    return save_path


# =============================================================================
# 主流程
# =============================================================================
def main():
    print("=" * 72)
    print("SO101 奖励分解与行为诊断")
    print("=" * 72)
    print(f"模型: {MODEL_PATH}")
    print(f"归一化参数: {VEC_NORMALIZE_PATH}")
    print(f"episode 数: {N_EPISODES}")
    print("=" * 72)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"未找到模型：{MODEL_PATH}")
    if not os.path.exists(VEC_NORMALIZE_PATH):
        raise FileNotFoundError(f"未找到归一化参数：{VEC_NORMALIZE_PATH}")

    env = DummyVecEnv([make_env])
    env = VecNormalize.load(VEC_NORMALIZE_PATH, env)
    env.training = False
    env.norm_reward = False
    model = SAC.load(MODEL_PATH, env=env)

    base_env = _get_base(env)
    eef_site_id = base_env.robots[0].eef_site_id["right"]
    table_z = float(base_env.model.mujoco_arena.table_offset[2])

    os.makedirs(SAVE_DIR, exist_ok=True)
    all_results = []
    for ep in range(N_EPISODES):
        r = diagnose_episode(model, env, base_env, eef_site_id, table_z, ep)
        all_results.append(r)
        print_report(r)
        plot_diagnosis(r, SAVE_DIR)

    # 汇总
    print("\n" + "#" * 72)
    print("多 episode 汇总")
    print("#" * 72)
    avg_shaped = np.mean([r["total_shaped_reward"] for r in all_results])
    avg_orig = np.mean([r["original_reward_total"] for r in all_results])
    print(f"  平均 shaped 总奖励: {avg_shaped:.2f}")
    print(f"  平均 original 总奖励: {avg_orig:.2f}  (>0 表示有 episode 完成任务)")
    print(f"  各分量平均累计:")
    for name in REWARD_COMPONENTS:
        vals = [r["component_totals"][name] for r in all_results]
        print(f"    {name:20s}: {np.mean(vals):+8.3f}")
    avg_close = np.mean([r["trajectory_stats"]["steps_within_5cm"] for r in all_results])
    avg_grasp = np.mean([r["trajectory_stats"]["grasp_steps"] for r in all_results])
    print(f"  平均进入5cm步数: {avg_close:.1f}/{MAX_STEPS}")
    print(f"  平均抓取步数: {avg_grasp:.1f}/{MAX_STEPS}")

    # 保存 JSON 报告
    report_path = os.path.join(SAVE_DIR, "reward_breakdown_report.json")
    import json
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"episodes": all_results, "summary": {
            "avg_shaped": float(avg_shaped),
            "avg_original": float(avg_orig),
            "avg_steps_within_5cm": float(avg_close),
            "avg_grasp_steps": float(avg_grasp),
        }}, f, indent=2, ensure_ascii=False)
    print(f"\n  JSON 报告已保存：{report_path}")

    env.close()


if __name__ == "__main__":
    main()
