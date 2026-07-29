"""
SO101 Lift 奖励景观分析脚本。

逐步将 EEF 靠近 cube，观察奖励如何随距离变化：
  1. 计算 reaching reward 在不同距离下的值（理论值）
  2. 用 IK 逐步将 EEF 移向 cube，每步打印 EEF pos / cube pos / 距离 / reward / _check_grasp
  3. 在 EEF 接近 cube 时尝试闭合夹爪，检查 grasp 判定

运行方式：
    conda run -n robosuite python debug_reward_landscape.py
"""

import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym

import robosuite as suite
from robosuite.utils.placement_samplers import UniformRandomSampler


# =============================================================================
# 环境创建（easy 模式，与训练脚本 make_env 一致）
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
        "Lift",
        robots="SO101",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=200,
        reward_shaping=True,
        reward_scale=1.0,
        use_object_obs=True,
        initialization_noise=None,
        table_friction=(1.0, 5e-3, 1e-4),
        table_full_size=(0.8, 0.8, 0.05),
        placement_initializer=placement_initializer,
    )
    return env


def main():
    env = make_env()
    env.reset()

    sim = env.sim
    robot = env.robots[0]
    gripper = robot.gripper["right"]
    prefix = robot.robot_model.naming_prefix
    cube = env.cube

    eef_site_id = sim.model.site_name2id(prefix + "grip_site")
    cube_body_id = sim.model.body_name2id(cube.root_body)

    # =========================================================================
    # 1. 理论 reaching reward 曲线
    # =========================================================================
    print("=" * 70)
    print("1. 理论 reaching reward 曲线: r = 1 - tanh(10 * dist)")
    print("=" * 70)
    distances = [0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05, 0.02, 0.01, 0.0]
    for d in distances:
        r = 1.0 - np.tanh(10.0 * d)
        print(f"  dist={d:.3f}  →  reward={r:.4f}")

    # =========================================================================
    # 2. IK 逐步靠近 cube
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. IK 逐步靠近 cube，记录 EEF/cube/distance/reward/grasp")
    print("=" * 70)

    eef_pos = sim.data.site_xpos[eef_site_id].copy()
    cube_pos = sim.data.body_xpos[cube_body_id].copy()
    init_dist = np.linalg.norm(eef_pos - cube_pos)
    print(f"  初始 EEF: {eef_pos}")
    print(f"  初始 cube: {cube_pos}")
    print(f"  初始距离: {init_dist:.4f}")

    # 目标：沿直线将 EEF 移向 cube，分多步完成
    n_steps = 60
    target_pos = cube_pos.copy()
    # 不要完全重叠，稍微偏移让夹爪能包住 cube
    # SO101 夹爪在 EEF site 下方，需要把 EEF 对准 cube 上方
    target_pos[2] += 0.02  # 略微高于 cube

    # 逐步行进
    step_size = (target_pos - eef_pos) / n_steps

    # 先打开夹爪
    open_action = np.zeros(env.action_dim)
    open_action[-1] = -1.0  # 夹爪打开
    for _ in range(30):
        env.step(open_action)

    eef_pos = sim.data.site_xpos[eef_site_id].copy()
    step_size = (target_pos - eef_pos) / n_steps

    print(f"\n  {'step':>4s}  {'EEF_x':>8s} {'EEF_y':>8s} {'EEF_z':>8s}  "
          f"{'cube_x':>8s} {'cube_y':>8s} {'cube_z':>8s}  "
          f"{'dist':>8s} {'reward':>8s} {'grasp':>5s}")
    print("  " + "-" * 90)

    for step in range(n_steps):
        # 计算当前 EEF 到目标的差
        eef_pos = sim.data.site_xpos[eef_site_id].copy()
        cube_pos = sim.data.body_xpos[cube_body_id].copy()
        diff = cube_pos - eef_pos
        dist = np.linalg.norm(diff)

        # 使用简单的动作映射：沿差方向移动
        # 取差的前 5 维映射到 arm action（归一化到 [-1, 1]）
        action = np.zeros(env.action_dim)
        if dist > 0.005:
            # 将差映射到动作空间（大致方向）
            direction = diff / dist
            # 保守步进：每次移动距离的一小部分
            move_scale = min(dist * 2.0, 1.0)
            # 映射到关节动作（简化：使用笛卡尔方向映射）
            # 这里用 JAC 逆或者简单的位置差映射
            # robosuite 的 JOINT_POSITION 控制器直接输出关节位置增量
            # 简化做法：直接给一个朝着 cube 方向的笛卡尔动作
            action[:3] = direction[:3] * move_scale * 0.5  # xy + z
            action[3:5] = 0.0  # 其余关节
        action[-1] = -1.0  # 保持夹爪打开

        obs, reward, done, info = env.step(action)

        # 每隔几步打印
        if step % 5 == 0 or dist < 0.05:
            grasp = env._check_grasp(gripper=gripper, object_geoms=cube)
            print(f"  {step:4d}  {eef_pos[0]:8.4f} {eef_pos[1]:8.4f} {eef_pos[2]:8.4f}  "
                  f"{cube_pos[0]:8.4f} {cube_pos[1]:8.4f} {cube_pos[2]:8.4f}  "
                  f"{dist:8.4f} {reward:8.4f} {str(grasp):>5s}")

        if done:
            break

    # =========================================================================
    # 3. EEF 接近 cube 时闭合夹爪
    # =========================================================================
    print("\n" + "=" * 70)
    print("3. EEF 接近 cube 时闭合夹爪")
    print("=" * 70)

    eef_pos = sim.data.site_xpos[eef_site_id].copy()
    cube_pos = sim.data.body_xpos[cube_body_id].copy()
    dist = np.linalg.norm(eef_pos - cube_pos)
    print(f"  当前 EEF-cube 距离: {dist:.4f}")

    # 闭合夹爪
    close_action = np.zeros(env.action_dim)
    close_action[-1] = 1.0  # 闭合

    for i in range(60):
        env.step(close_action)
        if i % 10 == 0:
            eef_pos = sim.data.site_xpos[eef_site_id].copy()
            cube_pos = sim.data.body_xpos[cube_body_id].copy()
            dist = np.linalg.norm(eef_pos - cube_pos)
            grasp = env._check_grasp(gripper=gripper, object_geoms=cube)
            # 获取 reward
            obs, reward, done, info = env.step(np.zeros(env.action_dim))
            print(f"  step {i:3d}: dist={dist:.4f}, grasp={grasp}, reward={reward:.4f}")
            if done:
                break

    # 最终判定
    final_grasp = env._check_grasp(gripper=gripper, object_geoms=cube)
    cube_height = sim.data.body_xpos[cube_body_id][2]
    table_height = env.model.mujoco_arena.table_offset[2]
    success = cube_height > table_height + 0.04
    print(f"\n  最终 grasp: {final_grasp}")
    print(f"  cube 高度: {cube_height:.4f}, 桌面高度: {table_height:.4f}")
    print(f"  抬起判定 (cube_z > table_z + 0.04): {success}")

    print("\n完成。")


if __name__ == "__main__":
    main()
