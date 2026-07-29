"""
SO101 控制器零动作漂移诊断脚本。

诊断项：
  1. 发送零动作 50 步，记录 qpos 漂移和 EEF 漂移
  2. 测试方向控制：发送 ±1 动作逐关节，记录 EEF 变化
  3. 检查驱动器类型（sim.model.actuator_biastype）
  4. 检查控制器 goal_qpos、重力补偿力矩

运行方式：
    conda run -n robosuite python debug_controller.py
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
    prefix = robot.robot_model.naming_prefix

    # =========================================================================
    # 1. 零动作漂移测试
    # =========================================================================
    print("=" * 70)
    print("1. 零动作漂移测试（50 步）")
    print("=" * 70)

    # 记录初始状态
    init_qpos = sim.data.qpos.copy()
    eef_site_id = sim.model.site_name2id(prefix + "grip_site")
    init_eef = sim.data.site_xpos[eef_site_id].copy()

    zero_action = np.zeros(env.action_dim)
    print(f"  action_dim: {env.action_dim}")
    print(f"  初始 qpos (arm): {init_qpos[:5]}")
    print(f"  初始 EEF: {init_eef}")

    qpos_history = [init_qpos[:5].copy()]
    eef_history = [init_eef.copy()]

    for step in range(50):
        env.step(zero_action)
        qpos_history.append(sim.data.qpos[:5].copy())
        eef_history.append(sim.data.site_xpos[eef_site_id].copy())

    qpos_history = np.array(qpos_history)
    eef_history = np.array(eef_history)

    # 漂移统计
    qpos_drift = qpos_history[-1] - qpos_history[0]
    eef_drift = eef_history[-1] - eef_history[0]
    max_qpos_drift = np.max(np.abs(qpos_history - qpos_history[0]), axis=0)
    max_eef_drift = np.max(np.abs(eef_history - eef_history[0]), axis=0)

    print(f"\n  最终 qpos 漂移: {qpos_drift}")
    print(f"  最终 EEF 漂移: {eef_drift} (范数: {np.linalg.norm(eef_drift):.6f})")
    print(f"  最大 qpos 漂移: {max_qpos_drift}")
    print(f"  最大 EEF 漂移: {max_eef_drift}")

    # 判定
    if np.linalg.norm(eef_drift) > 0.01:
        print("  ⚠ EEF 漂移 > 1cm，控制器零动作存在明显漂移！")
    else:
        print("  ✓ EEF 漂移 < 1cm，零动作漂移在可接受范围内。")

    # =========================================================================
    # 2. 方向控制测试
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. 方向控制测试（±1 逐关节，20 步）")
    print("=" * 70)

    n_arm_joints = 5  # SO101 有 5 个 arm 关节

    for joint_idx in range(n_arm_joints):
        for sign in [1, -1]:
            env.reset()
            init_eef = sim.data.site_xpos[eef_site_id].copy()

            action = np.zeros(env.action_dim)
            action[joint_idx] = sign * 1.0

            for _ in range(20):
                env.step(action)

            final_eef = sim.data.site_xpos[eef_site_id].copy()
            delta_eef = final_eef - init_eef
            print(f"  joint[{joint_idx}] action={sign:+d}: ΔEEF = {delta_eef}, "
                  f"‖Δ‖ = {np.linalg.norm(delta_eef):.5f}")

    # =========================================================================
    # 3. 驱动器类型检查
    # =========================================================================
    print("\n" + "=" * 70)
    print("3. 驱动器类型检查")
    print("=" * 70)

    n_actuators = sim.model.nu
    print(f"  驱动器数量: {n_actuators}")
    for i in range(n_actuators):
        name = sim.model.actuator_names[i] if hasattr(sim.model, 'actuator_names') else f"actuator_{i}"
        biastype = sim.model.actuator_biastype[i]
        gainprm = sim.model.actuator_gainprm[i]
        biasprm = sim.model.actuator_biasprm[i]
        ctrlrange = sim.model.actuator_ctrlrange[i]
        print(f"  [{i}] name={name}, biastype={biastype}, "
              f"gainprm={gainprm}, biasprm={biasprm}, ctrlrange={ctrlrange}")

    # =========================================================================
    # 4. 控制器内部状态
    # =========================================================================
    print("\n" + "=" * 70)
    print("4. 控制器内部状态")
    print("=" * 70)

    # 获取控制器引用
    part_controllers = robot.part_controllers
    for name, ctrl in part_controllers.items():
        print(f"\n  控制器: {name} (type: {type(ctrl).__name__})")
        if hasattr(ctrl, 'goal_qpos'):
            print(f"    goal_qpos: {ctrl.goal_qpos}")
        if hasattr(ctrl, 'goal_vel'):
            print(f"    goal_vel: {ctrl.goal_vel}")
        if hasattr(ctrl, 'goal_torque'):
            print(f"    goal_torque: {ctrl.goal_torque}")
        if hasattr(ctrl, 'kp'):
            print(f"    kp: {ctrl.kp}")
        if hasattr(ctrl, 'damping_ratio'):
            print(f"    damping_ratio: {ctrl.damping_ratio}")
        if hasattr(ctrl, 'output_max'):
            print(f"    output_max: {ctrl.output_max}")
        if hasattr(ctrl, 'output_min'):
            print(f"    output_min: {ctrl.output_min}")

    # 重力补偿 / qfrc_bias
    print(f"\n  重力偏置力矩 (qfrc_bias): {sim.data.qfrc_bias[:n_arm_joints]}")

    # 当前关节状态
    print(f"\n  当前 qpos: {sim.data.qpos[:n_arm_joints]}")
    print(f"  当前 qvel: {sim.data.qvel[:n_arm_joints]}")
    print(f"  当前 ctrl: {sim.data.ctrl[:n_actuators]}")

    print("\n完成。")


if __name__ == "__main__":
    main()
