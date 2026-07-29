"""
SO101 夹爪抓取检测（_check_grasp）诊断脚本。

逐项验证 SO101Gripper 的 grasp 判定逻辑：
  1. 打印所有含 "pad"/"jaw"/"cube"/"gripper" 的 geom 名称
  2. 打印 SO101Gripper 的 important_geoms（含 naming_prefix）
  3. 打印 cube 的 contact_geoms
  4. 手动闭合夹爪，检查接触力和 _check_grasp
  5. 将 cube 移分移动到夹爪内，再检查 _check_grasp

运行方式：
    conda run -n robosuite python debug_grasp.py
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

    # =========================================================================
    # 1. 打印所有含关键字的 geom 名称
    # =========================================================================
    print("=" * 70)
    print("1. 关键 geom 名称")
    print("=" * 70)
    keywords = ["pad", "jaw", "cube", "gripper"]
    for kw in keywords:
        matched = [sim.model.geom_id2name(i) for i in range(sim.model.ngeom)
                   if kw in sim.model.geom_id2name(i).lower()]
        print(f"  含 '{kw}' 的 geom: {matched}")

    # =========================================================================
    # 2. 打印 SO101Gripper 的 important_geoms（含 naming_prefix）
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. SO101Gripper.important_geoms（含 naming_prefix）")
    print("=" * 70)
    prefix = gripper.naming_prefix
    print(f"  naming_prefix: '{prefix}'")
    for key, geoms in gripper.important_geoms.items():
        print(f"  {key}: {geoms}")

    # =========================================================================
    # 3. 打印 cube 的 contact_geoms
    # =========================================================================
    print("\n" + "=" * 70)
    print("3. cube.contact_geoms")
    print("=" * 70)
    cube = env.cube
    print(f"  cube.root_body: {cube.root_body}")
    print(f"  cube.contact_geoms: {cube.contact_geoms}")

    # =========================================================================
    # 4. 手动闭合夹爪，检查接触和 _check_grasp
    # =========================================================================
    print("\n" + "=" * 70)
    print("4. 手动闭合夹爪 → 检查接触和 _check_grasp")
    print("=" * 70)

    # 先读取 EEF 和 cube 位置
    eef_site_id = sim.model.site_name2id(prefix + "grip_site")
    eef_pos = sim.data.site_xpos[eef_site_id].copy()
    cube_body_id = sim.model.body_name2id(cube.root_body)
    cube_pos = sim.data.body_xpos[cube_body_id].copy()
    print(f"  初始 EEF 位置: {eef_pos}")
    print(f"  初始 cube 位置: {cube_pos}")
    print(f"  EEF-cube 距离: {np.linalg.norm(eef_pos - cube_pos):.4f}")

    # 闭合夹爪动作 (gripper action = 1 表示闭合)
    close_action = np.zeros(env.action_dim)
    close_action[-1] = 1.0  # 最后一维是夹爪

    print("\n  闭合夹爪 50 步...")
    for i in range(50):
        env.step(close_action)

    # 检查接触
    print("\n  --- MuJoCo 接触信息 ---")
    for c_id in range(sim.data.ncon):
        c = sim.data.contact[c_id]
        g1 = sim.model.geom_id2name(c.geom1)
        g2 = sim.model.geom_id2name(c.geom2)
        # 只打印与夹爪/cube 相关的接触
        if any(kw in g1.lower() or kw in g2.lower() for kw in ["pad", "jaw", "cube"]):
            print(f"    接触 {c_id}: {g1} <-> {g2}, dist={c.dist:.6f}")

    # _check_grasp
    grasp_result = env._check_grasp(gripper=gripper, object_geoms=cube)
    print(f"\n  _check_grasp 结果: {grasp_result}")

    # =========================================================================
    # 5. 移动 cube 到夹爪内，再检查 _check_grasp
    # =========================================================================
    print("\n" + "=" * 70)
    print("5. 将 cube 移分移动到夹爪内 → 检查 _check_grasp")
    print("=" * 70)

    # 重新 reset
    env.reset()

    # 获取夹爪 pad 的世界位置，用来估计夹爪中心
    fixed_pad_id = sim.model.geom_name2id(prefix + "fixed_jaw_pad")
    moving_pad_id = sim.model.geom_name2id(prefix + "moving_jaw_pad")
    fixed_pad_pos = sim.data.geom_xpos[fixed_pad_id].copy()
    moving_pad_pos = sim.data.geom_xpos[moving_pad_id].copy()
    gripper_center = (fixed_pad_pos + moving_pad_pos) / 2.0
    print(f"  fixed_jaw_pad 位置: {fixed_pad_pos}")
    print(f"  moving_jaw_pad 位置: {moving_pad_pos}")
    print(f"  夹爪中心估计: {gripper_center}")

    # 将 cube body 直接移动到夹爪中心（通过 qpos0 修改）
    cube_jnt_adrs = sim.model.body_jntadr[cube_body_id]
    cube_jnt_qposadr = sim.model.jnt_qposadr[cube_jnt_adrs]
    print(f"  移动前 cube 位置: {sim.data.qpos[cube_jnt_qposadr:cube_jnt_qposadr+3]}")
    sim.data.qpos[cube_jnt_qposadr:cube_jnt_qposadr + 3] = gripper_center
    sim.forward()  # 重新计算衍生量
    print(f"  移动后 cube 位置: {sim.data.body_xpos[cube_body_id]}")

    # 闭合夹爪
    print("\n  闭合夹爪 100 步...")
    for i in range(100):
        env.step(close_action)

    # 检查接触
    print("\n  --- MuJoCo 接触信息 ---")
    for c_id in range(sim.data.ncon):
        c = sim.data.contact[c_id]
        g1 = sim.model.geom_id2name(c.geom1)
        g2 = sim.model.geom_id2name(c.geom2)
        if any(kw in g1.lower() or kw in g2.lower() for kw in ["pad", "jaw", "cube"]):
            print(f"    接触 {c_id}: {g1} <-> {g2}, dist={c.dist:.6f}")

    grasp_result = env._check_grasp(gripper=gripper, object_geoms=cube)
    print(f"\n  _check_grasp 结果: {grasp_result}")

    # 额外：逐步检查 fingerpad 组的接触
    print("\n  --- fingerpad 接触分解 ---")
    left_pad_geoms = gripper.important_geoms["left_fingerpad"]
    right_pad_geoms = gripper.important_geoms["right_fingerpad"]
    cube_geoms = cube.contact_geoms
    print(f"  left_fingerpad geoms: {left_pad_geoms}")
    print(f"  right_fingerpad geoms: {right_pad_geoms}")
    print(f"  cube contact_geoms: {cube_geoms}")
    left_contact = env.check_contact(left_pad_geoms, cube_geoms)
    right_contact = env.check_contact(right_pad_geoms, cube_geoms)
    print(f"  left_fingerpad 接触 cube: {left_contact}")
    print(f"  right_fingerpad 接触 cube: {right_contact}")
    print(f"  两者同时接触 → _check_grasp = {left_contact and right_contact}")

    print("\n完成。")


if __name__ == "__main__":
    main()
