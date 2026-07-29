"""
SO101 夹爪间距（jaw gap）vs gripper qpos 测量脚本。

逐步改变夹爪关节角度，测量：
  1. 各 gripper qpos 下的两 jaw 间距
  2. 各关键 geom 的位置
  3. cube 尺寸、摩擦值、contype/conaffinity

用于判断夹爪闭合时能否物理夹住 cube（jaw gap < cube size）。

运行方式：
    conda run -n robosuite python debug_jaw_measure.py
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
    cube = env.cube

    # 关键 geom 名称
    fixed_pad_name = prefix + "fixed_jaw_pad"
    moving_pad_name = prefix + "moving_jaw_pad"
    fixed_jaw_col_name = prefix + "fixed_jaw_collision"
    moving_jaw_col_name = prefix + "moving_jaw_collision"

    fixed_pad_id = sim.model.geom_name2id(fixed_pad_name)
    moving_pad_id = sim.model.geom_name2id(moving_pad_name)

    # gripper 关节索引
    gripper_joint_name = prefix + "gripper"
    gripper_jnt_id = sim.model.joint_name2id(gripper_joint_name)
    gripper_jnt_qposadr = sim.model.jnt_qposadr[gripper_jnt_id]
    gripper_jnt_range = sim.model.jnt_range[gripper_jnt_id]

    print("=" * 70)
    print("SO101 夹爪间距 vs gripper qpos 测量")
    print("=" * 70)
    print(f"  gripper 关节: {gripper_joint_name}")
    print(f"  关节范围: [{gripper_jnt_range[0]:.4f}, {gripper_jnt_range[1]:.4f}] rad")
    print(f"           = [{np.degrees(gripper_jnt_range[0]):.1f}, {np.degrees(gripper_jnt_range[1]):.1f}] deg")

    # =========================================================================
    # 1. 逐步改变 gripper qpos，测量 jaw gap
    # =========================================================================
    print("\n" + "=" * 70)
    print("1. gripper qpos → jaw 间距")
    print("=" * 70)

    # 从最小到最大角度，均匀采样
    qpos_values = np.linspace(
        gripper_jnt_range[0], gripper_jnt_range[1], 20
    )

    print(f"\n  {'qpos(rad)':>10s} {'qpos(deg)':>10s}  "
          f"{'fixed_pad_x':>12s} {'moving_pad_x':>12s}  "
          f"{'gap_x':>10s} {'gap_3d':>10s}")
    print("  " + "-" * 75)

    for qpos_val in qpos_values:
        # 直接设置 gripper 关节角度
        sim.data.qpos[gripper_jnt_qposadr] = qpos_val
        sim.forward()

        fixed_pad_pos = sim.data.geom_xpos[fixed_pad_id].copy()
        moving_pad_pos = sim.data.geom_xpos[moving_pad_id].copy()

        # X 方向间距（夹爪开合方向）
        gap_x = np.abs(fixed_pad_pos[0] - moving_pad_pos[0])
        # 3D 距离
        gap_3d = np.linalg.norm(fixed_pad_pos - moving_pad_pos)

        print(f"  {qpos_val:10.4f} {np.degrees(qpos_val):10.2f}  "
              f"{fixed_pad_pos[0]:12.6f} {moving_pad_pos[0]:12.6f}  "
              f"{gap_x:10.6f} {gap_3d:10.6f}")

    # =========================================================================
    # 2. 关键 geom 在完全闭合时的位置
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. 关键 geom 位置（完全闭合 vs 完全打开）")
    print("=" * 70)

    for qpos_val, label in [(gripper_jnt_range[0], "完全打开(最小角度)"),
                             (gripper_jnt_range[1], "完全闭合(最大角度)")]:
        sim.data.qpos[gripper_jnt_qposadr] = qpos_val
        sim.forward()

        print(f"\n  --- {label}: qpos = {qpos_val:.4f} rad ({np.degrees(qpos_val):.1f} deg) ---")

        for geom_name in [fixed_pad_name, moving_pad_name,
                          fixed_jaw_col_name, moving_jaw_col_name]:
            try:
                geom_id = sim.model.geom_name2id(geom_name)
                pos = sim.data.geom_xpos[geom_id]
                size = sim.model.geom_size[geom_id]
                print(f"    {geom_name}: pos={pos}, size={size}")
            except Exception as e:
                print(f"    {geom_name}: 读取失败 ({e})")

        fixed_pad_pos = sim.data.geom_xpos[fixed_pad_id].copy()
        moving_pad_pos = sim.data.geom_xpos[moving_pad_id].copy()
        gap = np.linalg.norm(fixed_pad_pos - moving_pad_pos)
        print(f"    pad 间距 (3D): {gap:.6f}")

    # =========================================================================
    # 3. cube 尺寸与摩擦
    # =========================================================================
    print("\n" + "=" * 70)
    print("3. cube 尺寸、摩擦值、contype/conaffinity")
    print("=" * 70)

    cube_geoms = cube.contact_geoms
    print(f"  cube.contact_geoms: {cube_geoms}")

    for geom_name in cube_geoms:
        full_name = geom_name if prefix in geom_name else prefix + geom_name
        try:
            geom_id = sim.model.geom_name2id(full_name)
        except Exception:
            # 可能不需要 prefix
            try:
                geom_id = sim.model.geom_name2id(geom_name)
            except Exception:
                print(f"  {geom_name}: 无法找到 geom ID")
                continue

        size = sim.model.geom_size[geom_id]
        friction = sim.model.geom_friction[geom_id]
        contype = sim.model.geom_contype[geom_id]
        conaffinity = sim.model.geom_conaffinity[geom_id]
        pos = sim.data.geom_xpos[geom_id]
        print(f"  {geom_name}:")
        print(f"    size={size}, friction={friction}")
        print(f"    contype={contype}, conaffinity={conaffinity}")
        print(f"    pos={pos}")

    # cube 整体尺寸
    print(f"\n  cube 设计尺寸: size_min={cube.size_min}, size_max={cube.size_max}")

    # 与夹爪间距对比
    print("\n" + "=" * 70)
    print("4. 夹持可行性判断")
    print("=" * 70)

    # 完全闭合时间距
    sim.data.qpos[gripper_jnt_qposadr] = gripper_jnt_range[1]
    sim.forward()
    fixed_pad_pos = sim.data.geom_xpos[fixed_pad_id].copy()
    moving_pad_pos = sim.data.geom_xpos[moving_pad_id].copy()
    min_gap = np.linalg.norm(fixed_pad_pos - moving_pad_pos)

    cube_size = cube.size_min[0] * 2  # cube 边长 = size * 2

    print(f"  完全闭合时 pad 最小间距: {min_gap:.6f} m")
    print(f"  cube 边长: {cube_size:.6f} m (size_min={cube.size_min[0]:.4f})")
    if min_gap < cube_size:
        print(f"  ✓ pad 间距 ({min_gap:.4f}) < cube 边长 ({cube_size:.4f})，闭合时可以物理夹住 cube")
    else:
        print(f"  ⚠ pad 间距 ({min_gap:.4f}) >= cube 边长 ({cube_size:.4f})，闭合时无法几何夹住 cube！")
        print(f"    需要依赖摩擦力夹持（fingerpad 摩擦系数 5.0）")

    print("\n完成。")


if __name__ == "__main__":
    main()
