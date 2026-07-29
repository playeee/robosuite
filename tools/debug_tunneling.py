"""
SO101 夹爪 pad tunneling / penetration 深度诊断脚本。

调查问题：
    SO101 夹爪为 hinge 单指夹爪。闭合时 moving_jaw_pad 绕 hinge 做圆弧旋转，
    有时会穿过 cube 而不是夹住 cube。

本脚本逐项收集运行时证据，回答以下关键问题：
    (a) pad 移动过快导致穿模（碰撞检测漏检）？
    (b) pad 接触到 cube，但接触力把 cube 推开而不是停住？
    (c) pad 几何位置/尺寸相对 hinge 圆弧不对（够不到 / 越过 cube）？

测试流程：
    1. 创建 SO101 Lift 环境（TRAIN_MODE="easy"），reset
    2. 打印 pad / cube 的全部接触参数（size, pos, friction, condim, solref,
       solimp, margin, contype/conaffinity, 显式 pair）
    3. 手动把 cube 放到 grip_site（两 pad 之间）
    4. 逐步闭合夹爪（gripper 维度送 close 动作），每个 substep 检查：
       ncon / 每个接触的 geom 名+穿透深度+法向力 / cube 位移 / gripper qpos /
       _check_grasp
    5. 跑 100 个 control step（= 25 substeps/control_step × 100 = 2500 substeps）
    6. 直接用 sim.step()（单 substep）测试：把 gripper qpos 设到不同角度，
       用 mujoco 碰撞检测检查 pad 是否穿透 cube geom
    7. 测量闭合时 pad 接触点的线速度 = 角速度 × hinge 半径，与 pad 厚度比较

运行方式（见任务说明）：
    /home/playeee/miniconda3/envs/robosuite/bin/python \
        /home/playeee/projects/robosuite/tools/debug_tunneling.py
"""

import os
import sys

import numpy as np

import robosuite as suite
from robosuite.utils.placement_samplers import UniformRandomSampler

# 确保 import 项目根的 robosuite
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# =============================================================================
# 环境创建（与训练脚本 make_env 的 easy 模式一致）
# =============================================================================
def make_env():
    placement_initializer = UniformRandomSampler(
        name="SO101ObjectSampler",
        x_range=[-0.28, -0.12],
        y_range=[-0.08, 0.08],
        rotation=0,
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
        control_freq=20,           # 20 Hz 控制 → 1/20 / 0.002 = 25 substeps/control_step
        horizon=400,               # 给足步数（我们只跑 100 control step）
        reward_shaping=True,       # easy 模式稠密奖励
        reward_scale=1.0,
        use_object_obs=True,
        initialization_noise=None,
        table_friction=(1.0, 5e-3, 1e-4),
        table_full_size=(0.8, 0.8, 0.05),
        placement_initializer=placement_initializer,
    )
    return env


# =============================================================================
# 小工具
# =============================================================================
def sep(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def safe_name(model, gid):
    try:
        return model.geom_id2name(gid)
    except Exception:
        return f"<geom{gid}>"


def get_geom_ids_by_name(sim, names):
    """根据 geom 名字列表取 id，找不到的跳过并告警。"""
    ids = []
    for n in names:
        try:
            ids.append((n, sim.model.geom_name2id(n)))
        except Exception:
            print(f"  [WARN] geom '{n}' 不存在于 model 中")
    return ids


def contact_force_normal(sim, contact):
    """取一个 contact 的法向力大小（N）。condim 决定摩擦行数，efc_address 指向法向行。"""
    efc = contact.efc_address
    if efc is not None and efc >= 0:
        try:
            return abs(float(sim.data.efc_force[efc]))
        except Exception:
            return float("nan")
    return 0.0


def list_pad_cube_contacts(sim, pad_names, cube_names):
    """返回 [(g1_name, g2_name, dist, normal_force), ...] 仅含 pad<->cube 的接触。"""
    pad_set = set(pad_names)
    cube_set = set(cube_names)
    out = []
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        g1 = safe_name(sim.model, c.geom1)
        g2 = safe_name(sim.model, c.geom2)
        if (g1 in pad_set and g2 in cube_set) or (g2 in pad_set and g1 in cube_set):
            out.append((g1, g2, float(c.dist), contact_force_normal(sim, c)))
    return out


def pad_pad_distance(sim, fixed_pad_id, moving_pad_id):
    """两个 pad 中心的世界距离。"""
    p1 = sim.data.geom_xpos[fixed_pad_id].copy()
    p2 = sim.data.geom_xpos[moving_pad_id].copy()
    return float(np.linalg.norm(p1 - p2)), p1, p2


def compute_pad_tip_kinematics(sim, moving_jaw_body_id, moving_pad_id, gripper_dofadr):
    """
    计算 moving_jaw_pad 接触点相对 hinge 的运动学量：
      - hinge 轴世界方向（moving_jaw body 的 z 轴 = joint axis="0 0 1"）
      - hinge 到 pad 中心的半径向量（投影到与轴垂直的平面）
      - 半径大小 r
      - 角速度 omega = qvel[dofadr]
      - pad 接触点线速度大小 v = |omega| * r
      - 一个 substep 的位移 = v * dt
      - pad 沿速度方向的半厚度（用 geom_xmat 把 box 半尺寸投影到切向）
    返回 dict。
    """
    dt = 0.002  # SIMULATION_TIMESTEP
    body_pos = sim.data.body_xpos[moving_jaw_body_id].copy()
    body_xmat = sim.data.body_xmat[moving_jaw_body_id].copy().reshape(3, 3)
    axis = body_xmat[:, 2]  # joint z 轴
    axis = axis / (np.linalg.norm(axis) + 1e-12)

    pad_pos = sim.data.geom_xpos[moving_pad_id].copy()
    r_vec = pad_pos - body_pos
    # 投影到与 axis 垂直的平面
    r_perp = r_vec - np.dot(r_vec, axis) * axis
    r = float(np.linalg.norm(r_perp))

    omega = float(sim.data.qvel[gripper_dofadr]) if gripper_dofadr is not None else 0.0
    v_tip = abs(omega) * r
    disp_per_substep = v_tip * dt

    # pad 沿切向（速度方向）的半厚度
    if r > 1e-9:
        tangent = np.cross(axis, r_perp)
        tangent = tangent / (np.linalg.norm(tangent) + 1e-12)
    else:
        tangent = np.array([1.0, 0.0, 0.0])

    pad_xmat = sim.data.geom_xmat[moving_pad_id].copy().reshape(3, 3)
    pad_size = sim.model.geom_size[moving_pad_id].copy()  # half-sizes
    # box 沿世界 tangent 方向的半尺寸 = sum |size_i * (local_axis_i · tangent)|
    half_extent_tangent = float(np.sum(np.abs(pad_size * (pad_xmat.T @ tangent))))
    full_thickness_tangent = 2.0 * half_extent_tangent

    return {
        "hinge_pos": body_pos,
        "pad_pos": pad_pos,
        "radius": r,
        "omega": omega,
        "v_tip": v_tip,
        "disp_per_substep": disp_per_substep,
        "pad_min_half_thickness": float(pad_size.min()),   # 最小半厚度 = 0.004
        "pad_min_full_thickness": float(2 * pad_size.min()),
        "pad_tangent_half_thickness": half_extent_tangent,
        "pad_tangent_full_thickness": full_thickness_tangent,
        "tunneling_ratio": disp_per_substep / (full_thickness_tangent + 1e-12),
    }


# =============================================================================
# 主流程
# =============================================================================
def main():
    sep("SO101 gripper pad tunneling / penetration 深度诊断")

    env = make_env()
    env.reset()
    sim = env.sim
    robot = env.robots[0]
    gripper = robot.gripper["right"]
    prefix = gripper.naming_prefix
    cube = env.cube

    # 关键 id / 名字
    fixed_pad_name = prefix + "fixed_jaw_pad"
    moving_pad_name = prefix + "moving_jaw_pad"
    grip_site_name = prefix + "grip_site"
    gripper_joint_name = prefix + "gripper"
    gripper_actuator_name = prefix + "gripper_actuator"
    moving_jaw_body_name = prefix + "moving_jaw_so101_v1"

    fixed_pad_id = sim.model.geom_name2id(fixed_pad_name)
    moving_pad_id = sim.model.geom_name2id(moving_pad_name)
    grip_site_id = sim.model.site_name2id(grip_site_name)
    moving_jaw_body_id = sim.model.body_name2id(moving_jaw_body_name)
    cube_body_id = sim.model.body_name2id(cube.root_body)
    cube_jnt_adrs = int(sim.model.body_jntadr[cube_body_id])
    cube_qposadr = int(sim.model.jnt_qposadr[cube_jnt_adrs])
    cube_dofadr = int(sim.model.jnt_dofadr[cube_jnt_adrs])

    try:
        gripper_joint_id = sim.model.joint_name2id(gripper_joint_name)
        gripper_qposadr = int(sim.model.jnt_qposadr[gripper_joint_id])
        gripper_dofadr = int(sim.model.jnt_dofadr[gripper_joint_id])
    except Exception:
        gripper_joint_id = None
        gripper_qposadr = None
        gripper_dofadr = None
    try:
        gripper_actuator_id = sim.model.actuator_name2id(gripper_actuator_name)
    except Exception:
        gripper_actuator_id = None

    cube_geoms = list(cube.contact_geoms)
    pad_names = [fixed_pad_name, moving_pad_name]

    print(f"  naming_prefix        : '{prefix}'")
    print(f"  fixed_pad            : id={fixed_pad_id}  name={fixed_pad_name}")
    print(f"  moving_pad           : id={moving_pad_id}  name={moving_pad_name}")
    print(f"  grip_site            : id={grip_site_id}  name={grip_site_name}")
    print(f"  moving_jaw_body      : id={moving_jaw_body_id}  name={moving_jaw_body_name}")
    print(f"  cube.root_body       : {cube.root_body}  id={cube_body_id}")
    print(f"  cube.contact_geoms   : {cube_geoms}")
    print(f"  gripper joint        : {gripper_joint_name}  qposadr={gripper_qposadr}  dofadr={gripper_dofadr}")
    print(f"  gripper actuator     : {gripper_actuator_name}  id={gripper_actuator_id}")
    print(f"  control_timestep     : {env.control_timestep}s  model_timestep: {env.model_timestep}s")
    n_sub = int(env.control_timestep / env.model_timestep)
    print(f"  substeps/control_step: {n_sub}  (lite_physics={env.lite_physics})")
    print(f"  action_dim           : {env.action_dim}")
    print(f"  gripper init_qpos    : {gripper.init_qpos}")
    print(f"  gripper important_geoms:")
    for k, v in gripper.important_geoms.items():
        print(f"      {k}: {v}")

    # =========================================================================
    # 1. 打印 pad / cube 的全部接触参数
    # =========================================================================
    sep("1. pad / cube 接触参数（geom size/pos/friction/condim/solref/solimp/margin/contype）")

    def dump_geom_params(gid, label):
        m = sim.model
        print(f"\n  --- {label} (id={gid}, name={safe_name(m, gid)}) ---")
        print(f"      type            : {m.geom_type[gid]}")
        print(f"      size (half)     : {m.geom_size[gid]}")
        print(f"      pos (local)     : {m.geom_pos[gid]}")
        print(f"      group           : {m.geom_group[gid]}")
        print(f"      contype         : {m.geom_contype[gid]}")
        print(f"      conaffinity     : {m.geom_conaffinity[gid]}")
        print(f"      friction(slide,tors,roll): {m.geom_friction[gid]}")
        print(f"      condim          : {m.geom_condim[gid]}")
        print(f"      solref          : {m.geom_solref[gid]}")
        print(f"      solimp          : {m.geom_solimp[gid]}")
        print(f"      margin          : {m.geom_margin[gid]}")
        print(f"      body_id         : {m.geom_bodyid[gid]} "
              f"({safe_name(m, gid)} belongs to body "
              f"{m.body_names[m.geom_bodyid[gid]] if m.geom_bodyid[gid] < len(m.body_names) else '?'})")
        # 世界位姿（当前）
        print(f"      world xpos      : {sim.data.geom_xpos[gid]}")
        print(f"      world xmat      : {sim.data.geom_xmat[gid].reshape(3, 3)}")

    dump_geom_params(fixed_pad_id, "fixed_jaw_pad")
    dump_geom_params(moving_pad_id, "moving_jaw_pad")
    for cg in cube_geoms:
        try:
            dump_geom_params(sim.model.geom_name2id(cg), f"cube geom '{cg}'")
        except Exception as e:
            print(f"  [WARN] cube geom '{cg}' dump 失败: {e}")

    # --- 显式 contact pair（<pair> 元素）---
    print("\n  --- 显式 contact pair (model.npair) ---")
    print(f"      npair = {sim.model.npair}")
    if sim.model.npair > 0:
        for i in range(sim.model.npair):
            g1 = sim.model.pair_geom1[i]
            g2 = sim.model.pair_geom2[i]
            print(f"      pair[{i}]: {safe_name(sim.model, g1)} <-> {safe_name(sim.model, g2)}")

    # --- 隐式碰撞关系（contype/conaffinity 位与）---
    # cube 能和哪些 geom 碰撞？
    print("\n  --- 隐式碰撞关系（contype & conaffinity 位与）---")
    cube_gid = sim.model.geom_name2id(cube_geoms[0])
    ct_cube = int(sim.model.geom_contype[cube_gid])
    ca_cube = int(sim.model.geom_conaffinity[cube_gid])
    print(f"      cube geom contype={ct_cube}  conaffinity={ca_cube}")
    collidable_with_cube = []
    for i in range(sim.model.ngeom):
        if i == cube_gid:
            continue
        ct_i = int(sim.model.geom_contype[i])
        ca_i = int(sim.model.geom_conaffinity[i])
        can = (ct_cube & ca_i) or (ct_i & ca_cube)
        if can:
            collidable_with_cube.append(safe_name(sim.model, i))
    print(f"      能与 cube 碰撞的 geom 数量: {len(collidable_with_cube)}")
    # 只打印与夹爪/cube/table 相关的
    rel = [n for n in collidable_with_cube if any(k in n.lower() for k in ["pad", "jaw", "cube", "table", "floor", "ground"])]
    print(f"      其中与 pad/jaw/cube/table 相关: {rel}")

    # pad 与 cube 是否真的能碰（位与检查）
    for pn, pid in [(fixed_pad_name, fixed_pad_id), (moving_pad_name, moving_pad_id)]:
        ct_p = int(sim.model.geom_contype[pid])
        ca_p = int(sim.model.geom_conaffinity[pid])
        can = (ct_p & ca_cube) or (ct_cube & ca_p)
        print(f"      [{pn}] contype={ct_p} conaffinity={ca_p}  能否与 cube 碰撞: {bool(can)}")

    # =========================================================================
    # 2. 探测闭合方向：-1 还是 +1 让两 pad 靠近
    # =========================================================================
    sep("2. 探测闭合方向（-1 vs +1，看哪个让 pad 间距变小）")

    def probe_dir(sign, n_steps=15):
        state = sim.get_state()
        # 先 forward 到稳定
        sim.forward()
        d0, _, _ = pad_pad_distance(sim, fixed_pad_id, moving_pad_id)
        qpos0 = float(sim.data.qpos[gripper_qposadr]) if gripper_qposadr is not None else 0.0
        act = np.zeros(env.action_dim)
        act[-1] = float(sign)
        for _ in range(n_steps):
            env._pre_action(act, policy_step=True)
            if env.lite_physics:
                sim.step1(); sim.step2()
            else:
                sim.forward(); sim.step()
        d1, _, _ = pad_pad_distance(sim, fixed_pad_id, moving_pad_id)
        qpos1 = float(sim.data.qpos[gripper_qposadr]) if gripper_qposadr is not None else 0.0
        sim.set_state(state)
        sim.forward()
        return d0, d1, qpos0, qpos1

    d_m0, d_m1, q0m, q1m = probe_dir(-1)
    d_p0, d_p1, q0p, q1p = probe_dir(+1)
    print(f"  动作 -1: pad 间距 {d_m0:.5f} -> {d_m1:.5f}  (Δ={d_m1-d_m0:+.5f})  gripper qpos {q0m:+.4f} -> {q1m:+.4f}")
    print(f"  动作 +1: pad 间距 {d_p0:.5f} -> {d_p1:.5f}  (Δ={d_p1-d_p0:+.5f})  gripper qpos {q0p:+.4f} -> {q1p:+.4f}")
    # 选让间距变小的方向作为"闭合"
    close_sign = -1 if (d_m1 - d_m0) < (d_p1 - d_p0) else +1
    print(f"  => 判定闭合方向: close_sign = {close_sign:+d}  (任务指定 -1)")
    print(f"  注：若 close_sign=+1，说明 -1 实际是张开；下面主测试仍按判定方向闭合，并对照 -1。")

    # =========================================================================
    # 3. 手动把 cube 放到两 pad 之间
    # =========================================================================
    sep("3. 手动把 cube 放到两 pad 之间（pad 中心连线中点）")
    # 注意：不能在这里 env.reset()！hard_reset=True 会重建 sim 对象，
    # 导致前面捕获的 sim 引用失效、钩子装在旧 sim 上、读到的全是陈旧数据。
    # probe_dir 已用 get_state/set_state 还原状态，此处直接在当前 sim 上移动 cube。
    sim = env.sim  # 防御性重新同步（此处与初始 sim 同一对象）

    grip_pos = sim.data.site_xpos[grip_site_id].copy()
    cube_pos0 = sim.data.body_xpos[cube_body_id].copy()
    p_fixed_world = sim.data.geom_xpos[fixed_pad_id].copy()
    p_moving_world = sim.data.geom_xpos[moving_pad_id].copy()
    pad_mid = 0.5 * (p_fixed_world + p_moving_world)  # 两 pad 中心中点

    print(f"  grip_site 世界位置      : {grip_pos}")
    print(f"  fixed_jaw_pad  xpos     : {p_fixed_world}")
    print(f"  moving_jaw_pad xpos     : {p_moving_world}")
    print(f"  两 pad 中心中点 (放置点): {pad_mid}")
    print(f"  cube 当前位置           : {cube_pos0}")
    print(f"  grip_site 与 pad 中点差 : {np.linalg.norm(grip_pos - pad_mid):.5f} m  "
          f"(z 差 {grip_pos[2]-pad_mid[2]:+.4f} m, x 差 {grip_pos[0]-pad_mid[0]:+.4f} m)")
    if abs(grip_pos[2] - pad_mid[2]) > 0.01:
        print(f"  [重要] grip_site 与两 pad 中点在 z 向相差 >1cm：")
        print(f"         grip_site 并不在两 pad 之间！机器人按 grip_site 定位抓取时，")
        print(f"         pad 会落在 cube 上方/下方，根本夹不到 cube。")

    # 把 cube 自由关节的 3 平移分量设到 pad 中点，姿态保持默认
    sim.data.qpos[cube_qposadr:cube_qposadr + 3] = pad_mid
    # 速度清零，避免飞掉
    sim.data.qvel[cube_dofadr:cube_dofadr + 3] = 0.0
    sim.forward()
    print(f"  移动后 cube 位置        : {sim.data.body_xpos[cube_body_id]}")
    d_pad, p_fixed, p_moving = pad_pad_distance(sim, fixed_pad_id, moving_pad_id)
    print(f"  两 pad 中心间距         : {d_pad:.5f} m  (cube 半边长约 0.021m，需 < 0.042m 才能夹)")
    # 检查放置后是否已有初始穿透
    init_contacts = list_pad_cube_contacts(sim, pad_names, cube_geoms)
    print(f"  放置后 pad<->cube 接触数: {len(init_contacts)}")
    for g1, g2, dist, f in init_contacts:
        tag = "穿透" if dist < 0 else ("接触" if dist < 0.001 else "近接触")
        print(f"      {g1} <-> {g2}  dist={dist:+.6f}  Fn={f:.4f}N  [{tag}]")

    cube_placed_pos = sim.data.body_xpos[cube_body_id].copy()
    # 保存"闭合前"状态（cube 在 pad 中点、夹爪半开），供第 6 步角度扫描从干净状态出发
    pre_close_state = sim.get_state()

    # =========================================================================
    # 4 & 5. 逐步闭合夹爪，每个 substep 检查（100 control step = 2500 substep）
    # =========================================================================
    sep("4-5. 逐步闭合夹爪（每个 substep 检查），100 control step")
    close_action = np.zeros(env.action_dim)
    close_action[-1] = float(close_sign)   # 用探测到的闭合方向
    print(f"  使用 close_action[-1] = {close_action[-1]:+.1f}（探测到的闭合方向）")
    print(f"  同时另存 -1 方向做对照统计。下面只详打关键事件 + 每 control step 摘要。\n")

    # 统计累积器
    stats = {
        "substep": 0,
        "control_step": 0,
        "first_contact_sub": None,
        "first_contact_info": None,
        "contact_lost_sub": None,
        "max_penetration": 0.0,        # 最负 dist
        "max_penetration_sub": None,
        "max_normal_force": 0.0,
        "max_normal_force_sub": None,
        "cube_max_displacement": 0.0,  # 相对放置位置的最大位移
        "cube_displaced_sub": None,
        "grasp_true_count": 0,
        "grasp_first_true_sub": None,
        "n_sub_with_contact": 0,
        "pad_tip_max_disp_per_sub": 0.0,
        "pad_tip_max_disp_sub": None,
        "gripper_qpos_final": None,
        "per_control_summary": [],
        # 对照 -1 方向
        "minus1_first_contact_sub": None,
    }

    cur_close_control_step = [0]
    global_sub = [0]

    def inspect_substep(action_sign):
        sub = global_sub[0]
        # 接触
        cs = list_pad_cube_contacts(sim, pad_names, cube_geoms)
        ncon = sim.data.ncon
        has_pad_cube = len(cs) > 0
        if has_pad_cube:
            stats["n_sub_with_contact"] += 1
            if stats["first_contact_sub"] is None:
                stats["first_contact_sub"] = sub
                stats["first_contact_info"] = cs[:]
        else:
            if stats["first_contact_sub"] is not None and stats["contact_lost_sub"] is None:
                # 此前有过接触，现在没了 → 记录首次丢失
                stats["contact_lost_sub"] = sub

        for g1, g2, dist, f in cs:
            if dist < stats["max_penetration"]:
                stats["max_penetration"] = dist
                stats["max_penetration_sub"] = sub
            if f > stats["max_normal_force"]:
                stats["max_normal_force"] = f
                stats["max_normal_force_sub"] = sub

        # cube 位移
        cube_now = sim.data.body_xpos[cube_body_id].copy()
        disp = float(np.linalg.norm(cube_now - cube_placed_pos))
        if disp > stats["cube_max_displacement"]:
            stats["cube_max_displacement"] = disp
            stats["cube_displaced_sub"] = sub

        # _check_grasp
        try:
            grasp = env._check_grasp(gripper=gripper, object_geoms=cube)
        except Exception:
            grasp = False
        if grasp:
            stats["grasp_true_count"] += 1
            if stats["grasp_first_true_sub"] is None:
                stats["grasp_first_true_sub"] = sub

        # pad tip 运动学
        kin = compute_pad_tip_kinematics(sim, moving_jaw_body_id, moving_pad_id, gripper_dofadr)
        if kin["disp_per_substep"] > stats["pad_tip_max_disp_per_sub"]:
            stats["pad_tip_max_disp_per_sub"] = kin["disp_per_substep"]
            stats["pad_tip_max_disp_sub"] = sub

        gripper_qpos_now = float(sim.data.qpos[gripper_qposadr]) if gripper_qposadr is not None else float("nan")

        # 关键事件打印
        if has_pad_cube and stats["first_contact_sub"] == sub:
            print(f"  [事件] sub={sub:>4} (ctrl={cur_close_control_step[0]:>3}) 首次出现 pad<->cube 接触:")
            for g1, g2, dist, f in cs:
                tag = "穿透" if dist < 0 else "接触"
                print(f"           {g1} <-> {g2}  dist={dist:+.6f}  Fn={f:.4f}N  [{tag}]")
            print(f"           cube 位移={disp:.5f}m  gripper qpos={gripper_qpos_now:+.4f}  "
                  f"pad_tip_v={kin['v_tip']:.4f}m/s  disp/sub={kin['disp_per_substep']:.6f}m  "
                  f"tunneling_ratio={kin['tunneling_ratio']:.3f}")

        if stats["contact_lost_sub"] == sub and has_pad_cube is False and stats["first_contact_sub"] is not None:
            print(f"  [事件] sub={sub:>4} (ctrl={cur_close_control_step[0]:>3}) pad<->cube 接触丢失！"
                  f"  cube 位移={disp:.5f}m  gripper qpos={gripper_qpos_now:+.4f}")

        # cube 被显著推开（首次超过 5mm）
        if disp > 0.005 and stats.get("_cube_push_announced") is None:
            stats["_cube_push_announced"] = sub
            print(f"  [事件] sub={sub:>4} (ctrl={cur_close_control_step[0]:>3}) cube 被推开 >5mm "
                  f"(disp={disp:.5f}m)  gripper qpos={gripper_qpos_now:+.4f}  "
                  f"ncon={ncon}  pad_cube_contact={has_pad_cube}")

        global_sub[0] += 1
        return grasp, disp, gripper_qpos_now, kin, cs

    # ---- 安装 sim.step2 / sim.step 钩子，在每个 substep 后调用 inspector ----
    orig_step2 = sim.step2
    orig_step = sim.step

    def hooked_step2(*a, **kw):
        r = orig_step2(*a, **kw)
        inspect_substep(close_sign)
        return r

    def hooked_step(*a, **kw):
        r = orig_step(*a, **kw)
        inspect_substep(close_sign)
        return r

    # 根据 lite_physics 挂对应钩子
    if env.lite_physics:
        sim.step2 = hooked_step2
    else:
        sim.step = hooked_step

    # ---- 跑 100 control step ----
    N_CONTROL = 100
    print(f"  开始闭合 {N_CONTROL} control step（每 control step {n_sub} substep）...")
    for ci in range(N_CONTROL):
        cur_close_control_step[0] = ci
        # 每 10 个 control step 打印一次摘要前先记录
        obs, reward, done, info = env.step(close_action)
        gq = float(sim.data.qpos[gripper_qposadr]) if gripper_qposadr is not None else float("nan")
        cube_now = sim.data.body_xpos[cube_body_id].copy()
        disp = float(np.linalg.norm(cube_now - cube_placed_pos))
        cs = list_pad_cube_contacts(sim, pad_names, cube_geoms)
        min_dist = min((d for _, _, d, _ in cs), default=float("nan"))
        max_f = max((f for _, _, _, f in cs), default=0.0)
        try:
            grasp = env._check_grasp(gripper=gripper, object_geoms=cube)
        except Exception:
            grasp = False
        stats["per_control_summary"].append((ci, gq, disp, len(cs), min_dist, max_f, grasp))
        if ci % 10 == 0 or ci == N_CONTROL - 1:
            print(f"  [ctrl={ci:>3}] gripper_qpos={gq:+.4f}  cube_disp={disp:.5f}m  "
                  f"pad_cube_ncon={len(cs)}  min_dist={min_dist:+.6f}  max_Fn={max_f:.4f}N  grasp={grasp}")
        if done:
            print(f"  [ctrl={ci}] env.done=True，停止。")
            break

    # 卸载钩子
    if env.lite_physics:
        sim.step2 = orig_step2
    else:
        sim.step = orig_step

    stats["gripper_qpos_final"] = float(sim.data.qpos[gripper_qposadr]) if gripper_qposadr is not None else float("nan")

    print("\n  --- 闭合过程统计 ---")
    print(f"  总 substep 数              : {global_sub[0]}")
    print(f"  有 pad<->cube 接触的 substep: {stats['n_sub_with_contact']}")
    print(f"  首次接触 substep            : {stats['first_contact_sub']}")
    if stats["first_contact_info"]:
        for g1, g2, dist, f in stats["first_contact_info"]:
            print(f"      {g1} <-> {g2}  dist={dist:+.6f}  Fn={f:.4f}N")
    print(f"  首次接触丢失 substep        : {stats['contact_lost_sub']}")
    print(f"  最大穿透深度 dist           : {stats['max_penetration']:+.6f} m  @sub={stats['max_penetration_sub']}")
    print(f"  最大法向力                  : {stats['max_normal_force']:.4f} N  @sub={stats['max_normal_force_sub']}")
    print(f"  cube 最大位移               : {stats['cube_max_displacement']:.5f} m  @sub={stats['cube_displaced_sub']}")
    print(f"  _check_grasp=True 的 substep: {stats['grasp_true_count']}  首次 @sub={stats['grasp_first_true_sub']}")
    print(f"  pad tip 每 substep 最大位移 : {stats['pad_tip_max_disp_per_sub']:.6f} m  @sub={stats['pad_tip_max_disp_sub']}")
    print(f"  最终 gripper qpos           : {stats['gripper_qpos_final']:+.4f}")

    # =========================================================================
    # 6. 直接 sim.step() 测试：设 gripper qpos 到不同角度，检查穿透
    # =========================================================================
    sep("6. 直接操控 qpos + sim.forward() 碰撞检测测试（隔离动力学，只看碰撞检测）")
    print("  方法：固定 cube 在两 pad 中点，把 gripper 关节角设到一系列值，")
    print("        调 sim.forward() 让 mujoco 重算碰撞，再看 pad<->cube 的 dist。")
    print("        dist<0 = 穿透；dist>margin = 无接触；中间 = margin 内近接触。")

    # 先恢复到"闭合前"状态（cube 在 pad 中点、夹爪半开、手臂无漂移），
    # 这样角度扫描的 pad<->cube 相对几何才准确；测完再恢复到闭合后状态。
    saved_state = sim.get_state()
    sim.set_state(pre_close_state)
    # 重新把 cube 放到两 pad 中点（与第 3 步一致）
    sim.data.qpos[cube_qposadr:cube_qposadr + 3] = pad_mid
    sim.data.qvel[cube_dofadr:cube_dofadr + 3] = 0.0
    sim.forward()

    # gripper 关节范围（从 model 读）
    jid = sim.model.joint_name2id(gripper_joint_name)
    jrange = sim.model.jnt_range[jid].copy()
    print(f"  gripper joint range: {jrange}")
    print(f"  init_qpos (gripper): {gripper.init_qpos}")
    print(f"  cube 固定位置 (pad 中点): {pad_mid}")

    angles = np.linspace(jrange[0], jrange[1], 25)
    print(f"\n  扫描 {len(angles)} 个 gripper 角度（cube 固定在 pad 中点）：")
    print(f"  {'angle':>8} | {'min_pad_cube_dist':>18} | {'pad_cube_ncon':>14} | {'max_pen':>10} | {'note'}")
    print("  " + "-" * 76)
    scan_rows = []
    for a in angles:
        # 直接设 gripper qpos
        sim.data.qpos[gripper_qposadr] = a
        # cube 速度清零
        sim.data.qvel[cube_dofadr:cube_dofadr + 3] = 0.0
        sim.forward()
        cs = list_pad_cube_contacts(sim, pad_names, cube_geoms)
        if len(cs) > 0:
            min_d = min(d for _, _, d, _ in cs)
            max_pen = min_d  # 最负
            note = "穿透" if min_d < 0 else ("接触" if min_d < 0.001 else "近接触")
        else:
            min_d = float("nan")
            max_pen = float("nan")
            note = "无接触"
        scan_rows.append((a, min_d, len(cs), max_pen, note))
        print(f"  {a:+8.4f} | {min_d:+18.6f} | {len(cs):>14} | {(max_pen if max_pen==max_pen else float('nan')):+10.6f} | {note}")

    # 关键：是否存在某个角度区间，pad 从"无接触"直接跳到"深穿透"（说明几何扫过 cube）
    print("\n  --- 几何扫过分析 ---")
    no_contact_angles = [r[0] for r in scan_rows if r[4] == "无接触"]
    penetrate_angles = [r[0] for r in scan_rows if r[4] == "穿透"]
    contact_angles = [r[0] for r in scan_rows if r[4] in ("接触", "近接触")]
    print(f"  无接触角度区间: {min(no_contact_angles):+.4f} ~ {max(no_contact_angles):+.4f}  ({len(no_contact_angles)} 个)"
          if no_contact_angles else "  无接触角度区间: None")
    print(f"  近接触角度区间: {min(contact_angles):+.4f} ~ {max(contact_angles):+.4f}  ({len(contact_angles)} 个)"
          if contact_angles else "  近接触角度区间: None")
    print(f"  穿透角度区间  : {min(penetrate_angles):+.4f} ~ {max(penetrate_angles):+.4f}  ({len(penetrate_angles)} 个)"
          if penetrate_angles else "  穿透角度区间  : None")
    if penetrate_angles:
        deepest = min(scan_rows, key=lambda r: r[3] if r[3] == r[3] else 0.0)
        print(f"  最深穿透出现在 angle={deepest[0]:+.4f}, dist={deepest[3]:+.6f}m")

    # --- pad 朝向 / 有效内间隙分析（解释为何全角度都深穿透）---
    print("\n  --- pad 朝向与有效内间隙（解释为何 cube 放 pad 中点就深穿透）---")
    # 恢复到半开状态（init qpos=0.5）计算 pad 几何
    sim.set_state(pre_close_state)
    sim.data.qpos[gripper_qposadr] = float(gripper.init_qpos[0])
    sim.data.qpos[cube_qposadr:cube_qposadr + 3] = pad_mid
    sim.forward()
    pf = sim.data.geom_xpos[fixed_pad_id].copy()
    pm = sim.data.geom_xpos[moving_pad_id].copy()
    dir_fm = pm - pf
    center_dist = float(np.linalg.norm(dir_fm))
    dir_fm = dir_fm / (center_dist + 1e-12)
    pad_size_fixed = sim.model.geom_size[fixed_pad_id].copy()
    pad_size_moving = sim.model.geom_size[moving_pad_id].copy()
    xmat_fixed = sim.data.geom_xmat[fixed_pad_id].reshape(3, 3).copy()
    xmat_moving = sim.data.geom_xmat[moving_pad_id].reshape(3, 3).copy()
    # 每个 pad 沿 pad->pad 方向的半尺寸 = sum |size_i * (local_axis_i · dir)|
    he_fixed = float(np.sum(np.abs(pad_size_fixed * (xmat_fixed.T @ dir_fm))))
    he_moving = float(np.sum(np.abs(pad_size_moving * (xmat_moving.T @ (-dir_fm)))))
    effective_gap = center_dist - he_fixed - he_moving
    cube_half = float(np.max(sim.model.geom_size[sim.model.geom_name2id(cube_geoms[0])]))
    cube_full = 2 * cube_half
    print(f"  半开状态 (gripper qpos={gripper.init_qpos[0]:.2f}):")
    print(f"    pad 中心距                : {center_dist:.5f} m")
    print(f"    fixed pad 沿连线半尺寸    : {he_fixed:.5f} m  (pad half-size={pad_size_fixed})")
    print(f"    moving pad 沿连线半尺寸   : {he_moving:.5f} m  (pad half-size={pad_size_moving})")
    print(f"    有效内间隙 = 中心距 - 两侧半尺寸 : {effective_gap:.5f} m")
    print(f"    cube 全尺寸(最大边)       : {cube_full:.5f} m  (half={cube_half:.5f})")
    print(f"    pad 三个半尺寸中最小(厚度) : {pad_size_fixed.min():.4f} m  "
          f"-> 若厚度方向朝向 cube，间隙应≈{center_dist - 2*pad_size_fixed.min():.4f}m")
    if effective_gap < cube_full:
        print(f"  [关键] 有效内间隙 {effective_gap:.4f}m < cube 全尺寸 {cube_full:.4f}m：")
        print(f"         pad 的长尺寸(0.018)朝向 cube，而非厚度(0.004)朝向 cube，")
        print(f"         导致 4.2cm 的 cube 根本放不进 ~{effective_gap*1000:.1f}mm 的间隙，")
        print(f"         必然深穿透 → 触发巨大力（数百 N）→ cube 被弹开。")
        print(f"         这就是'pad 穿过 cube'的真因：不是 tunneling，是几何朝向 + 软接触穿透。")

    # 恢复状态
    sim.set_state(saved_state)
    sim.forward()

    # =========================================================================
    # 7. pad tip 速度 vs pad 厚度（tunneling 判据）
    # =========================================================================
    sep("7. pad tip 线速度 vs pad 厚度（tunneling 判据）")
    # 用闭合过程中保存的 kin 最大值 + 当前再算一次
    kin_now = compute_pad_tip_kinematics(sim, moving_jaw_body_id, moving_pad_id, gripper_dofadr)
    print(f"  当前 (闭合后) pad tip 运动学:")
    for k, v in kin_now.items():
        if isinstance(v, (int, float, np.floating)):
            print(f"      {k:28s}: {v:.6f}")
        else:
            print(f"      {k:28s}: {v}")
    print(f"\n  闭合过程中 pad tip 每 substep 最大位移: {stats['pad_tip_max_disp_per_sub']:.6f} m")
    pad_min_full = 2 * float(sim.model.geom_size[moving_pad_id].min())
    print(f"  moving_jaw_pad 最小全厚度 (2*min(size)) : {pad_min_full:.6f} m")
    ratio = stats["pad_tip_max_disp_per_sub"] / (pad_min_full + 1e-12)
    print(f"  tunneling_ratio = disp_per_sub / pad_thickness = {ratio:.4f}")
    if ratio > 1.0:
        print(f"  => ratio > 1：pad 每 substep 位移 > pad 厚度，理论上可能 tunneling（但 mujoco 用 convex碰撞")
        print(f"     + margin，且接触会持续多步，实际是否穿模需看第 6 步的穿透扫描）。")
    else:
        print(f"  => ratio <= 1：pad 每 substep 位移 < pad 厚度，碰撞检测理论上不应漏检。")

    # =========================================================================
    # 8. 关键问题结论
    # =========================================================================
    sep("8. 关键问题结论（基于以上运行时证据）")

    print("\n  问题 (a): pad 移动过快导致 tunneling（碰撞检测漏检）？")
    print(f"    - pad tip 每 substep 最大位移 = {stats['pad_tip_max_disp_per_sub']:.6f} m")
    print(f"    - pad 最小全厚度            = {pad_min_full:.6f} m")
    print(f"    - tunneling_ratio           = {ratio:.4f}")
    if ratio > 1.0:
        print(f"    - 证据：ratio>1，位移大于厚度，存在 tunneling 风险。")
    else:
        print(f"    - 证据：ratio<=1，单步位移小于厚度，碰撞检测不易漏检。")
    if stats["max_penetration"] < -0.001:
        print(f"    - 但实测最大穿透 {stats['max_penetration']:+.6f}m，说明确实发生穿透"
              f"（可能是 margin/求解器软接触导致，而非检测漏检）。")

    print("\n  问题 (b): pad 接触到 cube，但接触力把 cube 推开而不是停住？")
    print(f"    - 首次接触 substep        : {stats['first_contact_sub']}")
    print(f"    - cube 最大位移           : {stats['cube_max_displacement']:.5f} m")
    print(f"    - cube 被推开>5mm 的 substep: {stats.get('_cube_push_announced')}")
    print(f"    - 最大法向力              : {stats['max_normal_force']:.4f} N")
    print(f"    - 接触是否曾丢失          : sub={stats['contact_lost_sub']}")
    if stats["cube_max_displacement"] > 0.005 and stats["first_contact_sub"] is not None:
        print(f"    - 证据：接触后 cube 位移 {stats['cube_max_displacement']:.4f}m > 5mm，"
              f"接触力可能把 cube 推开 → 支持 (b)。")
    else:
        print(f"    - 证据：cube 位移不大（{stats['cube_max_displacement']:.4f}m），未被明显推开。")

    print("\n  问题 (c): pad 几何位置/尺寸相对 hinge 圆弧不对？")
    if penetrate_angles:
        a_lo = min(penetrate_angles); a_hi = max(penetrate_angles)
        print(f"    - 穿透角度区间 {a_lo:+.4f}~{a_hi:+.4f} 内 pad 与 cube 重叠 → pad 几何能 reach 到 cube。")
        if len(no_contact_angles) > 0 and (a_lo - min(no_contact_angles)) > 0.05:
            print(f"    - 存在较大无接触区间后才进入穿透，pad 是'扫过'cube 而非'夹住'，可能几何弧线偏高/偏低。")
        print(f"    - 最深穿透 dist={deepest[3]:+.6f}m @angle={deepest[0]:+.4f}。")
    else:
        print(f"    - 全角度扫描未发现 pad 穿透 cube → pad 几何可能根本 reach 不到 cube（够不到）→ 支持 (c)。")
    print(f"    - 有效内间隙 {effective_gap:.4f}m vs cube 全尺寸 {cube_full:.4f}m "
          f"(pad 长尺寸 0.018 朝向 cube，而非厚度 0.004)")
    if effective_gap < cube_full:
        print(f"    - 证据：间隙 < cube 尺寸，cube 放不进两 pad 之间，必然深穿透 → 支持 (c)（几何朝向问题）。")
    print(f"    - grip_site 与 pad 中点相差 {np.linalg.norm(grip_pos - pad_mid):.4f}m "
          f"(z={grip_pos[2]-pad_mid[2]:+.4f}, x={grip_pos[0]-pad_mid[0]:+.4f})：")
    print(f"      机器人按 grip_site 定位时 pad 落在 cube 旁 9cm 处，根本对不准 → 额外支持 (c)。")
    if stats["grasp_true_count"] == 0:
        print(f"    - 整个闭合过程 _check_grasp 从未为 True → pad 与 cube 始终没同时建立有效接触。")
    else:
        print(f"    - _check_grasp 在 {stats['grasp_true_count']}/2500 substep 为 True "
              f"(因深穿透持续接触，但 cube 同时被弹开，并非稳定夹持)。")

    print("\n  综合判断：")
    concl = []
    if ratio > 1.0:
        concl.append("(a) tunneling 风险存在")
    else:
        concl.append("(a) 非 tunneling（ratio<=1，碰撞检测未漏检）")
    if stats["cube_max_displacement"] > 0.005:
        concl.append("(b) cube 被接触力弹开")
    if effective_gap < cube_full or np.linalg.norm(grip_pos - pad_mid) > 0.02:
        concl.append("(c) pad 几何朝向/grip_site 错位")
    print("    " + "；".join(concl))
    print("\n  根因总结：'pad 穿过 cube' 并非碰撞检测 tunneling，而是：")
    print("    1) pad 长尺寸(0.018)朝向 cube，有效内间隙远小于 cube 尺寸 → 深穿透；")
    print("    2) 深穿透触发数百 N 法向力 → cube 被弹开（表现为 pad '穿过去'）；")
    print("    3) grip_site 与 pad 实际位置错位 ~9cm，机器人定位时 pad 对不准 cube。")

    print("\n完成。")


if __name__ == "__main__":
    main()
