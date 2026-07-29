# SO101 robosuite 集成：架构、Bug 修复与训练优化全记录

本文档汇总了将 SO-ARM100 (SO101) 机械臂集成到 robosuite 框架进行 RL 训练过程中遇到的全部核心问题、修复方案与设计原理。内容涵盖模型定义、控制器语义、奖励函数设计、训练超参数调优四个层面。

---

## 目录

1. [SO101 模型架构](#1-so101-模型架构)
   - 1.1 [机械臂本体](#11-机械臂本体)
   - 1.2 [夹爪与 fingerpad 设计](#12-夹爪与-fingerpad-设计)
   - 1.3 [Geom 命名与 `_important_geoms`](#13-geom-命名与-_important_geoms)
2. [控制器架构](#2-控制器架构)
   - 2.1 [JOINT_POSITION 控制器原理](#21-joint_position-控制器原理)
   - 2.2 [`<position>` vs `<motor>` 执行器语义冲突](#22-position-vs-motor-执行器语义冲突)
   - 2.3 [GRIP 夹爪控制器](#23-grip-夹爪控制器)
3. [奖励函数设计](#3-奖励函数设计)
   - 3.1 [robosuite 内置 Lift 奖励](#31-robosuite-内置-lift-奖励)
   - 3.2 [`_check_grasp` 判定逻辑](#32-_check_grasp-判定逻辑)
   - 3.3 [自定义 Reward Shaping Wrapper](#33-自定义-reward-shaping-wrapper)
   - 3.4 [`lift_grasping_reward` 为什么看起来不是离散的](#34-lift_grasping_reward-为什么看起来不是离散的)
4. [训练优化](#4-训练优化)
   - 4.1 [SAC 超参数](#41-sac-超参数)
   - 4.2 [探索不足的诊断与修复](#42-探索不足的诊断与修复)
5. [诊断方法论与工具](#5-诊断方法论与工具)
6. [完整诊断案例：从"抓取奖励为零"到"执行器语义冲突"（SAC_18）](#6-完整诊断案例从抓取奖励为零到执行器语义冲突)
   - 6.1 ~ 6.9 分层诊断、根因定位、修复验证、面试问答
7. [完整诊断案例：pad 穿模与物理参数优化（SAC_21 / SAC_22）](#7-完整诊断案例pad-穿模与物理参数优化sac_21--sac_22)
   - 7.1 ~ 7.10 几何—物理—判定三层诊断、多维协同优化、面试问答

---

## 1. SO101 模型架构

### 1.1 机械臂本体

**文件**: `robosuite/models/assets/robots/so101/robot.xml`
**Python**: `robosuite/models/robots/manipulators/so101_robot.py`

SO101 是 SO-ARM100 项目中的 5-DOF 桌面级机械臂，使用 STS3215 舵机驱动。

**关键设计参数**:

| 参数 | 值 | 说明 |
|------|-----|------|
| 自由度 | 5 (arm) + 1 (gripper) | shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll + gripper hinge |
| 初始姿态 (`init_qpos`) | `[0, 0.7, -0.2, -0.2, -0.5]` | "预备抓取"姿态，EEF 落在桌面附近 (z≈0.87) |
| 基座偏移 (`base_xpos_offset`) | `(-0.6, 0, 0.8)` | 桌面级机械臂，直接放在桌面上 (z=0.8) |
| 工作空间半径 | 0.35m | 桌面可达半径 |
| 关节阻尼 | 0.6 | SO101Robot.__init__ 中统一设置 |
| 舵机力矩限制 | ±3.35 Nm | STS3215 物理参数 |

**init_qpos 设计原理**:

全零姿态会让 EEF 悬浮在桌面上方约 0.2m (z≈1.03)，策略难以学会"下降-接近-抓取"。`[0, 0.7, -0.2, -0.2, -0.5]` 使肩部前倾、肘部微屈，让 EEF 落在 z≈0.87，更接近 cube 的 z≈0.82。

### 1.2 夹爪与 fingerpad 设计

**文件**: `robosuite/models/assets/grippers/so101_gripper.xml`
**Python**: `robosuite/models/grippers/so101_gripper.py`

SO101 夹爪是**单指夹爪**（1-DOF hinge），由一个固定颚 (`fixed_jaw`) 和一个活动颚 (`moving_jaw`) 组成。这与 Panda 的平行双指夹爪有本质区别。

#### 原始问题

原始 XML 中夹爪使用 mesh 碰撞体，存在两个严重问题：

1. **Hinge 闭合时间距变大**: SO101 的 hinge 关节正 qpos 使活动颚**远离**固定颚（打开），负 qpos 使其靠近（闭合）。但即使最"闭合"状态 (qpos=-0.1745)，jaw 间距仍为 0.031m，而 cube 最大边长约 0.043m——间距 < cube 边长，理论上可以夹住，但 mesh 碰撞体面积大、法线方向不确定，hinge 旋转会"推"cube 而非夹住。

2. **摩擦不足**: 默认摩擦系数 slide=1.0，不足以通过摩擦力稳定夹持。

#### 修复方案演进：从 box 薄板到 sphere 点接触

fingerpad 设计经历了两个阶段（详见第 7 章诊断案例）：

**阶段 1（box 薄板）**：在两个 jaw 内侧各加一个高摩擦 box 薄板。但 box 长尺寸 (0.018m) 在 hinge 旋转时会意外对齐闭合方向，导致有效内间隙 (19mm) < cube (44mm)，产生深穿透、cube 被弹开（"穿模"现象）。

**阶段 2（sphere 点接触，当前方案）**：改用各向同性的 sphere，闭合方向上的半尺寸恒为 radius，彻底消除方向依赖。

```xml
<!-- 固定颚内侧 pad：高摩擦小球 -->
<geom name="fixed_jaw_pad" type="sphere" group="0"
      size="0.001"
      pos="0.014 -0.0002 0.0195"
      friction="5.0 0.01 0.0001" condim="6"
      rgba="0.2 0.8 0.2 0.6"
      solref="0.02 1.0" solimp="0.99 0.999 0.0001"
      margin="0.008"/>

<!-- 活动颚内侧 pad：高摩擦小球，local x 外移增大间距，y 补偿 hinge 旋转的 z 降低 -->
<geom name="moving_jaw_pad" type="sphere" group="0"
      size="0.001"
      pos="-0.044 0.0043 0.019"
      friction="5.0 0.01 0.0001" condim="6"
      rgba="0.2 0.8 0.2 0.6"
      solref="0.02 1.0" solimp="0.99 0.999 0.0001"
      margin="0.008"/>
```

| 参数 | 值 | 作用 |
|------|-----|------|
| `type` | `sphere` | 各向同性，闭合方向半尺寸恒为 radius，消除方向依赖 |
| `size` | `0.001` | radius=1mm，增大有效内间隙 |
| `friction` | `5.0 0.01 0.0001` | 滑动摩擦 5.0（默认 1.0 的 5 倍） |
| `condim` | `6` | 允许切向摩擦 + 扭转 + 滚动摩擦 |
| `solref` | `0.02 1.0` | 最优接触刚度（太硬弹开 cube，太软穿透反增，详见 7.6.1） |
| `solimp` | `0.99 0.999 0.0001` | 硬化接触，减少穿透 |
| `margin` | `0.008` | 提前 8mm 检测接触，防止一帧内穿透（详见 7.8 Q6） |
| moving_pad `pos` | `(-0.044, 0.0043, 0.019)` | local x 外移增大间距，y 补偿 hinge 旋转导致的 z 降低 |

同时将原始 mesh 碰撞体设为 `contype=0 conaffinity=0`，不参与碰撞检测：

```xml
<geom name="fixed_jaw_collision" type="mesh" contype="0" conaffinity="0" .../>
<geom name="moving_jaw_collision" type="mesh" contype="0" conaffinity="0" .../>
```

**eef body 位置对齐**：moving_pad local 改变后需同步更新 eef body pos，使 `grip_site` 对齐 pad 中点（hinge angle=0.5 半开时），否则机器人按 grip_site 定位时 pad 会系统性偏离 cube。

```xml
<!-- eef body pos = pad 中点 (right_gripper 坐标系) -->
<body name="eef" pos="-0.0032 -0.0002 -0.0106" quat="0.707107 -0 0.707107 -2.37788e-17">
    <site name="grip_site" pos="0 0 0" .../>
</body>
```

**设计原理**：sphere 点接触 + 高摩擦 + condim=6，在不推开 cube 的前提下提供足够夹持力。相比 box 薄板的线接触，sphere 各向同性避免了 hinge 旋转时接触方向意外对齐闭合方向导致的穿模。

**gripper joint 阻尼**：`damping=2.0`（原 0.60），减慢闭合速度减小冲击力，防止高速闭合的动能把 cube 弹开。

> **演进说明**：box→sphere 的变更原因、旋转矩阵分析、cube 投影计算、solref/margin 调参的完整诊断过程见 [第 7 章](#7-完整诊断案例pad-穿模与物理参数优化sac_21--sac_22)。

### 1.3 Geom 命名与 `_important_geoms`

**文件**: `robosuite/models/grippers/so101_gripper.py`

`_check_grasp` 通过 `gripper.important_geoms["left_fingerpad"]` 和 `["right_fingerpad"]` 获取夹爪的 geom 名称列表，然后检查这些 geom 是否与 cube 的 geom 发生接触。

#### Bug 历史

1. **geom 名字不存在**：原始 `_important_geoms` 填写的 geom 名称（如 `moving_jaw_so101_v1_collision`）在 XML 中不存在，导致 `check_contact` 永远匹配不上，`_check_grasp` 永远返回 False。

2. **两组设为同一列表导致误判**：中期修复时将 `left_fingerpad` 和 `right_fingerpad` 都设为 `["moving_jaw_pad", "fixed_jaw_pad"]`（两组相同）。`_check_grasp` 要求两组各自至少有一个 geom 接触 cube 才返回 True，但两组是同一列表，导致**任一 pad 接触 cube** 即判 True。穿模时单侧 pad 穿过 cube 也被判为"抓取成功"，策略因此学到穿模行为（SAC_21 的 34% 抓取率全是误判）。

#### 修复后的 `_important_geoms`（当前方案）

```python
@property
def _important_geoms(self):
    return {
        "left_finger": ["moving_jaw_pad", "moving_jaw_collision"],
        "right_finger": ["fixed_jaw_pad", "fixed_jaw_collision"],
        # _check_grasp 要求两组各自至少有一个 geom 与 cube 接触才判 True。
        # 分别只取一个 pad，确保两侧 pad 同时接触 cube 才算"夹住"，
        # 避免穿模时单侧接触被误判为抓取。
        "left_fingerpad": ["moving_jaw_pad"],   # 活动颚
        "right_fingerpad": ["fixed_jaw_pad"],   # 固定颚
    }
```

**关键设计**：`left_fingerpad` 只取活动颚 pad，`right_fingerpad` 只取固定颚 pad。`_check_grasp` 要求两组**各自**至少有一个 geom 与 cube 接触才返回 True，即两侧 pad 必须**同时**接触 cube 才算抓取成功。这避免了穿模时单侧接触的误判，确保奖励信号与物理事实一致。

> **演进说明**：误判 bug 的发现过程、对策略学习的负面影响（reward hacking）详见 [第 7.3 节](#73-layer-1判定层-_important_geoms-误判)。

---

## 2. 控制器架构

### 2.1 JOINT_POSITION 控制器原理

**文件**: `robosuite/controllers/parts/generic/joint_pos.py`
**配置**: `robosuite/controllers/config/robots/default_so101.json`

robosuite 的 `JointPositionController` 内部实现了**阻抗控制**（impedance control），计算流程：

```
1. 策略输出 action ∈ [-1, 1]^5（归一化关节位置增量）
2. scale_action: action × output_max → delta_qpos ∈ [-0.06, 0.06] rad
3. goal_qpos = current_qpos + delta_qpos
4. PD 控制: desired_torque = kp × (goal_qpos - qpos) + kd × (-qvel)
5. 重力补偿: torques = mass_matrix @ desired_torque + qfrc_bias
6. 写入 sim.data.ctrl[:5] = torques
```

**配置参数**:

| 参数 | 值 | 说明 |
|------|-----|------|
| `output_max/min` | ±0.06 rad | 每步最大关节角度变化（约 3.4°） |
| `kp` | 50 | 位置比例增益 |
| `damping_ratio` | 1.2 | 临界阻尼比 |
| `use_torque_compensation` | True (默认) | 启用重力补偿 + 质量矩阵前馈 |

### 2.2 `<position>` vs `<motor>` 执行器语义冲突

**这是导致 SAC_18/SAC_19 训练完全失败的根本原因。**

#### 问题机制

robosuite 的 `JointPositionController.run_controller()` 输出的是**力矩**（torque），然后写入 `sim.data.ctrl`。MuJoCo 根据 XML 中执行器类型解释 `ctrl` 的语义：

| 执行器类型 | MuJoCo 对 `ctrl` 的解释 | 力矩公式 |
|-----------|------------------------|---------|
| `<position>` | **位置目标** (目标角度) | `τ = kp × (ctrl - qpos) - kv × qvel` |
| `<motor>` | **直接力矩** | `τ = ctrl × gear` |

**原始 SO101 的 robot.xml 使用 `<position>` 执行器**：

```xml
<!-- 原始（错误）-->
<position class="sts3215" name="shoulder_pan" joint="shoulder_pan"
          forcerange="-3.35 3.35" ctrlrange="-1.91986 1.91986"/>
```

robosuite 控制器计算的力矩值（如 `[-0.001, -1.19, -0.72, -0.24, -0.002]`）被写入 `sim.data.ctrl`，MuJoCo 把这些**力矩值当作弧度目标**：

- 力矩 -1.19 Nm → MuJoCo 认为目标角度是 -1.19 rad → 关节被驱动到错误位置
- 力矩值被 `forcerange` clip 到 ±3.35 → 持续以最大力矩驱动
- **零动作时 EEF 上升 0.374m，任何方向的动作都让机械臂向上漂移**

#### 修复：改为 `<motor>` 执行器

```xml
<!-- 修复后 -->
<motor name="shoulder_pan" joint="shoulder_pan"
       ctrllimited="true" ctrlrange="-3.35 3.35" gear="1"/>
```

`<motor>` 执行器把 `ctrl` 直接当作力矩施加，与 robosuite 控制器的输出语义匹配。

**修复效果**：

| 指标 | 修复前 (`<position>`) | 修复后 (`<motor>`) |
|------|----------------------|-------------------|
| 零动作 100 步 EEF 漂移 | 0.374m (向上飞) | 0.000001m (稳定) |
| 关节最大漂移 | 1.03 rad | 0.000001 rad |
| 方向控制 | 所有方向都向上 | 双向可控 |

#### 为什么 gripper 保留 `<position>` 执行器

夹爪使用 `SimpleGripController`（GRIP 类型），它的 `run_controller()` 输出的是**速度/位置目标**（`self.vels`），不是力矩。MuJoCo 的 `<position>` 执行器接收这个值作为位置目标，执行自身的 PD 控制——这正好匹配 STS3215 舵机的真实行为（舵机内部做位置 PD 控制）。

**总结**：arm 用 `<motor>`（robosuite 做力矩控制），gripper 用 `<position>`（MuJoCo 做位置控制），各自语义匹配。

### 2.3 GRIP 夹爪控制器

**文件**: `robosuite/controllers/parts/gripper/simple_grip.py`

`SimpleGripController` 的控制流程：

```
1. 策略输出 action[5] ∈ [-1, 1]（gripper 维度）
2. scale_action: 映射到 actuator ctrlrange
3. 输出 vels（位置目标）
4. 写入 sim.data.ctrl[5] = vels
5. MuJoCo <position> 执行器: τ = 998.22 × (ctrl - qpos) - 2.731 × qvel
```

gripper 的 `<position>` 执行器参数 `kp=998.22, kv=2.731` 是根据 STS3215 舵机比例增益 16 计算的，模拟真实舵机的位置跟踪行为。

---

## 3. 奖励函数设计

### 3.1 robosuite 内置 Lift 奖励

**文件**: `robosuite/environments/manipulation/lift.py` L222-L396

Lift 任务的奖励分为三个阶段：

```
if _check_success():              # 阶段3: 成功
    reward = 2.25
elif reward_shaping:              # 阶段1+2: 稠密奖励
    reaching = 1 - tanh(10 * dist)    # 阶段1: 接近 [0, 1]
    reward += reaching
    if _check_grasp(...):             # 阶段2: 抓取 {0, 0.25}
        reward += 0.25
# 最终归一化: reward *= reward_scale / 2.25
```

| 阶段 | 公式 | 范围 | 说明 |
|------|------|------|------|
| Reaching | `1 - tanh(10 × dist)` | [0, 1] | dist=0→1, dist=0.1→0.24, dist=1.0→≈0 |
| Grasping | `_check_grasp()` ? 0.25 : 0 | {0, 0.25} | 二值，pad 接触 cube |
| Success | `_check_success()` ? 2.25 : 0 | {0, 2.25} | cube 抬起 > 桌面 + 4cm |

**成功条件** (`_check_success`)：`cube_height > table_height + 0.04`

### 3.2 `_check_grasp` 判定逻辑

**文件**: `robosuite/environments/manipulation/manipulation_env.py` L356-L405

```python
def _check_grasp(self, gripper, object_geoms):
    # gripper 是 GripperModel → 取 left_fingerpad 和 right_fingerpad 两组 geom
    g_geoms = [gripper.important_geoms["left_fingerpad"],
               gripper.important_geoms["right_fingerpad"]]
    # object_geoms 是 MujocoModel → 取 contact_geoms (如 cube_g0)
    o_geoms = object_geoms.contact_geoms

    # 每组都必须至少有一个 geom 与物体接触
    for g_group in g_geoms:
        if not self.check_contact(g_group, o_geoms):
            return False
    return True
```

`check_contact` 遍历 `sim.data.contact` 中所有活跃接触对，检查是否有任何一对满足 (g_group 中的 geom, o_geoms 中的 geom) 的组合。

**SO101 的适配**：`left_fingerpad=["moving_jaw_pad"]`（活动颚），`right_fingerpad=["fixed_jaw_pad"]`（固定颚），两组各自只取一个 pad。`_check_grasp` 要求两侧 pad **同时**接触 cube 才返回 True，确保是真正的"夹住"而非单侧穿模接触。这一配置是经过 SAC_21 误判 bug 后修正的（详见 [7.3 节](#73-layer-1判定层-_important_geoms-误判)）。

### 3.3 自定义 Reward Shaping Wrapper

**文件**: `robosuite/demos/so101_realistic/wrappers.py`

`SO101LiftRewardShapingWrapper` 在所有模式（含 easy）下叠加自定义 shaping 分量：

#### 奖励分量清单

| 分量 | 公式 | 范围 | 系数 | 作用 |
|------|------|------|------|------|
| `r_reach` (PBRS) | `α × (tanh(k×d_prev) - tanh(k×d_now))` | [-α, +α] | α=20, k=5 | 接近=正, 远离=负, 静止=0 |
| `r_grasp` | `w × _check_grasp()` | {0, w} | w=1.0 | 接触奖励（比内置 0.25 强 4×） |
| `r_lift` | `w × clip(Δh / target, 0, 1)` | [0, w] | w=1.5, target=0.04m | 连续抬升信号 |
| `r_grip_close` | `w × proximity × (1 - openness)` | [0, w] | w=0.5, threshold=0.10m | **接近时闭合夹爪引导** |
| `r_smooth` | `-w × Σ(a_t - a_{t-1})²` | [-∞, 0] | w=0.02 | 动作平滑度 |
| `r_vel` | `-w × Σ(qvel²)` | [-∞, 0] | w=0.002 | 关节速度 |
| `r_ee_vel` | `-w × Σ(Δeef²)` | [-∞, 0] | w=0.03 | 末端抖动 |
| `r_z_float` | `-w × max(0, eef_z - cube_z - threshold)` | [-∞, 0] | w=0.5, threshold=0.04m | 末端悬浮惩罚 |

#### PBRS 接近奖励原理

原设计 `r_reach = w × (1 - tanh(k × dist))` 是**绝对形式**——策略不动也得 +0.50/step，形成"远距离静止高原"。

改为 **PBRS（Potential-Based Reward Shaping）差分形式**：

```
势函数: F(s) = -α × tanh(k × dist)
差分奖励: r_reach = α × (tanh(k × d_prev) - tanh(k × d_now))
```

| 场景 | d_prev → d_now | r_reach |
|------|----------------|---------|
| 静止悬浮 | 0.4 → 0.4 | **0** (消除高原) |
| 接近 1cm | 0.05 → 0.04 | +0.96 |
| 远离 3cm | 0.07 → 0.10 | -1.18 |
| 全 episode 累计 | 0.4 → 0 | +7.6 |

#### `r_grip_close` 设计

这是解决"接近但不抓取"问题的关键信号：

```python
grip_openness = self._get_gripper_opening_ratio()  # 0=闭合, 1=张开
proximity = max(0, 1 - dist / threshold)            # 10cm内开始计分
r_grip_close = 0.5 * proximity * (1 - grip_openness)
```

- EEF 在 10cm 以内时，夹爪越闭合得分越高
- 远离 cube 时 `proximity=0`，奖励自动归零，不阻碍接近
- 与 `r_reach` PBRS 协同：接近时同时获得 reach 正梯度 + grip_close 正梯度

#### 最终 shaped reward 合成

```python
shaped_reward = (
    original_reward       # robosuite 内置 (reaching + grasping 或 success)
    + r_reach             # PBRS 接近差分
    + r_grasp             # 接触奖励 (w=1.0)
    + r_lift              # 抬升过渡 (w=1.5)
    + r_grip_close        # 闭合引导 (w=0.5)
    + r_smooth            # 动作平滑 (-)
    + r_vel               # 关节速度 (-)
    + r_ee_vel            # 末端抖动 (-)
    + r_z_float           # 悬浮惩罚 (-)
)
```

### 3.4 `lift_grasping_reward` 为什么看起来不是离散的

robosuite 的 `Lift.reward()` 中 grasping 奖励是严格二值的：

```python
if self._check_grasp(...):
    reward += 0.25   # {0, 0.25}
```

但 TensorBoard 上记录的不是每步原始值，而是**多个 step × 多个并行 env 的平均值**：

```python
# callbacks.py L153
means = {name: self._acc[name] / n for name in names}
self.logger.record(f"reward_breakdown/{name}", means[name])
```

例如 16 个 env × 10000 步中只有 1 步 1 个 env 触发了 `_check_grasp=True`：
- TensorBoard 值 = 0.25 / (16 × 10000) ≈ 0.000002

所以曲线看起来是连续的，但底层原始值是二值的。最大值 0.0023 (SAC_19) 意味着约 0.9% 的步骤成功抓取。

---

## 4. 训练优化

### 4.1 SAC 超参数

**文件**: `robosuite/demos/train_rl_sb3_so101_realistic.py`

```python
sac_kwargs = dict(
    learning_rate=3e-4,
    buffer_size=200_000,
    batch_size=256,
    tau=0.005,            # 软更新系数
    gamma=0.99,           # 折扣因子
    ent_coef="auto",      # 熵系数自动调节
    target_entropy=-1.5,  # 默认 -dim(A)/2 = -3，改为 -1.5 增加探索
    policy_kwargs=dict(net_arch=[256, 256], use_sde=False),
)
```

### 4.2 探索不足的诊断与修复

#### SAC_19 的问题

SAC_19 中 `ent_coef` 从 0.944 降到 0.0007（降 99.93%），策略在 500K 步后失去探索能力，收敛到"接近但不抓取"的局部最优。

#### 修复：调整 `target_entropy`

SAC 的熵正则项：`J = E[Q(s,a)] + α × H(π(·|s))`

- `target_entropy` 是熵的目标值，`α` 自动调节使策略熵趋近目标
- 默认 `target_entropy = -dim(A)/2 = -3`（动作维度 6）
- 改为 `-1.5`（更接近 0），策略需要维持更高熵，延长探索期

| 参数 | 默认值 | 修改后 | 效果 |
|------|--------|--------|------|
| `target_entropy` | -3.0 | -1.5 | 更高熵目标 → 更多探索 |
| `ent_coef` | "auto" | "auto" | α 自动调节 |

---

## 5. 诊断方法论与工具

### 诊断流程

```
1. TensorBoard 曲线分析 → 确认哪个 reward 分量为 0
2. Rollout 轨迹分析 → 确认 EEF/cube/gripper 的物理轨迹
3. 控制器零动作测试 → 确认控制器是否稳定
4. 执行器类型检查 → 确认 ctrl 语义是否匹配
5. 接触检测验证 → 确认 pad 是否能接触 cube
6. 奖励景观分析 → 确认梯度信号是否足够
```

### 工具脚本

| 脚本 | 位置 | 用途 |
|------|------|------|
| `analyze_sac19.py` | `tools/` | TensorBoard 日志分析，提取各 reward 分量曲线 |
| `diagnose_sac19.py` | `tools/` | Rollout 轨迹深度分析，生成 EEF/cube/gripper 轨迹图 |
| `visualize_rollout_offline.py` | `robosuite/demos/` | 从 .npz 文件绘制 rollout 诊断图（奖励分量、动作、EEF 高度） |

### 使用方法

```bash
# 分析 TensorBoard 日志
conda run -n robosuite python tools/analyze_sac19.py

# 分析 rollout 轨迹
conda run -n robosuite python tools/diagnose_sac19.py

# 离线可视化单个 rollout
conda run -n robosuite python robosuite/demos/visualize_rollout_offline.py \
    ./logs/sac_lift_so101_realistic/test_rollouts/rollout_test_ep000.npz

# 批量可视化
conda run -n robosuite python robosuite/demos/visualize_rollout_offline.py \
    ./logs/sac_lift_so101_realistic/test_rollouts/ --max-files 10
```

### TensorBoard 启动

```bash
conda run -n robosuite tensorboard --logdir=./logs/sac_lift_so101_realistic/ --port=6008
```

---

## 修改文件清单（第 6 章执行器语义修复 + 第 7 章穿模物理修复）

| 文件 | 修改内容 | 对应章节 |
|------|---------|---------|
| `robosuite/models/assets/robots/so101/robot.xml` | 5 个 arm 执行器从 `<position>` 改为 `<motor>`，修复 ctrl 语义冲突 | 6.6 |
| `robosuite/models/assets/grippers/so101_gripper.xml` | pad 从 box 改为 sphere (r=0.001)；moving_pad local 调整+z补偿；solref=0.02，margin=0.008；gripper damping=2.0；eef body pos 修正对齐 pad 中点 | 7.6.1, 7.6.2 |
| `robosuite/models/grippers/so101_gripper.py` | `_important_geoms` left/right fingerpad 分别只取一个 pad，要求两侧同时接触才判抓取成功 | 7.3 |
| `robosuite/environments/manipulation/lift.py` | cube 尺寸 [0.018-0.020]，density=8000，friction=[3,0.01,0.001]；`UniformRandomSampler` rotation=0 保证轴对齐 | 7.6.3 |
| `robosuite/demos/so101_realistic/wrappers.py` | easy 模式启用 reward shaping；新增 `r_grip_close`；PBRS 接近奖励；移除 `r_gripper_move` | 3.3 |
| `robosuite/demos/train_rl_sb3_so101_realistic.py` | SAC `target_entropy` 从 -3 改为 -1.5 增加探索 | 4.2 |
| `robosuite/demos/visualize_rollout_offline.py` | 分离聚合键与分量键，独立子图，自动检测奖励键 | 7.9 |
| `tools/analyze_training.py` | 通用化 SAC 分析工具（CLI 参数，自动检测指标键，支持任意 run） | 7.9 |

---

## 6. 完整诊断案例：从"抓取奖励为零"到"执行器语义冲突"

本章节以案例研究的形式，记录 SAC_18 训练失败到定位根因的完整诊断推理过程。适合面试时复述"如何排查一个 RL 训练不收敛的问题"。

### 6.1 问题背景

**症状**：SO101 机械臂在 robosuite Lift 任务上训练 100 万步后，TensorBoard 显示 `lift_grasping_reward` 始终为 0，`ep_rew_mean` 仅从 0.30 升至 0.52，成功率 0%。

**关键上下文**：此前已修复了三个 bug（geom 名字、单指夹爪判定、fingerpad 物理夹持），SAC_18 是修复后重新训练的结果。这意味着三个已修复的 bug 不是根因，存在更深层的问题。

**对比基准**：同一框架下 Panda 机械臂的 Lift 训练能成功收敛（`ep_rew_mean` 达到 2.0+），说明 robosuite 框架本身没问题，问题出在 SO101 特有的配置上。

### 6.2 分析方法：分层诊断

采用**自底向上**的分层诊断策略，从奖励检测层逐步深入到物理层、控制层：

```
Layer 1: 奖励检测层 — _check_grasp 是否能返回 True？
    ↓ (能，但训练中从未触发)
Layer 2: 物理接触层 — pad 是否能物理接触 cube？
    ↓ (能，但需要 EEF 到达 cube 附近)
Layer 3: 控制器层 — 控制器能否让 EEF 到达 cube？
    ↓ (不能，零动作时 EEF 上升 0.374m)
Layer 4: 执行器层 — ctrl 语义是否匹配？
    → 根因：position vs motor 语义冲突
```

每一层都用独立的诊断脚本验证，避免"猜测→修改→训练→失败"的低效循环。

### 6.3 Layer 1：奖励检测层

**问题**：`lift_grasping_reward` 始终为 0，是检测逻辑有 bug 还是物理上没接触？

**方法**：写诊断脚本，手动把 cube 放入夹爪，检查 `_check_grasp` 是否返回 True。

**发现**：
- pad geom 名称正确，`contype/conaffinity` 允许接触
- 手动把 cube 放入夹爪时 `_check_grasp=True`
- **结论：检测逻辑正确，问题在物理层——训练中夹爪从未接触 cube**

### 6.4 Layer 2：物理接触层

**问题**：训练中夹爪为什么接触不到 cube？

**方法**：分析 test_rollouts 中的 EEF 和 cube 轨迹。

**发现**：
- 5 条轨迹中 EEF-cube 最小距离均值为 0.117m（最近 0.059m），远超 0.05m 接触阈值
- EEF 初始 z≈0.87m（cube z≈0.82m），之后 EEF 反而**上升**到 1.2m 以上
- **策略学到了"向上飘"的行为，从未下降到 cube 高度**

**关键转折**：这不是策略探索不足的问题，而是控制器层面存在根本性缺陷——EEF 在零动作时也会上升。

### 6.5 Layer 3：控制器层

**问题**：零动作时 EEF 为什么会上升？

**方法**：发送零动作 50 步，记录关节 qpos 和 EEF 位置变化。

**发现**：
- 零动作 50 步后 EEF z 上升 **0.374m**（从 0.87m → 1.25m）
- shoulder_lift 关节漂移 -1.03 rad（大幅偏离初始位置）
- 逐关节测试 ±1 动作：**所有方向的动作都导致 EEF 上升**
- 控制器输出的力矩值被 MuJoCo clip 到 ±3.35 Nm 极限

**排除项**：
- `ramp_ratio=0.2`：`interpolation=null`，插值器未启用，无影响
- `reset_goal`：reset 后 `goal_qpos == joint_pos`，零动作不应产生位置误差
- 重力补偿：控制器使用 `qfrc_bias` 补偿，逻辑正确

**结论**：控制器输出正确，但 MuJoCo 执行端对输出值的解释错误。

### 6.6 Layer 4：执行器层（根因）

**问题**：MuJoCo 如何解释控制器写入的 `sim.data.ctrl` 值？

**方法**：检查 `sim.model.actuator_biastype` 确认执行器类型。

**发现**：
- SO101 的 robot.xml 使用 `<position>` 执行器
- robosuite `JointPositionController` 输出力矩，写入 `sim.data.ctrl`
- MuJoCo `<position>` 执行器把 `ctrl` 当作**位置目标**（弧度），而非力矩
- 力矩值 -1.19 Nm → MuJoCo 认为目标角度是 -1.19 rad → 关节飞到极限

**为什么 Panda 没有这个问题**：Panda 的 XML 使用 `<motor>` 执行器，`ctrl` 直接当作力矩，与 robosuite 控制器语义匹配。

### 6.7 修复与验证

**修复**：将 5 个 arm 执行器从 `<position>` 改为 `<motor>`，保留 gripper 的 `<position>`。

**验证结果**：

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 零动作 100 步 EEF 漂移 | 0.374m（向上飞） | 0.000001m（稳定） |
| 关节最大漂移 | 1.03 rad | 0.000001 rad |
| 方向控制 | 所有方向向上 | 双向可控 |
| IK 可达性 | 无法到达 cube | 可达 cube 附近 4.8mm |

### 6.8 面试关键问答

#### Q1: 如何判断是奖励检测问题还是策略学习问题？

**A**: 写诊断脚本，手动构造"抓取成功"的场景（把 cube 放入夹爪），检查 `_check_grasp` 是否返回 True。如果返回 True，说明检测逻辑正确，问题在策略/控制层面；如果返回 False，说明检测逻辑有 bug。这是最快速的隔离方法——不需要跑训练就能定位问题层级。

#### Q2: 为什么零动作时机械臂会上升？

**A**: 这是执行器类型不匹配导致的。robosuite 的 `JointPositionController` 内部做了 PD 控制 + 重力补偿，输出的是力矩值，写入 `sim.data.ctrl`。但 SO101 的 XML 用了 `<position>` 执行器，MuJoCo 把这个力矩值当作**位置目标**（弧度）来解释。比如力矩 -1.19 Nm 被当作"目标角度 -1.19 rad"，关节就被驱动到错误位置，而且 `forcerange` clip 到 ±3.35 后持续以最大力矩驱动，导致机械臂失控上飘。

#### Q3: 为什么 Panda 没有这个问题？

**A**: Panda 的 XML 用的是 `<motor>` 执行器，`ctrl` 直接当作力矩施加，与 robosuite 控制器的输出语义天然匹配。SO101 的 XML 沿用了 LeRobot 项目中 STS3215 舵机的 `<position>` 执行器配置（因为真实舵机内部做位置 PD 控制），但在 robosuite 框架中，控制器已经做了力矩计算，不需要 MuJoCo 再做一次位置控制。两层 PD 控制叠加导致语义冲突。

#### Q4: 为什么 gripper 保留 `<position>` 执行器？

**A**: 夹爪使用 `SimpleGripController`（GRIP 类型），它的 `run_controller()` 输出的是位置目标（`self.vels`），不是力矩。MuJoCo 的 `<position>` 执行器接收这个值作为位置目标，执行自身的 PD 控制——这正好匹配 STS3215 舵机的真实行为（舵机内部做位置 PD 控制）。所以 arm 用 `<motor>`（robosuite 做力矩控制），gripper 用 `<position>`（MuJoCo 做位置控制），各自语义匹配。

#### Q5: 诊断过程中如何避免"修改→训练→失败"的低效循环？

**A**: 采用分层诊断策略，每一层用独立脚本验证，不依赖训练结果：
1. 先验证检测逻辑（手动构造成功场景）
2. 再验证物理可达性（分析 rollout 轨迹）
3. 再验证控制器稳定性（零动作测试）
4. 最后检查执行器语义（actuator_biastype）

每一层只需要几秒钟的脚本运行，而不是几小时的训练。这样能在几分钟内定位到根因，而不是反复训练试错。

#### Q6: fingerpad 设计的核心思路是什么？

**A**: SO101 是单指 hinge 夹爪，与 Panda 的平行双指夹爪有本质区别。原始 mesh 碰撞体存在两个问题：①接触面积大、法线方向不确定，hinge 旋转会"推"cube 而非夹住；②摩擦系数不足。fingerpad 的设计思路是"在真实夹爪上贴一层高摩擦橡胶垫片"——最初用小尺寸 box 实现近似线接触，配合 5 倍摩擦系数。但后续发现 box 的长尺寸会意外对齐闭合方向导致穿模（详见第 7 章），最终改为各向同性的 sphere 点接触 + condim=6 + friction=5.0，彻底消除方向依赖。同时将原始 mesh 碰撞体设为 `contype=0` 不参与碰撞，避免大面积面接触干扰。

#### Q7: `_important_geoms` 中两侧 fingerpad 应该如何配置？

**A**: `_check_grasp` 要求 `left_fingerpad` 和 `right_fingerpad` 两组**各自**至少有一个 geom 与 cube 接触才返回 True。最初曾将两组都设为 `[moving_jaw_pad, fixed_jaw_pad]`（同一列表），试图适配 SO101 hinge 闭合时两侧可能不同时接触的特点——这样任一侧 pad 接触即可判 True。但后续发现这会导致**误判**：穿模时单侧 pad 穿过 cube 也被判为"抓取成功"，策略因此学到穿模行为（SAC_21 的 34% 抓取率全是误判，详见第 7.3 节）。**正确做法**是两组分别只取一个 pad（`left_fingerpad=["moving_jaw_pad"]`，`right_fingerpad=["fixed_jaw_pad"]`），要求两侧 pad 同时接触 cube 才算抓取，确保奖励信号与物理事实一致。

### 6.9 诊断工具脚本

| 脚本 | 位置 | 用途 |
|------|------|------|
| `debug_grasp.py` | 项目根目录 | 验证 `_check_grasp` 检测逻辑、pad geom、接触检测 |
| `debug_controller.py` | 项目根目录 | 诊断控制器零动作漂移、方向控制、执行器类型 |
| `debug_reward_landscape.py` | 项目根目录 | 分析奖励梯度、EEF 接近 cube 的可达性 |
| `debug_jaw_measure.py` | 项目根目录 | 测量 jaw 间距 vs gripper qpos、摩擦系数、cube 尺寸 |
| `visualize_rollout_offline.py` | `robosuite/demos/` | 从 .npz 文件绘制 rollout 诊断图 |

```bash
# 验证抓取检测
conda run -n robosuite python debug_grasp.py

# 诊断控制器稳定性
conda run -n robosuite python debug_controller.py

# 分析奖励景观
conda run -n robosuite python debug_reward_landscape.py

# 测量夹爪间距
conda run -n robosuite python debug_jaw_measure.py

# 离线可视化 rollout
conda run -n robosuite python robosuite/demos/visualize_rollout_offline.py \
    ./logs/sac_lift_so101_realistic/test_rollouts/
```

---

## 7. 完整诊断案例：pad 穿模与物理参数优化（SAC_21 / SAC_22）

本章节记录第二次大规模诊断：执行器语义修复后（第 6 章），SAC_21 仍然"穿模"夹不住 cube，SAC_22 的 `reward_grasp` 甚至始终为 0。问题从控制层深入到**几何与物理接触层**，涉及旋转矩阵分析、cube 投影计算、接触求解器调参。适合面试时复述"如何排查一个仿真物理接触异常"。

### 7.1 问题背景

**症状演进**：
| 阶段 | 训练结果 | 关键现象 |
|------|---------|---------|
| SAC_18 | `lift_grasping_reward=0` | EEF 上飘，不下降（第 6 章已修复执行器语义） |
| SAC_21 | `reward_grasp≈0.34`（34% 步数） | 夹爪"穿模"：pad 穿过 cube 而非夹住，cube 被弹开 7.7cm |
| SAC_22 | `reward_grasp=0`（全程零） | 修复 `_important_geoms` 后判定变严格，但穿模仍存在 |

**关键矛盾**：SAC_21 的 34% 抓取率是**误判**（`_important_geoms` 配置错误），SAC_22 修复判定后 `reward_grasp=0` 是**真实结果**——夹爪确实从未同时用两侧 pad 接触 cube。

**诊断核心问题**：为什么 pad 会穿过 cube 而不是夹住？为什么接触力把 cube 弹开 7.7cm？

### 7.2 分析方法：几何—物理—判定三层诊断

```
Layer 1: 判定层 — _check_grasp 为什么 SAC_21 误判、SAC_22 全零？
    ↓ (_important_geoms 两组设为同一列表 → 任一 pad 接触即判 True)
Layer 2: 几何层 — pad 为什么穿过 cube？
    ↓ (pad 世界 z 差 4.6cm，连线倾斜，cube 投影 > 有效内间隙)
Layer 3: 物理层 — 接触力为什么把 cube 弹开？
    ↓ (cube 太轻 + margin 内软接触力推动 cube)
```

### 7.3 Layer 1：判定层 — `_important_geoms` 误判

**问题**：SAC_21 的 `reward_grasp` 有 34% 非零，但夹爪明显穿模；SAC_22 修复后变 0。

**根因**：`_check_grasp` 要求 `left_fingerpad` 和 `right_fingerpad` 两组**各自**至少有一个 geom 与 cube 接触才返回 True。原配置：
```python
"left_fingerpad": ["moving_jaw_pad", "fixed_jaw_pad"],
"right_fingerpad": ["moving_jaw_pad", "fixed_jaw_pad"],  # 两组相同！
```
两组是同一列表，导致**任一 pad 接触 cube** 时两组都能找到接触，`_check_grasp` 误判 True。穿模时单侧 pad 穿过 cube 也会被判为"抓取成功"。

**修复**：让两组各自只取一个 pad，要求两侧**同时**接触：
```python
"left_fingerpad": ["moving_jaw_pad"],   # 活动颚
"right_fingerpad": ["fixed_jaw_pad"],   # 固定颚
```

**面试要点**：这是一个"奖励信号与物理事实脱节"的典型 bug。策略利用了误判信号——它学会"让 pad 穿过 cube"就能获得 `reward_grasp`，而不是真正夹住 cube。修复判定后信号归零，策略才能学到正确的抓取行为。

### 7.4 Layer 2：几何层 — 旋转矩阵分析与 cube 投影

**问题**：pad 为什么穿过 cube？

**方法**：用 `debug_tunneling.py` 测量 pad 世界坐标，计算 pad-pad 连线方向，并求 `right_gripper` body 的世界旋转矩阵 R。

**关键测量**：
```
fixed_jaw_pad  世界 z = 0.924
moving_jaw_pad 世界 z = 0.878
z 差 = 0.046m  (4.6cm！)
```

**旋转矩阵分析**（面试核心）：
```
R = [[ 0.252,  0.154, -0.955],
     [-0.522,  0.853,  0.   ],
     [ 0.815,  0.498,  0.296]]
R[2,0]=0.815  R[2,1]=0.498  R[2,2]=0.296
```
- R[2,2]=0.296：local z 轴在世界 z 方向投影仅 0.296（right_gripper body 近似水平躺置）
- 要让 fixed_pad 世界 z 降低 4.6cm 对齐 moving_pad：
  - 只调 local z：需 Δz = -0.046/0.296 = -0.155m（pad 偏离 jaw 15cm，不可行）
  - 只调 local x：需 Δx = -0.046/0.815 = -0.056m（会与 moving_pad 交叉，不可行）
  - 组合：Δx=-0.034 + Δy=-0.037（仍太大且交叉）
- **结论：机械结构限制，无法让 pad 世界 z 对齐**

**cube 投影计算**：
pad-pad 连线单位向量 d，cube 沿 d 方向的投影半尺寸 = half × (|dx|+|dy|+|dz|)/|d|（轴对齐 cube 的 L1 范数投影）。
```
|d| = 0.0694m,  (|dx|+|dy|+|dz|)/|d| = 1.624
cube half = 0.0217m → 全投影 = 0.0217 × 1.624 × 2 = 0.0706m
有效内间隙 = 0.0694 - 2×0.001 = 0.0674m
穿透 = 0.0706 - 0.0674 = 0.0032m
```
连线倾斜使 cube 投影从边长 0.043m 放大到 0.071m，超过有效内间隙 → 穿透。

**为什么 Panda 没有这个问题**：Panda 平行夹爪两 pad 在同一水平面，连线水平，cube 投影 = 边长，不会放大。

### 7.5 Layer 3：物理层 — 动态穿透 vs 静态穿透

**关键区分**：
- **静态穿透**：cube 固定在 pad 中点，扫描 gripper angle。最深 -0.0046m @angle=-0.1745（最闭合角度）。
- **动态穿透**：闭合过程中实测的最大穿透。原始配置 -0.0078m（比静态更深）。

**为什么动态更深**：cube 被接触力弹开后，pad 追上弹开的 cube 产生**二次深穿透**。这是轻物体被大力推动的典型现象。

**接触力分析**：
- 原始配置：穿透 0.0078m → 法向力 142N
- cube 质量 ≈ 0.074kg（density=1000）→ 加速度 1239 m/s² → 0.01s 位移 6.2cm（与实测 7.7cm 吻合）
- **核心矛盾**：即使几何上不穿透，margin 内的软接触力仍会把轻 cube 推开

### 7.6 解决方法：多维参数协同优化

由于无法让 pad 世界 z 对齐（机械限制），采用**几何 + 物理 + 判定**多维协同方案：

#### 7.6.1 pad 几何优化（so101_gripper.xml）

| 参数 | 原值 | 新值 | 理由 |
|------|------|------|------|
| pad 半径 | 0.003 | 0.001 | 增大有效内间隙 0.004m |
| moving_pad local x | -0.036 | -0.044 | 增大 pad 间距 |
| moving_pad local y | 0 | 0.0043 | 补偿 hinge 旋转导致的 z 降低 |
| solref | "0.005 0.5" | "0.02 1.0" | 最优刚度（0.05 太软穿透反增，0.005 太硬力大） |
| margin | 0.003 | 0.008 | 提前检测接触防漏检（0.002 太小穿透增大） |
| gripper damping | 0.60 | 2.0 | 减慢闭合速度减小冲击 |

**solref 调参的关键发现**（面试亮点）：
- solref=0.005（硬）：穿透 0.0078m，力 142N
- solref=0.01：穿透 0.003m，力 96.6N
- solref=0.02（最优）：穿透 0.0025m，力 108N
- solref=0.05（软）：穿透 0.0092m（反增！），力 84N
- **结论**：太软导致接触约束无法阻止穿透，存在最优刚度点

#### 7.6.2 eef body 位置修正

moving_pad local 改变后 pad 中点偏移，但 eef body pos 未更新，导致 grip_site 与 pad 中点差 4.5mm。机器人按 grip_site 定位时 pad 系统性偏离 cube。

**修正**：重新计算 pad 中点（right_gripper 坐标系）= (-0.0032, -0.0002, -0.0106)，更新 eef body pos。修正后误差 0.04mm。

#### 7.6.3 cube 物理参数优化（lift.py）

| 参数 | 原值 | 新值 | 理由 |
|------|------|------|------|
| 尺寸 | [0.020-0.022] | [0.018-0.020] | 减小沿连线投影，静态不穿透 |
| density | 1000 | 8000 | 增大质量 8 倍，更难被弹开 |
| friction | [1,0.005,0.0001] | [3,0.01,0.001] | 增大与桌面+pad 摩擦 |

**关键权衡**：减小尺寸让 cube 更轻（质量 ∝ size³），接触力弹得更远。因此必须同时增大 density 补偿。最终动态穿透从 0.0078m 降到 **0**。

### 7.7 效果对比

| 指标 | 原始 (SAC_21) | 修复后 |
|------|--------------|--------|
| 动态穿透深度 | 0.0078m | **0** |
| 静态最深穿透 | -0.0068m | -0.0046m |
| 最大法向力 | 142N | 224N（cube 更重，约束更硬） |
| grip_site 对齐误差 | 4.5mm | 0.04mm |
| _check_grasp 误判 | 2500/2500（全误判） | 14/2500（严格判定） |
| cube 位移（诊断最坏情况） | 7.7cm | 6.8cm |

**注意**：诊断脚本是最坏情况（cube 固定在空中，持续用力闭合）。实际训练时 cube 在桌面有摩擦支撑，策略控制闭合力度，效果会更好。

### 7.8 面试关键问答

#### Q1: 为什么 pad 世界 z 对不齐？如何分析？

**A**: SO101 的 moving_jaw body 相对 right_gripper 有绕 x 轴 90° 的姿态变换，加上 hinge 旋转，导致 moving_pad 的 local 坐标到世界 z 的映射复杂。我通过 mujoco 的 `body_xmat` 取出 right_gripper 的世界旋转矩阵 R，发现 R[2,2]=0.296（local z 轴在世界 z 方向投影极小），意味着调 local z 对世界 z 影响很弱。要让 fixed_pad 世界 z 降低 4.6cm，需 local z 降 15.5cm（脱离 jaw），或 local x 降 5.6cm（与 moving_pad 交叉），都不可行。这是机械结构固有限制。

#### Q2: cube 沿 pad-pad 连线的投影怎么算？为什么连线倾斜会让穿透变严重？

**A**: 对于轴对齐的 cube（rotation=0），沿任意方向 d 的投影半尺寸 = half × (|dx|+|dy|+|dz|)/|d|（L1 范数投影）。当 pad-pad 连线水平时（Panda），(|dx|+|dy|+|dz|)/|d| = 1，投影 = 边长。当连线倾斜时，这个比值 > 1（本例 1.624），投影被放大 1.624 倍，从 0.043m 变成 0.071m，超过有效内间隙 0.067m 就穿透。这是 hinge 单指夹爪（pad 走弧线、两 pad 不共面）特有的问题。

#### Q3: 为什么 solref 不是越小越好（越硬）或越大越好（越软）？

**A**: solref 是 MuJoCo 接触约束的 timeconst，刚度 ∝ 1/timeconst²。
- 太硬（0.005）：穿透瞬间产生巨大力（142N），把 cube 弹开
- 太软（0.05）：接触约束无法阻止 pad 深入 cube，穿透反而增大到 0.0092m
- 最优（0.02）：穿透最小（0.0025m），是刚度与约束力的平衡点
这类似真实世界中"硬橡胶 vs 软海绵"——太硬会弹开物体，太软夹不住，需要中等硬度。

#### Q4: 为什么减小 cube 尺寸反而让穿透变严重？

**A**: 几何上减小尺寸应该减小投影、减小穿透。但 cube 质量 ∝ size³，减小 10% 尺寸质量减小 22%。接触力把更轻的 cube 弹得更远，pad 追上弹开的 cube 产生二次深穿透，动态穿透从 0.0026m 反增到 0.0078m。这说明**几何优化必须配合质量优化**——我同时增大 density（1000→8000）补偿质量损失，最终动态穿透降到 0。这是"几何—物理耦合"的典型例子。

#### Q5: `_important_geoms` 误判为什么会导致策略学到错误行为？

**A**: RL 策略优化奖励信号，不管信号是否与物理事实一致。当 `_check_grasp` 误判（任一 pad 接触即判 True）时，策略发现"让 pad 穿过 cube"就能获得 `reward_grasp`，于是收敛到穿模行为。修复判定后 `reward_grasp=0`，策略失去错误信号，才能探索真正的抓取动作。这体现了**奖励设计的一致性原则**——奖励判定必须与任务目标的物理定义一致，否则策略会 reward hacking。

#### Q6: margin 在接触检测中起什么作用？为什么 0.008 比 0.002 好？

**A**: margin 是 MuJoCo 的"提前检测距离"——当两 geom 距离 < margin 时就开始产生软接触力（按 solref 刚度），防止一帧内从 dist>0 突变到 dist<0（漏检）。
- margin=0.002：太小，pad 闭合速度快时一帧内穿透，约束来不及响应，穿透 0.00098m
- margin=0.008：足够大，pad 在 8mm 外就开始减速，动态穿透 = 0
- 但 margin 也不能太大，否则 pad 还没接近 cube 就开始推它。0.008 是平衡点。

#### Q7: 诊断中如何区分"几何穿透"和"物理弹开"？

**A**: 用两种测试区分：
1. **静态扫过**：cube 固定不动，扫描 gripper angle，测 dist。这是纯几何问题。
2. **动态闭合**：cube 自由运动，持续闭合夹爪，测 cube 位移和法向力。这是几何+物理耦合。
如果静态穿透小但动态位移大，说明是物理弹开（接触力推 cube）。本例中修复后静态仍有 -0.0046m 穿透，但动态穿透=0且 cube 仍被弹开 6.8cm，说明剩余问题是接触力推轻 cube，需要增大 cube 质量而非继续调几何。

### 7.9 诊断工具

| 脚本 | 用途 |
|------|------|
| `tools/debug_tunneling.py` | 测量 pad 世界坐标、接触力、穿透深度、cube 位移、_check_grasp |
| `tools/analyze_training.py` | 通用 SAC 训练分析（任意 run，自动检测指标键） |
| `robosuite/demos/visualize_rollout_offline.py` | 从 npz 绘制 rollout 图（分离聚合/分量奖励） |

```bash
# 穿模深度诊断
conda run -n robosuite python tools/debug_tunneling.py

# 通用训练分析
conda run -n robosuite python tools/analyze_training.py \
    --logdir ./logs/sac_lift_so101_realistic/ --run SAC_22

# rollout 可视化（奖励曲线分离显示）
conda run -n robosuite python robosuite/demos/visualize_rollout_offline.py \
    ./logs/sac_lift_so101_realistic/test_rollouts/rollout_test_ep000.npz
```

### 7.10 修改文件清单（本次会话）

| 文件 | 修改内容 |
|------|---------|
| `robosuite/models/assets/grippers/so101_gripper.xml` | pad r=0.001，moving_pad local 调整+z补偿，solref=0.02，margin=0.008，damping=2.0，eef body pos 修正 |
| `robosuite/models/grippers/so101_gripper.py` | `_important_geoms` left/right fingerpad 分别只取一个 pad |
| `robosuite/environments/manipulation/lift.py` | cube 尺寸[0.018-0.020]，density=8000，friction=[3,0.01,0.001] |
| `robosuite/demos/visualize_rollout_offline.py` | 分离聚合键与分量键，独立子图 |
| `tools/analyze_training.py` | 通用化 SAC 分析工具（CLI 参数，自动检测） |
