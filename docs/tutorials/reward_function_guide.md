# robosuite 奖励函数设计完全指南

> 本文档面向刚接触 robosuite 与 MuJoCo 的开发者，系统讲解如何从零开始理解环境架构、
> 读取仿真状态、设计并实现自定义奖励函数。内容覆盖：代码架构、MuJoCo 核心概念、
> `self.sim.data` 数据结构、robosuite 便利封装、奖励函数设计模板、实战示例、调试技巧、
> 学习资源与常见陷阱。

---

## 目录

- [1. 代码架构与继承关系](#1-代码架构与继承关系)
- [2. MuJoCo 的核心概念](#2-mujoco-的核心概念)
- [3. self.sim.data 全景地图](#3-selfsimdata-全景地图)
- [4. robosuite 在 MuJoCo 之上的便利封装](#4-robosuite-在-mujoco-之上的便利封装)
- [5. 奖励函数与环境的调用关系](#5-奖励函数与环境的调用关系)
- [6. 写奖励函数的"流水线模板"](#6-写奖励函数的流水线模板)
- [7. 常见奖励类型"配方表"](#7-常见奖励类型配方表)
- [8. 形状函数怎么选](#8-形状函数怎么选)
- [9. 奖励函数优化的深层原理](#9-奖励函数优化的深层原理)
- [10. Lift 任务奖励函数逐行剖析](#10-lift-任务奖励函数逐行剖析)
- [11. 自定义奖励函数的三种方式](#11-自定义奖励函数的三种方式)
- [12. 调试技巧](#12-调试技巧)
- [13. 常见陷阱与解决方案](#13-常见陷阱与解决方案)
- [14. 学习路径与推荐资源](#14-学习路径与推荐资源)
- [15. 可立即运行的最小示例](#15-可立即运行的最小示例)
- [16. 总结](#16-总结)

---

## 1. 代码架构与继承关系

### 1.1 类继承层级

```
MujocoEnv (robosuite/environments/base.py)
   └── RobotEnv (robosuite/environments/robot_env.py)
          └── ManipulationEnv (robosuite/environments/manipulation/manipulation_env.py)
                 ├── Lift          (manipulation/lift.py)
                 ├── PickPlace     (manipulation/pick_place.py)
                 ├── Stack         (manipulation/stack.py)
                 ├── Door          (manipulation/door.py)
                 ├── NutAssembly   (manipulation/nut_assembly.py)
                 ├── ToolHang      (manipulation/tool_hang.py)
                 ├── Wipe          (manipulation/wipe.py)
                 └── TwoArm*       (双臂系列：two_arm_lift / handover / peg_in_hole / transport)
```

### 1.2 各层职责分工

| 层级 | 文件 | 职责 |
|---|---|---|
| **第 1 层：仿真基类** | `base.py` | 只关心 MuJoCo 仿真器本身：模型加载、`sim.forward/step`、渲染器、observables 管理、`reset/step` 主循环。定义 `reward()` / `_check_success()` 为抽象方法，强制子类实现。 |
| **第 2 层：机器人环境** | `robot_env.py` | 在仿真器之上接入"机器人"概念：实例化机器人、配置控制器、相机观测（RGB/深度/分割）、初始化噪声。`_pre_action` 把动作拆分给每个机器人的控制器。`reward` / `_check_success` 仍为抽象。 |
| **第 3 层：操作任务基类** | `manipulation_env.py` | 加入"被操作物体"相关工具：`_check_grasp`（检测夹爪是否抓住物体）、`_gripper_to_target`（夹爪到目标的距离）、可视化辅助。**仍不实现** `reward` / `_check_success`，留给具体任务。 |
| **第 4 层：具体任务** | `lift.py` 等 | 每个文件是一个具体任务，必须实现 `_load_model`（搭场景）、`_setup_observables`（配观测）、`reward`（奖励函数）、`_check_success`（成功条件）。 |

### 1.3 类比理解

可以把这套结构想象成**搭乐高**：

- `MujocoEnv` 是"桌子"——提供工作台
- `RobotEnv` 是"机械臂"——放在桌子上
- `ManipulationEnv` 是"工具箱"——提供抓取判定等通用工具
- `Lift` 是"具体作品"——把上述组合起来做一个具体任务

---

## 2. MuJoCo 的核心概念

写不出奖励函数，本质上是因为缺两块知识：

| 知识缺口 | 具体表现 |
|---|---|
| **MuJoCo 仿真器的数据结构** | 不知道 `self.sim.data` 里有什么、怎么取 |
| **机器人/几何概念** | 分不清 body / site / geom / frame，不知道"夹爪位置"指哪里 |

补齐这两块后，奖励函数不过是"读状态 → 算差值 → 套函数"的流水线。

### 2.1 MuJoCo 的"世界观"——3 个核心概念

MuJoCo 用一个 XML（MJCF）文件描述世界，世界由三类元素组成：

```
worldbody
└── body (刚体，有质量、位姿)
     ├── geom (几何形状，用于碰撞/接触)        ← 决定"碰到没有"
     ├── site (坐标系标记点，无质量)            ← 决定"位置在哪里"
     ├── joint (关节，连接父body)
     └── body (子刚体，递归定义)
```

### 2.2 body（刚体）

- **是什么**：有质量、有惯量的刚体，是物理仿真的基本单元
- **位姿**：6D 位姿（3D 位置 + 3D 朝向，或四元数）
- **怎么取**：

  ```python
  self.sim.data.body_xpos[body_id]      # 形状 (3,) 世界坐标位置
  self.sim.data.body_xquat[body_id]     # 形状 (4,) 世界坐标四元数 (w,x,y,z)
  self.sim.data.body_xvelp[body_id]    # 形状 (3,) 世界系下线速度
  ```

### 2.3 site（标记点）

- **是什么**：附属于 body 的坐标系锚点，**没有质量**，常用于：
  - 标记夹爪的"夹持中心"（grip_site）
  - 标记相机位置
  - 标记传感器位置
- **怎么取**：

  ```python
  self.sim.data.get_site_xpos("grip_site")    # 用名字取位置
  self.sim.data.site_xpos[site_id]           # 用 id 取位置
  ```

- **为什么夹爪位置用 site 而不用 body**：因为夹爪由多个 body 组成（左右指头、掌部），
  我们要的是"夹持中心"这个抽象点，而不是某个具体 body。

### 2.4 geom（几何形状）

- **是什么**：碰撞检测用的几何体（box / sphere / capsule / mesh 等）
- **作用**：决定哪些物体能互相接触，是判定"抓住/碰到"的关键
- **怎么用**：通过 [`check_contact`](../../robosuite/environments/base.py) 检测两组 geom 之间是否有接触

### 2.5 三者的关系图

```
                robot (body)
                /    |    \
        gripper  base  arm
         /  \
   finger_l  finger_r   ← body
      |         |
   fingerpad_l fingerpad_r  ← geom (接触判定用)
      ↑
   grip_site  ← site (位置标记用，挂在 gripper body 上)
```

---

## 3. self.sim.data 全景地图

这是 MuJoCo 仿真器内部状态的总入口。你写奖励函数的原料几乎都来自这里：

```python
# ───── 刚体状态（最常用） ─────
self.sim.data.body_xpos       # (nbody, 3)   所有刚体的世界坐标位置
self.sim.data.body_xquat      # (nbody, 4)   所有刚体的世界坐标四元数
self.sim.data.body_xvelp      # (nbody, 3)   世界系线速度
self.sim.data.body_xvelr      # (nbody, 3)   世界系角速度

# ───── 标记点与几何体 ─────
self.sim.data.site_xpos       # (nsite, 3)   所有 site 世界坐标
self.sim.data.geom_xpos       # (ngeom, 3)   所有 geom 世界坐标

# ───── 关节状态（低层） ─────
self.sim.data.qpos            # (nq,)        所有关节位置（广义坐标）
self.sim.data.qvel            # (nv,)        所有关节速度
self.sim.data.ctrl            # (nu,)        控制输入（你传的 action）

# ───── 传感器读数 ─────
self.sim.data.sensordata      # (nsensordata,) 所有传感器读数（force/IMU/触觉等）

# ───── 高级查询（推荐用 name-based API） ─────
self.sim.data.get_body_xpos("cube_body")
self.sim.data.get_site_xpos("grip_site")
self.sim.data.get_geom_xpos("cube_geom")
self.sim.data.body_name2id("cube_body")    # 名字 → id
```

> **关键经验**：能用 `get_xxx_xpos("name")` 就别用 `xxx_xpos[id]`——id 容易因模型改动而错位，
> name 更稳。

---

## 4. robosuite 在 MuJoCo 之上的便利封装

`ManipulationEnv` 把常见操作打包成几个高层方法，**你写奖励时基本只用这些**，不必直接碰 `sim.data`：

### 4.1 `_gripper_to_target`：夹爪到目标的距离

**实现核心**：

```python
gripper_pos = self.sim.data.get_site_xpos(gripper.important_sites["grip_site"])
target_pos  = self.sim.data.get_body_xpos(target.root_body)
diff = target_pos - gripper_pos
return np.linalg.norm(diff) if return_distance else diff
```

**典型用法**：

```python
dist = self._gripper_to_target(
    gripper=self.robots[0].gripper,    # 夹爪模型
    target=self.cube.root_body,        # 目标 body 名（注意传的是 root_body 字符串）
    target_type="body",                # 告诉它 target 是 body / site / geom
    return_distance=True,              # True 返回标量距离，False 返回 (3,) 向量
)
# 常见奖励: 1 - tanh(k * dist)
```

### 4.2 `_check_grasp`：判断是否抓住物体

**实现核心**：检查夹爪的 `left_fingerpad` 和 `right_fingerpad` 两组 geom 是否**都**与物体 geom 接触。

**典型用法**：

```python
if self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube):
    reward += 0.25
```

### 4.3 `check_contact`：任意两组 geom 接触判定

底层 API，`_check_grasp` 内部就是用它。适合自定义接触判定：

```python
# 判断机械臂任意部分是否碰到桌子
self.check_contact(self.robots[0].robot_model.contact_geoms, self.table.contact_geoms)
```

### 4.4 `get_contacts`：获取与某模型接触的所有 geom

返回一个 set，适合做更细致的分析：

```python
contacts = self.get_contacts(self.cube)
if "table" in " ".join(contacts):  # 立方体碰到桌子了
    ...
```

### 4.5 写奖励函数时可用的"工具箱"速查表

| 方法 | 作用 | 返回 |
|---|---|---|
| `_gripper_to_target` | 计算夹爪到目标的距离/向量 | `np.array` 距离向量或 float 距离 |
| `_check_grasp` | 判断夹爪是否抓住物体 | `bool` |
| `self.sim.data.body_xpos[id]` | 获取任意刚体世界坐标 | `np.array` shape (3,) |
| `self.sim.data.body_xquat[id]` | 获取刚体姿态（四元数） | `np.array` shape (4,) |
| `self.sim.data.qpos` / `qvel` | 所有关节位置/速度 | `np.array` |
| `self.sim.data.contact_...` | 接触力信息 | 各种数组 |
| `check_contact` | 任意两个 geom 组是否接触 | `bool` |
| `get_contacts` | 获取与某模型接触的所有 geom 名 | `set` |

---

## 5. 奖励函数与环境的调用关系

### 5.1 奖励函数被调用的位置

在 `base.py` 的 `step` → `_post_action` 中：

```python
def step(self, action):
    ...
    reward, done, info = self._post_action(action)  # 这里计算奖励
    ...

def _post_action(self, action):
    reward = self.reward(action)   # ← 调用子类实现的 reward
    self.done = (self.timestep >= self.horizon) and not self.ignore_done
    return reward, self.done, {}
```

也就是说：

- **环境**负责推进仿真、收集观测、判断 done
- **奖励函数**由具体任务实现，每一步 `step` 末尾被自动调用
- 奖励函数可以**读取仿真器的任何状态**（通过 `self.sim.data.body_xpos` 等），但不应该修改状态

### 5.2 关键设计模式：奖励与成功条件分离

robosuite 把"奖励"和"成功判定"分成两个方法：

| 方法 | 返回 | 作用 |
|---|---|---|
| `reward(action)` | `float` | 每步奖励，供 RL 训练优化 |
| `_check_success()` | `bool` | 判定 episode 是否成功完成 |

它们通常**配合使用**：`reward` 内部调用 `_check_success`，成功时给稀疏大奖。

---

## 6. 写奖励函数的"流水线模板"

把整个流程拆成 5 步，照着填空即可：

```python
def reward(self, action=None):
    reward = 0.0

    # ───── 步骤 1: 定义成功条件 ─────
    # 思考：任务"成功"的物理判定是什么？高度？位置？姿态？接触？
    if self._check_success():
        reward += 1.0                                  # 完成大奖
        return reward * self.reward_scale              # 归一化后返回

    # ───── 步骤 2: 提取关键状态 ─────
    # 问自己：策略要"靠近什么/避免什么"？
    #   - 想靠近物体 → 夹爪到物体的距离
    #   - 想物体到目标 → 物体到目标点的距离
    #   - 想保持稳定 → 速度/角速度大小
    #   - 想避免某事 → 接触/越界
    gripper = self.robots[0].gripper
    dist = self._gripper_to_target(
        gripper=gripper,
        target=self.cube.root_body,
        target_type="body",
        return_distance=True,
    )

    # ───── 步骤 3: 选奖励"形状函数" ─────
    # 距离类 → tanh / 1/(1+d) / exp(-d^2)
    # 接触类 → 二值 0/1
    # 方向类 → 点积 (cos θ)
    reaching = 1 - np.tanh(5.0 * dist)
    reward += 0.5 * reaching

    # ───── 步骤 4: 加子目标里程碑奖励 ─────
    if self._check_grasp(gripper=gripper, object_geoms=self.cube):
        reward += 0.25

    # ───── 步骤 5: 归一化 ─────
    if self.reward_scale is not None:
        reward *= self.reward_scale / 1.0    # 除数为各分量最大值之和
    return reward
```

---

## 7. 常见奖励类型"配方表"

这张表覆盖了 90% 的操作任务场景：

| 奖励类型 | 物理目标 | 代码配方 |
|---|---|---|
| **接近** | 让夹爪靠近物体 | `1 - np.tanh(k * self._gripper_to_target(..., return_distance=True))` |
| **物体到目标** | 让物体到指定位置 | `1 - np.tanh(k * np.linalg.norm(obj_pos - target_pos))` |
| **高度提升** | 把物体抬高 | `cube_z = self.sim.data.get_body_xpos(self.cube.root_body)[2]; reward = cube_z - table_z` |
| **抓取** | 抓住物体 | `0.25 if self._check_grasp(...) else 0.0` |
| **接触** | 任意接触 | `1.0 if self.check_contact(geom_a, geom_b) else 0.0` |
| **保持静止** | 稳定不抖动 | `-k * np.sum(self.sim.data.qvel ** 2)` |
| **动作平滑** | 控制能量 | `-k * np.sum(action ** 2)` |
| **朝向对齐** | 夹爪正对物体 | `np.dot(gripper_forward, dir_to_obj)`（点积） |
| **避免接触** | 不碰桌子 | `-1.0 if self.check_contact(robot, table) else 0.0` |

---

## 8. 形状函数怎么选

距离 `d` 的三种常用映射：

```
tanh(k*d):    1 → 0，平滑，常用，k 控制衰减
1/(1+d):      1 → 0，长尾，远距离也有一点信号
exp(-k*d²):   1 → 0，高斯型，距离近时梯度大
```

**经验法则**：

- 系数 `k` 大：只有很近时才有显著奖励（更"挑剔"）
- 系数 `k` 小：远距离也能获得部分奖励（更"宽容"）
- 调参范围：从 5-20 起步试，观察训练曲线

---

## 9. 奖励函数优化的深层原理

前面几章讲的是"**怎么写**"一个奖励函数。本章回答更本质的问题：**为什么这样写能让策略学到正确行为？什么样的奖励设计会让策略学废？**

这一章把奖励函数从"代码片段"提升为"**策略梯度的来源**"，并基于一个核心
观点展开：**奖励值的大小不重要，奖励对状态的梯度方向才决定策略学什么**。
理解这一点后，再看 SO101 Lift「机械臂抬起来就不动」这类症状，就能直接
从梯度量级反推根因，而不是靠经验试参。

> 本章内容配合 [`so101_reward_diagnostic.md`](./so101_reward_diagnostic.md)
> 的诊断工具与真实案例食用更佳——那里展示了如何用奖励分量分解把本章的
> 理论落地为可操作的排查流程。

### 9.1 奖励函数 = 策略梯度的来源

#### 9.1.1 策略梯度定理的回顾

RL 算法（PPO/SAC/DDPG 等）本质都在估计并最大化期望回报 `J(θ)`：

```
J(θ) = E_τ~π_θ [ Σ γ^t · r_t ]
```

策略梯度定理告诉我们：

```
∇J(θ) = E[ ∇_θ log π_θ(a|s) · Q^π(s,a) ]
       ≈ E[ ∇_θ log π_θ(a|s) · Σ_t γ^(t-τ) · r_τ ]   (REINFORCE / GAE)
```

也就是说，**每一步奖励 `r_t` 都通过 `Q`（优势函数）按折扣因子 `γ` 反向
贡献到策略参数的更新方向**。这意味着：

| 现象 | 后果 |
|---|---|
| 某分项每步都给一个固定正奖励 | 策略对**任何**状态都学到"这是个好状态"，相当于无信号 |
| 某分项梯度方向与任务目标不一致 | 策略会被引导到错误方向 |
| 某分项梯度远大于其他分项 | 其他分项被淹没，策略只优化主导分项（reward hacking 的本质） |

#### 9.1.2 关键洞见：奖励值高 ≠ 梯度好

一个常见的错误直觉是「奖励总值大 = 任务进展好」。考虑两个分项：

```python
# 分项 A：每步固定给 +0.5
r_A = +0.5   # 200 步累计 +100

# 分项 B：只有成功时给 +2.25
r_B = +2.25 if success else 0   # 失败时累计 0
```

A 的总值远大于 B，但 A **对策略梯度毫无贡献**（与状态无关，优势函数为 0）。
B 虽然稀疏，但只要策略能偶尔成功，B 的梯度就会指向"做更多能成功的动作"。

**推论**：评估一个奖励分项时，要问的不是"它贡献多少分"，而是
"**它对状态的梯度方向是否指向任务目标、量级是否足够**"。

### 9.2 形状函数的梯度分析

第 8 章给了三种形状函数的曲线，但**真正影响策略学习的是它们的梯度**。

#### 9.2.1 tanh 的梯度公式

设 `r(d) = w · (1 - tanh(k·d))`，则其对距离 `d` 的梯度为：

```
dr/dd = -w · k · sech²(k·d)
```

- `sech²(0) = 1`：在 `d=0` 处梯度最大，为 `-w·k`
- `sech²(x)` 随 `|x|` 增大指数衰减：`sech²(2) ≈ 0.07`，`sech²(3) ≈ 0.0099`
- 也就是说，**梯度集中在 `d ∈ [0, 2/k]` 这个窗口内**，远处几乎为 0

#### 9.2.2 不同 k 值的梯度对比

下表量化了「tanh 系数 k 如何决定梯度作用窗口」：

| k | 梯度有效窗口 (d < 2/k) | dist=0.1 处梯度 | dist=0.4 处梯度 | 适用场景 |
|---|---|---|---|---|
| 15 | 0.13m | -0.077·w | ≈0 | 极近程精修（策略已能接近） |
| 5 | 0.40m | -2.21·w | -0.18·w | 中程引导（默认值，常偏陡） |
| 2 | 1.00m | -3.84·w | -0.89·w | 远程引导（破"远距离悬浮"） |
| 1 | 2.00m | -2.94·w | -1.46·w | 全程稀疏拉力（梯度太平均） |

**调参直觉**：

- 想让策略**从远处开始下降** → 用小 k（如 2），让 dist=0.4 处仍有显著梯度
- 想让策略**接近后精修对位** → 用大 k（如 15），但**只在 dist 已小时启用**（否则远处无信号）
- 实践中常用**双尺度叠加**：`r = w₁·(1-tanh(k₁·d)) + w₂·(1-tanh(k₂·d))`，小 k 引导远→近，大 k 精修近→对

#### 9.2.3 SO101 真实案例的梯度反推

参考 [`so101_reward_diagnostic.md` §8](./so101_reward_diagnostic.md) 的案例，
策略稳定悬浮在 `eef_z=1.20`，cube 在 `z=0.82`，3D 距离 `dist≈0.4m`。

旧设计 `r_reach = 0.5·(1 - tanh(5.0·0.4))`：

```
r_reach/step = 0.5 · (1 - tanh(2.0)) = 0.5 · 0.036 = 0.018
梯度 dr/ddist = -0.5 · 5 · sech²(2.0) = -0.18
```

即「eef 每下降 1cm，reach 多拿 0.0018」——相比 z_float 每步 -0.13 的
噪声，**reach 的梯度信号被完全淹没**。SAC 看不到"下降能拿更多分"，自然
选择"悬浮不动"。

修复方向（已落地于 [`wrappers.py`](../../robosuite/demos/so101_realistic/wrappers.py)）：

- `k: 5.0 → 2.0`：dist=0.4 处梯度从 -0.18 → -0.89（5×）
- `w: 0.50 → 0.80`：峰值奖励从 0.018 → 0.27（15×）
- 净效果：每下降 1cm 的净增益从 0.0058 → 0.0139（2.4×）

### 9.3 多分量奖励的量级平衡

#### 9.3.1 单步贡献 vs 折扣累积贡献

奖励分项的"重要性"不能只看单步系数，而要看**折扣累积**：

```
单步贡献：     Δr = w · f(s)
折扣累积贡献： Σ_t γ^t · Δr_t ≈ Δr · 1/(1-γ)  （若分项稳定）
```

- `γ=0.99` 时累积因子 ≈ 100，一个 `w=0.005` 的分项单步看似很小，累积可达 0.5
- 但若该分项**只在特定状态才触发**（如 grasp 二值），则不能这样算

#### 9.3.2 正负信号配比的经验法则

把分项按梯度方向分两类：

- **正信号**（任务进度）：reach / grasp / lift，**驱动**策略向目标移动
- **负信号**（运动惩罚）：smooth / vel / ee_vel，**抑制**特定行为

经验法则（来自 SO101 调参）：

| 正:负 总量比 | 策略行为倾向 | 风险 |
|---|---|---|
| 5:1 或更高 | 激进接近，运动可能粗糙 | 损坏硬件（实机部署时） |
| **3:1**（推荐） | 平衡：敢动但不过度 | — |
| 1:1 | 谨慎，运动偏慢 | 探索效率低 |
| 1:3 或更低 | **完全不动最划算** | 陷入"静止局部最优"（SO101 旧版症状） |

#### 9.3.3 量级失衡的两种典型症状

**症状 A：负信号过强 → 静止局部最优**

```
reward_ee_vel ≈ -0.0005   （实测值，策略已完全不动）
reward_smooth  ≈ -1.35    （仍有动作惩罚残留）
reward_reach   ≈ +5.33    （200 步累计，每步仅 0.027）
```

策略发现"不动时 ee_vel/smooth 都为 0"，于是停在悬浮位置。这是
**惩罚函数最危险的失败模式**——策略不会崩溃，只是学到一个"安全但无
进展"的行为。

**症状 B：正信号虚高 → reward hacking**

```
reward_lift   ≈ +109    （cube 不动却持续得分）
original_reward ≈ 0     （任务从未成功）
```

策略学到"把臂举高"就持续拿 lift 分。**根因不是系数太大，而是基线选错**
（用桌面高度而非 cube 静止高度）。修复方法是修正基线，不是降低 w_lift。

### 9.4 奖励地形与局部最优

#### 9.4.1 奖励地形的概念

把奖励看作状态空间上的"地形"，策略梯度法是"小球在重力下滚向低能态"：

- **山头**（局部高奖励区）：策略会被吸引过去
- **山谷**（局部低奖励区）：策略会逃离
- **全局最优**：最高的山头 = 任务真正成功

reward hacking 的本质是**地形里有假山头**（奖励设计漏洞造成的高奖励区），
策略被困在假山头上而不去真正的顶峰。

#### 9.4.2 局部最优的形成机制

SO101「悬浮不动」就是一个教科书级的局部最优：

```
状态 s*：eef 悬浮在 z=1.20，dist=0.4
├── r_reach   = +0.018/step  ← 弱正信号，几乎无梯度
├── r_z_float = -0.13/step   ← 强负信号，但只罚不引导方向
├── r_ee_vel  = 0/step       ← 不动就没惩罚
├── r_smooth  = 0/step       ← 不动就没惩罚
└── r_vel     = 0/step       ← 不动就没惩罚
```

策略在 s* 处的"局部收益"是 -0.11/step。尝试下降会发生：

1. 立刻触发 ee_vel / smooth / vel 惩罚（短期负反馈）
2. reach 涨得很慢（梯度只有 0.18/cm）
3. 在 SAC 的时序差分下，"立刻的负反馈"压制了"远期 reach 收益"

**破局原理**（已在 SO101 修复中落地）：

- **增强正梯度**：降低 `reach_tanh_scale`，让下降能拿到显著 reach 增益
- **降低负梯度**：降低 `w_ee_vel` / `w_joint_vel` / `w_action_smooth`，让策略敢动
- **保留方向信号**：**不要降低** `w_z_float`，它是少数"明确指向下降方向"的负梯度
  来源，反而要略增，配合 reach 形成清晰信号

#### 9.4.3 为什么不能只加大负惩罚

一个常见误区是「策略不动 → 加大运动惩罚逼它动」。这是**反方向**的：

- 加大 `w_ee_vel` → 策略更不敢动 → 更稳定地停在 s*
- 加大 `w_z_float` → 惩罚更重但**梯度方向不变**，策略仍无下降动力

正确思路是：**惩罚只告诉策略"不要在哪"，引导要靠正信号的梯度**。
z_float 的负梯度虽然指向"下降"，但只在小阈值附近有效；远距离下降必须靠
reach 的正梯度驱动。

### 9.5 稀疏 vs 稠密的本质权衡

#### 9.5.1 探索难题的数学根源

稀疏奖励（只在成功时给）的困难在于**梯度估计方差极大**：

- 失败轨迹：Q = 0，梯度无信息
- 成功轨迹：Q = 2.25，但成功率 1/10000 时，10000 步才有一个有效梯度信号

稠密奖励（每步都给）通过**每步提供梯度**降低方差，但引入新风险：

- reward hacking（假山头）
- 量级失衡（局部最优）

#### 9.5.2 Potential-Based Reward Shaping（PBRS）理论

理论上保证「shaping 不改变最优策略」的形式是 **potential-based**：

```
F(s, s') = γ · Φ(s') - Φ(s)
```

其中 `Φ(s)` 是状态势函数。这样 `Σ F` 沿轨迹 telescopes 为
`γ^T Φ(s_T) - Φ(s_0)`，不影响最优策略。

**实践启示**：

- reach 奖励 `w·(1 - tanh(k·d))` 可视为 `Φ(s) = w·(1 - tanh(k·d))` 的近似
  PBRS——它只依赖状态 `s`，理论上较安全
- 二值 grasp 奖励**不是** potential-based，可能引入次优行为（如"持续抓
  住不动刷分"），需配合截断或仅在抓取后才解锁 lift 奖励

> 严格 PBRS 在机器人任务中很少直接用（难以设计满足 telescoping 的 Φ），
> 但其精神——「shaping 应是状态的势函数，而非行为奖励」——是判断设计
> 是否容易引发 hacking 的启发式准则。

#### 9.5.3 实战：差分 reach 在 SO101 上的破局效果

绝对形式 `r_reach = w·(1 - tanh(k·d))` 在 SO101 Lift 任务上被验证存在
"远距离静止高原"问题：策略即使 `k` 从 5→2→1 一路降低，仍学到悬浮在
z=1.22m 不动，因为悬浮时 `dist=0.4` 仍能拿 `r_reach ≈ 0.50/step`。

**改为 PBRS 差分形式**（势函数 `Φ(s) = -α·tanh(k·d)`）：

```
r_reach = α · (tanh(k·d_prev) - tanh(k·d_now))
```

关键性质：**`d_prev = d_now` 时 `r_reach = 0`**，从原理上消除"静止得高分"
的高原。参数取 `α=20, k=5`。

**同一已训练策略（仍悬浮）下的诊断对比**：

| 分量 | 绝对形式（旧） | PBRS 差分（新） | 说明 |
|---|---|---|---|
| `reward_reach` 累计 | +100.3 | -9.8 | 悬浮时接近 0，因策略先接近后远离故为负 |
| `reward_z_float` 累计 | -34.0 | -34.0 | 不变 |
| `shaped_reward` 累计 | **+65**（局部最优） | **-44.5**（局部负最优） | 悬浮从"诱人高原"变"明显陷阱" |

**梯度方向验证**：

- 悬浮时 `r_reach=0`，`z_float=-0.18/step` → 净 -0.18/step（必须移动）
- 接近 1cm：`r_reach = α·(tanh(5·0.05) - tanh(5·0.04)) = 20·0.048 = +0.96`
- 远离 1cm：`r_reach = -0.96`（对称负奖励）

**调参风险**：纯差分形式可能让策略学到"来回小幅震荡刷分"。SO101 中靠
`z_float` 持续负惩罚抑制——震荡时 `z_float` 仍罚，整体仍需趋势向下才能
转正。若无此类协同项，需追加"单调接近"约束或限制 `α` 大小。

### 9.6 诊断方法论：从症状反推根因

#### 9.6.1 奖励分量诊断速查表

结合 SO101 案例，给出按"症状 → 根因 → 处方"组织的诊断表：

| 症状（分量统计） | 根因（梯度视角） | 处方 |
|---|---|---|
| `reach≈0` 且 `original_reward=0` | tanh 系数 k 过大，远距离梯度饱和 | 降低 `reach_tanh_scale`（如 5→2） |
| `reach` 总分高但策略不接近 | w_reach 过大导致梯度噪声主导 | 降低 w_reach，检查 dist 是否真下降 |
| `lift` 高但 `original_reward=0` | 抬升基线用桌面高度（reward hacking） | 基线改用 cube 静止高度 `_cube_rest_z` |
| `grasp=0` 全程 | 从未接触 cube（可达性/初始姿态问题） | 检查 cube 放置范围、`init_qpos` |
| `ee_vel` 接近 0 且策略不动 | w_ee_vel 过大压制运动 | 大幅降低 w_ee_vel（如 0.10→0.03） |
| `z_float` 很负但策略仍悬浮 | 单纯惩罚无引导，需配合正梯度 | **不要降 w_z_float**；降低 reach_tanh_scale |
| `smooth` 极负且动作高频 | 探索噪声过大或 w_action_smooth 过小 | 适当提高 w_action_smooth |
| 总奖励稳定但 success_rate=0 | 局部最优 | 综合调参：增强正梯度+降低负梯度 |

#### 9.6.2 梯度计算辅助诊断

当症状不明确时，**直接计算症状点的梯度**比看累计值更有诊断力。SO101
案例的诊断脚本（可复用）：

```python
import numpy as np

def reach_grad(w, k, d):
    """r_reach = w*(1 - tanh(k*d)) 对 d 的梯度"""
    return -w * k * (1.0 / np.cosh(k * d)) ** 2

# 症状点：策略悬浮在 dist=0.4
dist = 0.4
print(f"旧设计 k=5, w=0.5: 梯度 = {reach_grad(0.5, 5.0, dist):.4f}/m")
print(f"新设计 k=2, w=0.8: 梯度 = {reach_grad(0.8, 2.0, dist):.4f}/m")
print(f"每下降 1cm 净增益（旧）: {0.01*abs(reach_grad(0.5,5.0,dist)) + 0.01*0.40:.5f}")
print(f"每下降 1cm 净增益（新）: {0.01*abs(reach_grad(0.8,2.0,dist)) + 0.01*0.50:.5f}")
```

输出对比后，若新设计在症状点的净增益显著高于旧设计，则修复方向正确。

#### 9.6.3 量级平衡校验流程

修改奖励系数后，建议按以下流程校验：

1. **跑诊断脚本** `diagnose_so101_reward.py`，记录各分量 200 步累计值
2. **计算正负比**：`Σ(正分量) : |Σ(负分量)|`，目标在 2:1 ~ 4:1 之间
3. **检查主导分量**：是否有单分量占总奖励绝对值 >50%？若有，可能是量级失衡
4. **梯度反推**：用 §9.6.2 的脚本计算症状点梯度，确认方向指向任务目标
5. **训练前 5 万步盯 RewardBreakdownCallback**：确认 `original_reward` 出现非零，
   而非某分项虚高

### 9.7 调参决策树

给定症状，按以下决策树找到要调的参数（结合 SO101 wrappers.py 实际系数）：

```
策略不动（ee_vel≈0, std 极小）
├── 是 → 检查 motion 惩罚总量
│   ├── motion 惩罚 > 任务进度 1/3 → 降低 w_ee_vel / w_joint_vel / w_action_smooth
│   └── motion 惩罚 < 任务进度 1/3 → 检查是否局部最优（见下）
│
策略悬浮不动（z_float 很负）
├── 是 → 计算 dist 处 reach 梯度
│   ├── |dr/ddist| < 0.5 → 降低 reach_tanh_scale（最关键）
│   ├── |dr/ddist| > 0.5 但仍不动 → 提高 w_reach
│   └── 已满足但不动 → 检查是否 z_float 阈值过严（误伤正常接近）
│
策略动但 reward hacking（某正分量虚高，original_reward=0）
├── 是 → 定位哪个正分量
│   ├── lift 虚高 → 基线改用 cube 静止高度
│   ├── grasp 持续刷分 → 加截断或仅在 grasp 后解锁 lift
│   └── reach 总分高但 dist 不降 → 检查 dist 计算是否正确
│
策略高频抖动（smooth 极负）
├── 是 → 适当提高 w_action_smooth，或检查 SAC entropy 系数
```

### 9.8 本章要点速记

- **奖励值不重要，梯度方向和量级才决定策略学什么**
- **tanh 系数 k 决定梯度作用窗口**：大 k 精修近程，小 k 引导远程
- **正负信号配比 3:1** 是经验上的安全区，失衡会引发静止局部最优或 hacking
- **负惩罚只告诉策略"别在哪"，方向引导必须靠正信号梯度**——加大负惩罚往往适得其反
- **诊断要从梯度反推**，而非只看累计奖励值
- **PBRS 精神**：shaping 应尽量是状态的势函数，避免行为奖励引发刷分

> 下一章 [§10 Lift 任务奖励函数逐行剖析](#10-lift-任务奖励函数逐行剖析)
> 给出官方 Lift 的具体实现，结合本章原理可对照阅读。

---

## 10. Lift 任务奖励函数逐行剖析

`Lift.reward()` 是入门的最佳样本。它的设计分三层：

### 10.1 稀疏奖励（默认 `reward_shaping=False`）

```python
if self._check_success():
    reward = 2.25
```

只有抬起立方体才给 2.25，其它步都是 0。优点：无 reward hacking；缺点：难探索。

### 10.2 稠密奖励（`reward_shaping=True`）

```python
elif self.reward_shaping:
    # 阶段1: 接近奖励
    dist = self._gripper_to_target(...)
    reaching_reward = 1 - np.tanh(10.0 * dist)
    reward += reaching_reward

    # 阶段2: 抓取奖励（二值）
    if self._check_grasp(gripper=..., object_geoms=self.cube):
        reward += 0.25
```

**设计哲学**：把长 horizon 任务分解成多个子目标，每个子目标提供局部梯度信号。

### 10.3 奖励归一化

```python
if self.reward_scale is not None:
    reward *= self.reward_scale / 2.25
```

让"最大回报 = `reward_scale`（默认 1.0）"，跨任务可比较。

### 10.4 完整代码与逐行注释

```python
def reward(self, action=None):
    reward = 0.0

    # ① 成功判定（来自 _check_success: cube 高度 > 桌面+4cm）
    if self._check_success():
        reward = 2.25              # 三个分量最大值之和

    elif self.reward_shaping:
        # ② 接近奖励：grip_site 到 cube body 的距离
        dist = self._gripper_to_target(
            gripper=self.robots[0].gripper,
            target=self.cube.root_body,
            target_type="body",
            return_distance=True,
        )
        reward += 1 - np.tanh(10.0 * dist)    # 映射到 [0, 1]

        # ③ 抓取奖励：左右指腹都接触 cube
        if self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube):
            reward += 0.25

    # ④ 归一化
    if self.reward_scale is not None:
        reward *= self.reward_scale / 2.25
    return reward
```

### 10.5 设计上的关键技巧

| 技巧 | 作用 |
|---|---|
| `elif`（而非 `if`） | 成功后不再叠加 shaping 奖励，避免双重计数 |
| `tanh` 函数 | 平滑可微、有界 [0,1]、符合直觉 |
| 系数 `10.0` | 控制衰减速度——越大越"挑剔" |
| 2.25 这个魔法数 | `1.0(reach) + 0.25(grasp) + 1.0(lift)` 的总和，保证最大回报可预测 |

---

## 11. 自定义奖励函数的三种方式

### 11.1 方式 A：继承现有任务覆盖 `reward`

在 lift.py 中复制一份并修改：

```python
from robosuite.environments.manipulation.lift import Lift
import numpy as np

class MyLift(Lift):
    def reward(self, action=None):
        # 你的奖励函数
        reward = 0.0
        if self._check_success():
            reward = 5.0
        else:
            # 自定义稠密奖励
            dist = self._gripper_to_target(
                gripper=self.robots[0].gripper,
                target=self.cube.root_body,
                target_type="body",
                return_distance=True,
            )
            reward += 1 - np.tanh(5.0 * dist)  # 注意系数改小，更宽容
        return reward
```

然后通过 `robosuite.make` 注册或直接实例化：

```python
env = MyLift(robots="Panda", reward_shaping=True)
```

### 11.2 方式 B：从头创建新任务

继承 `ManipulationEnv`，必须实现以下方法：

```python
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv

class MyTask(ManipulationEnv):
    def _load_model(self):
        super()._load_model()
        # 1. 加载桌面、物体、放置采样器等

    def _setup_observables(self):
        observables = super()._setup_observables()
        # 2. 追加任务特有观测
        return observables

    def _check_success(self):
        # 3. 定义成功条件
        return ...

    def reward(self, action=None):
        # 4. 定义奖励函数
        reward = 0.0
        if self._check_success():
            reward = 1.0
        return reward
```

### 11.3 方式 C：脚本层临时覆盖（不修改源码）

适合快速实验：

```python
import robosuite
import numpy as np

env = robosuite.make("Lift", robots="Panda", reward_shaping=True)

# 动态替换奖励函数
original_reward = env.reward
def my_reward(action=None):
    if env._check_success():
        return 10.0
    dist = env._gripper_to_target(
        gripper=env.robots[0].gripper,
        target=env.cube.root_body,
        target_type="body",
        return_distance=True,
    )
    return 1 - np.tanh(5.0 * dist)
env.reward = my_reward  # monkey patch
```

---

## 12. 调试技巧

### 12.1 打印关键状态

在 `reward` 内部加：

```python
print(f"dist={dist:.3f}, cube_z={cube_z:.3f}")
```

### 12.2 用 `visualize` 辅助

`env.visualize({"env": True, "grippers": True})` 会画出夹爪到目标的线段，是一种直观的调试手段，便于人眼判断策略是否在接近目标。

### 12.3 跑随机策略

观察 reward 曲线是否合理（应当随接近物体而上升）。

### 12.4 手动设置状态

用 `env.sim.data.set_joint_qpos(...)` 把物体放到指定位置，看 reward 是否符合预期。

---

## 13. 常见陷阱与解决方案

| 陷阱 | 现象 | 解决 |
|---|---|---|
| Reward hacking | 策略找到漏洞拿分但不解决任务 | 检查策略行为；用稀疏奖励兜底 |
| 奖励量级失衡 | 某个分量淹没其他分量 | 归一化各分量到相近量级 |
| 双重计数 | 成功时既给大奖又给稠密奖 | 用 `elif` 而非 `if` |
| 梯度消失 | 距离很大时 tanh 输出几乎为 0 | 调小衰减系数（如 10→2） |
| 不可微尖峰 | 二值奖励无梯度 | 配合连续奖励使用 |

---

## 14. 学习路径与推荐资源

### 14.1 按顺序学习

1. **MuJoCo 官方文档（MJCF/XML 建模）**
   先看 `body / geom / site / joint` 这 4 个标签，理解 XML 怎么描述世界
   - 官方教程：<https://mujoco.readthedocs.io/en/latest/computation/index.html>
   - MJCF 参考：<https://mujoco.readthedocs.io/en/latest/XMLreference.html>

2. **MuJoCo Python API**
   重点掌握 `mjData` 的字段：`body_xpos / qpos / qvel / sensordata`
   - <https://mujoco.readthedocs.io/en/latest/python.html>

3. **robosuite 源码阅读**
   按这个顺序读最能循序渐进：
   - `robosuite/environments/base.py`：仿真器封装、step 主循环
   - `robosuite/environments/manipulation/manipulation_env.py`：`_check_grasp` / `_gripper_to_target` 工具
   - `robosuite/environments/manipulation/lift.py`：最简任务范例
   - `robosuite/environments/manipulation/pick_place.py` / `nut_assembly.py`：看不同任务如何复用工具

4. **RL 入门理论**
   先理解 reward、return、discount、done 的关系，再看 reward shaping 的原理
   - 推荐 Spinning Up：<https://spinningup.openai.com/>（DRL 概念部分）

5. **动手实验**
   - 先复制 Lift，把 `tanh(10*d)` 改成 `tanh(2*d)`，看训练曲线变化
   - 再把 reaching 改成 `1/(1+d)`，对比效果
   - 然后试着加一个新的"朝向奖励"分量

### 14.2 奖励函数设计的实战建议

#### 设计流程（推荐顺序）

1. **先定义 `_check_success`**——明确任务成功的物理判定
2. **先跑稀疏奖励**——验证环境逻辑正确（用随机策略或脚本策略测试）
3. **再加稠密奖励**——如果稀疏学不动，再分层加 reaching / grasping 等
4. **归一化**——用 `reward_scale` 让最大回报可控
5. **调参**——观察训练曲线，调整各分量权重和衰减系数

---

## 15. 可立即运行的最小示例

把这段代码保存为脚本运行，先建立"我改一改就能影响 reward"的直观感觉：

```python
import numpy as np
import robosuite

env = robosuite.make("Lift", robots="Panda", reward_shaping=True, has_renderer=True)

# 看看你能取到哪些状态
obs = env.reset()
print("夹爪到立方体的距离:")
print(env._gripper_to_target(
    gripper=env.robots[0].gripper,
    target=env.cube.root_body,
    target_type="body",
    return_distance=True,
))
print("立方体当前高度:")
print(env.sim.data.get_body_xpos(env.cube.root_body)[2])
print("是否抓住:")
print(env._check_grasp(gripper=env.robots[0].gripper, object_geoms=env.cube))

# 跑一步随机动作，看 reward 变化
for _ in range(10):
    action = np.random.uniform(-1, 1, env.action_dim)
    obs, reward, done, info = env.step(action)
    print(f"reward={reward:.4f}, done={done}")
    if done:
        obs = env.reset()
```

---

## 16. 总结

你的"无从下手"主要来自两个缺口：**MuJoCo 数据结构**和**几何概念**。补齐后，
奖励函数 = ① 状态提取（`sim.data` / robosuite 封装）→ ② 套形状函数（`tanh` 等）
→ ③ 子目标加和 → ④ 归一化。

### 核心要点速记

- **三层基类** `MujocoEnv → RobotEnv → ManipulationEnv` 各管一段，
  最终把奖励和成功判定的"接口"留给具体任务实现
- **`reward()` 在 `step()` 末尾被自动调用**，它能访问仿真器所有状态
- **`reward()` 和 `_check_success()` 通常配合使用**：成功时给稀疏大奖，未成功时给稠密 shaping 奖励
- **`ManipulationEnv` 提供了 `_gripper_to_target` / `_check_grasp` 等工具方法**，写奖励时直接复用
- **自定义奖励**有三种方式：继承现有任务覆盖 `reward`、创建全新任务、脚本层 monkey patch

建议从把 `Lift` 的 `tanh` 系数改一改、加个动作惩罚项开始，5 分钟就能看到效果，
再循序渐进到复杂任务。

---

## 附录：相关源码文件索引

| 文件 | 作用 |
|---|---|
| [`robosuite/environments/base.py`](../../robosuite/environments/base.py) | `MujocoEnv` 基类：仿真器封装、step 主循环、`reward()` / `_check_success()` 抽象方法、`check_contact` / `get_contacts` 工具 |
| [`robosuite/environments/robot_env.py`](../../robosuite/environments/robot_env.py) | `RobotEnv`：机器人实例化、控制器、相机观测、`_pre_action` 动作分发 |
| [`robosuite/environments/manipulation/manipulation_env.py`](../../robosuite/environments/manipulation/manipulation_env.py) | `ManipulationEnv`：`_check_grasp` / `_gripper_to_target` / `_visualize_gripper_to_target` 等操作任务工具 |
| [`robosuite/environments/manipulation/lift.py`](../../robosuite/environments/manipulation/lift.py) | `Lift` 任务：最简单的操作任务范例，含详尽中文注释 |
| [`robosuite/environments/manipulation/pick_place.py`](../../robosuite/environments/manipulation/pick_place.py) | `PickPlace` 任务：多阶段任务范例 |
| [`robosuite/environments/manipulation/nut_assembly.py`](../../robosuite/environments/manipulation/nut_assembly.py) | `NutAssembly` 任务：高精度位姿匹配范例 |
| [`robosuite/environments/manipulation/door.py`](../../robosuite/environments/manipulation/door.py) | `Door` 任务：关节角度判定范例 |
