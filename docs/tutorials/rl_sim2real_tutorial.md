# robosuite 强化学习与 Sim2Real 实战教程

## 目录

1. [前言：为什么学习这个？](#前言)
2. [第一章：强化学习基础](#第一章强化学习基础)
3. [第二章：robosuite 环境入门](#第二章robosuite-环境入门)
4. [第三章：奖励设计与任务定义](#第三章奖励设计与任务定义)
5. [第四章：控制器选择对 RL 的影响](#第四章控制器选择对-rl-的影响)
6. [第五章：接入主流 RL 框架](#第五章接入主流-rl-框架)
7. [第六章：Sim2Real 核心思想](#第六章sim2real-核心思想)
8. [第七章：域随机化实战](#第七章域随机化实战)
9. [第八章：演示初始化 RL](#第八章演示初始化-rl)
10. [第九章：传感器建模与延迟](#第九章传感器建模与延迟)
11. [第十章：完整训练流程与调参经验](#第十章完整训练流程与调参经验)
12. [第十一章：常见问题与解决方案](#第十一章常见问题与解决方案)
13. [扩展阅读](#扩展阅读)

---

## 前言

本教程面向希望使用 **robosuite** 学习机器人强化学习（Reinforcement Learning, RL）和仿真到真实迁移（Sim2Real）的初学者和进阶学习者。

### 学完本教程你能做什么？

- 理解 RL 在机器人操作任务中的完整流程
- 用 robosuite 构建并训练一个机器人操作任务
- 设计奖励函数、选择控制器、连接 Gym API
- 掌握 Sim2Real 的核心技术：域随机化、传感器建模、演示初始化
- 理解真实机器人部署时可能遇到的关键问题

### 前置知识

- Python 编程基础
- 简单的线性代数与概率论
- 接触过 PyTorch 或 TensorFlow 更佳（非必需）

### 推荐环境

- Ubuntu / Linux
- Python 3.9+
- MuJoCo 3.1.1
- robosuite v1.5

---

## 第一章：强化学习基础

### 1.1 什么是强化学习？

强化学习研究的是 **智能体（Agent）如何通过与环境（Environment）交互来学习最优行为策略**。

```
          ┌──────────────┐
          │   智能体      │
          │  （策略网络）  │
          └──────┬───────┘
                 │ action
                 ▼
          ┌──────────────┐
          │    环境       │
          │  （机器人+任务）│
          └──────┬───────┘
                 │ obs, reward, done
                 ▼
          ┌──────────────┐
          │   智能体      │
          └──────────────┘
```

在每个时间步，智能体：

1. 观测当前状态 $s_t$
2. 根据策略选择动作 $a_t \sim \pi(a | s_t)$
3. 环境返回下一状态 $s_{t+1}$ 和奖励 $r_t$
4. 智能体根据 $(s_t, a_t, r_t, s_{t+1})$ 更新策略

目标是最大化**累积回报（Return）**：

$$
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k}
$$

其中 $\gamma \in [0, 1)$ 是折扣因子，表示未来奖励的重要性。

### 1.2 RL 在机器人中的特殊挑战

| 挑战 | 说明 | robosuite 应对方式 |
|------|------|-------------------|
| 高维连续动作空间 | 机器人关节通常是连续控制 | 支持多种控制器（OSC/IK/Joint） |
| 样本效率低 | 真实机器人试错成本高 | 仿真环境快速并行采集数据 |
| 奖励设计困难 | 任务成功标准难量化 | 提供稀疏/稠密奖励切换 |
| Sim2Real 差距 | 仿真与现实物理/视觉差异 | 域随机化、传感器建模 |
| 安全约束 | 真实机器人不能随意碰撞 | 先在仿真中充分验证 |

### 1.3 关键概念速查

| 概念 | 符号 | 含义 |
|------|------|------|
| 状态 / 观测 | $s$ / $o$ | 环境信息（机器人可能只看到部分） |
| 动作 | $a$ | 智能体输出的控制指令 |
| 奖励 | $r$ | 环境对单步动作的反馈 |
| 策略 | $\pi(a \| s)$ | 状态到动作的映射 |
| 状态价值 | $V(s)$ | 从 $s$ 出发能获得的期望回报 |
| 动作价值 | $Q(s, a)$ | 在 $s$ 执行 $a$ 后能获得的期望回报 |
| 折扣因子 | $\gamma$ | 未来奖励的折扣系数 |

### 1.4 常用 RL 算法

| 算法 | 类型 | 适用场景 |
|------|------|---------|
| SAC  | 离线策略 Actor-Critic | 连续控制，样本效率优先 |
| PPO  | 在线策略 Actor-Critic | 稳定训练，支持并行采集 |
| TD3  | 离线策略 Actor-Critic | 连续控制，避免 Q 值高估 |
| DQN  | 基于值函数 | 离散动作（robosuite 不常用） |
| BC / DAgger | 模仿学习 | 有人类演示时首选 |

---

## 第二章：robosuite 环境入门

### 2.1 robosuite 是什么？

robosuite 是一个基于 **MuJoCo** 物理引擎的机器人操作仿真框架，由斯坦福 SVL、UT RPL 和 NVIDIA GEAR 实验室共同维护。它为机器人学习研究提供：

- 标准化的操作任务环境（Lift、Stack、Door、PickPlace 等）
- 多种机器人模型（Panda、Sawyer、Baxter、GR1 等）
- 多种控制器（OSC、IK、JOINT_VELOCITY、JOINT_TORQUE）
- 完善的渲染、传感器、域随机化工具

### 2.2 最小可运行示例

从 `demo_random_action.py` 开始：

```python
import robosuite as suite
from robosuite.robots import MobileRobot
import numpy as np

# 创建环境
env = suite.make(
    "Lift",
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    ignore_done=True,
    use_camera_obs=False,
    control_freq=20,
)

# RL 核心循环
env.reset()
for i in range(1000):
    # 随机动作（真实训练时替换为策略网络输出）
    action = np.random.randn(*env.action_spec[0].shape)
    obs, reward, done, info = env.step(action)
    env.render()
```

这个脚本展示了 RL 的基本骨架：

1. `env.reset()` 初始化环境
2. `np.random.randn()` 选择动作（随机策略）
3. `env.step(action)` 执行动作并返回四元组
4. `env.render()` 可视化

### 2.3 环境创建参数解析

| 参数 | 典型值 | 说明 |
|------|--------|------|
| `env_name` | `"Lift"` | 任务名称 |
| `robots` | `"Panda"` | 机器人型号 |
| `has_renderer` | `True` / `False` | 是否开启可视化窗口 |
| `has_offscreen_renderer` | `False` | 是否开启离屏渲染（图像观测用） |
| `use_camera_obs` | `False` | 是否使用相机图像作为观测 |
| `control_freq` | `20` | 控制频率 Hz |
| `horizon` | `200` / `400` | 每个 episode 的最大步数 |
| `reward_shaping` | `True` / `False` | 是否使用稠密奖励 |
| `ignore_done` | `True` / `False` | 是否忽略 done 信号 |

**训练时推荐配置**：

```python
env = suite.make(
    "Lift",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=20,
    horizon=200,
    reward_shaping=True,
)
```

### 2.4 控制器与动作空间

控制器决定了 RL 的动作空间维度和含义。见 [demo_control.py](../demos/demo_control.py)。

| 控制器 | 动作维度（含夹爪） | 含义 | RL 难度 |
|--------|------------------|------|---------|
| `OSC_POSE` | 7 | 3 维位置增量 + 3 维旋转增量 + 1 夹爪 | ⭐⭐ 推荐 |
| `OSC_POSITION` | 4 | 3 维位置增量 + 1 夹爪 | ⭐⭐ 推荐 |
| `IK_POSE` | 7 | 逆运动学位姿控制 | ⭐⭐ |
| `JOINT_VELOCITY` | 关节数 + 1 | 关节速度 + 夹爪 | ⭐⭐⭐ |
| `JOINT_POSITION` | 关节数 + 1 | 关节位置 + 夹爪 | ⭐⭐⭐ |
| `JOINT_TORQUE` | 关节数 + 1 | 关节力矩 + 夹爪 | ⭐⭐⭐⭐ |

**新手建议**：从 `OSC_POSE` 或 `OSC_POSITION` 开始。它们的动作空间直接对应任务空间（笛卡尔坐标），策略更容易理解"向前、向上"等概念。

---

## 第三章：奖励设计与任务定义

### 3.1 奖励函数：RL 的指挥棒

奖励函数告诉智能体什么行为是好的。设计不好会导致：

- **探索困难**：稀疏奖励下智能体可能永远看不到成功
- **Reward Hacking**：智能体找到漏洞获取奖励但不完成任务

### 3.2 Lift 任务的奖励分解

见 [lift.py](../environments/manipulation/lift.py) 中的 `reward()` 函数。

Lift 任务采用**分层奖励设计**：

```
最终目标：Lifting（抬起立方体）
    ↓
中间目标：Grasping（抓住立方体）
    ↓
过程引导：Reaching（接近立方体）
```

#### 稀疏奖励模式

```python
if self._check_success():
    reward = 2.25
```

只有成功抬起时才给奖励。定义简单，但学习困难。

#### 稠密奖励模式（reward_shaping=True）

```python
# Reaching reward：距离越近奖励越大
dist = self._gripper_to_target(...)
reaching_reward = 1 - np.tanh(10.0 * dist)
reward += reaching_reward

# Grasping reward：抓住时给额外奖励
if self._check_grasp(...):
    reward += 0.25
```

| 分量 | 值范围 | 作用 |
|------|--------|------|
| `reaching_reward` | [0, 1] | 引导夹爪接近立方体 |
| `grasping_reward` | {0, 0.25} | 鼓励抓取行为 |
| `lifting_reward` | 2.25 | 最终成功奖励 |

### 3.3 终止条件设计

见 `_check_success()`：

```python
cube_height = self.sim.data.body_xpos[self.cube_body_id][2]
table_height = self.model.mujoco_arena.table_offset[2]
return cube_height > table_height + 0.04
```

判定逻辑：**立方体中心高度 > 桌面高度 + 4cm 余量**。

- 余量防止"刚贴着桌面"被误判成功
- 用仿真器精确位姿而非视觉检测，简单可靠

### 3.4 奖励设计经验

1. **入门先用稠密奖励**，熟练后再挑战稀疏奖励
2. **奖励分量要平衡**：避免某个分量过大导致行为偏向
3. **用 tanh 做距离塑形**：平滑、有界、梯度好
4. **归一化奖励量级**：默认 `reward_scale / 2.25` 将最大值缩到 1.0
5. **加入稀疏成功奖励**：确保最终目标是明确的

---

## 第四章：控制器选择对 RL 的影响

控制器不仅影响动作空间，还影响 RL 的学习效率。

### 4.1 操作空间控制（OSC）

`OSC_POSE` 控制器通过**阻抗控制**将笛卡尔位姿误差转换为关节力矩。

```
策略输出: (dx, dy, dz, droll, dpitch, dyaw, gripper)
    ↓
OSC 控制器: 计算目标位姿 → 逆雅可比 → 关节力矩
    ↓
MuJoCo: 执行仿真
```

**优点**：
- 动作直观："向前移动 1cm"直接对应笛卡尔运动
- 策略学得更快（探索效率高）
- 避免冗余关节协调问题

**缺点**：
- 需要精确的机器人模型
- 奇异点附近可能失效

### 4.2 关节空间控制

`JOINT_VELOCITY` 或 `JOINT_TORQUE` 直接控制每个关节。

**优点**：
- 更接近底层电机控制
- 可学习全身协调行为

**缺点**：
- 动作维度高
- 探索空间大，样本效率低
- 需要处理动力学耦合

### 4.3 选择建议

| 场景 | 推荐控制器 |
|------|-----------|
| 入门学习 / 单臂操作 | `OSC_POSE` |
| 只需要平移，不需要旋转 | `OSC_POSITION` |
| 双臂协调 / 人形机器人 | `JOINT_VELOCITY` |
| 研究底层动力学 / Sim2Real 精细控制 | `JOINT_TORQUE` |

---

## 第五章：接入主流 RL 框架

### 5.1 为什么需要 GymWrapper？

robosuite 原生返回 `OrderedDict` 观测，而 Stable-Baselines3、CleanRL 等库要求 Gym API：

- 返回 numpy 数组观测
- 明确定义 `observation_space` 和 `action_space`
- `step()` 返回 5 元组 `(obs, reward, terminated, truncated, info)`

### 5.2 使用 GymWrapper

```python
import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import SAC

# 创建 robosuite 环境
env = suite.make(
    "Lift",
    robots="Panda",
    has_renderer=False,
    use_camera_obs=False,
    control_freq=20,
    horizon=200,
    reward_shaping=True,
)

# 包装为 Gym 环境
env = GymWrapper(env)

# 用 SAC 训练
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# 保存并测试
model.save("lift_panda_sac")
```

### 5.3 观测空间说明

默认 `flatten_obs=True` 时，观测会被拼接成一维向量，包含：

- `object-state`：物体位姿
- `robot0_proprio-state`：机器人本体感受（关节角度、速度、夹爪状态等）

### 5.4 训练提速技巧

1. **关闭渲染**：`has_renderer=False`
2. **关闭图像渲染**：`has_offscreen_renderer=False`
3. **向量化环境**：

```python
from stable_baselines3.common.vec_env import DummyVecEnv

env = DummyVecEnv([lambda: GymWrapper(suite.make("Lift", robots="Panda", ...)) for _ in range(4)])
model = SAC("MlpPolicy", env, verbose=1)
```

4. **观测归一化**：

```python
from stable_baselines3.common.vec_env import VecNormalize

env = VecNormalize(env, norm_obs=True, norm_reward=True)
```

---

## 第六章：Sim2Real 核心思想

### 6.1 什么是 Sim2Real 差距？

仿真器是真实世界的简化模型，两者之间存在差距：

| 差距类型 | 仿真假设 | 真实情况 |
|---------|---------|---------|
| 视觉 | 完美纹理、固定光照 | 光照变化、阴影、相机噪声 |
| 动力学 | 精确物理参数 | 摩擦、质量、阻尼不确定 |
| 传感器 | 无噪声、无延迟 | 噪声、延迟、异步采样 |
| 执行器 | 理想电机 | 齿轮间隙、控制延迟、饱和 |
| 接触模型 | 简化软接触 | 复杂材料变形、粘滑 |

### 6.2 Sim2Real 的三大技术路线

```
┌─────────────────────────────────────────────────────────────┐
│                     Sim2Real 技术路线                         │
├─────────────────┬─────────────────┬─────────────────────────┤
│ 域随机化 (DR)    │ 系统辨识 (SysID) │ 域自适应 (DA)            │
│ Domain          │ System          │ Domain                  │
│ Randomization   │ Identification  │ Adaptation              │
├─────────────────┼─────────────────┼─────────────────────────┤
│ 训练时随机化仿真 │ 用真实数据校准   │ 用真实数据微调策略        │
│ 参数，扩大分布   │ 仿真参数         │ 或学习域无关特征          │
├─────────────────┼─────────────────┼─────────────────────────┤
│ 无需真实数据     │ 需要少量真实数据 │ 需要较多真实数据          │
│ 实现简单         │ 仿真更准确      │ 迁移效果最好但成本高       │
└─────────────────┴─────────────────┴─────────────────────────┘
```

robosuite 主要支持 **域随机化** 和 **传感器建模**，这也是本教程重点。

---

## 第七章：域随机化实战

### 7.1 什么是域随机化？

**核心思想**：训练时主动随机化仿真环境参数，让策略在"分布外"也能工作。

数学上，策略学习最大化：

$$
\max_\pi \mathbb{E}_{\xi \sim p(\xi)} \left[ J(\pi, \xi) \right]
$$

其中 $\xi$ 是随机化的环境参数。如果策略能应对各种"奇怪的"仿真环境，真实世界只是其中一个特例。

### 7.2 robosuite 中的域随机化

见 [domain_randomization_wrapper.py](../wrappers/domain_randomization_wrapper.py)。

```python
import robosuite as suite
from robosuite.wrappers import DomainRandomizationWrapper, GymWrapper

env = suite.make("Lift", robots="Panda", use_camera_obs=True, ...)

env = DomainRandomizationWrapper(
    env,
    randomize_color=True,      # 颜色和纹理随机化
    randomize_camera=True,     # 相机参数随机化
    randomize_lighting=True,   # 光照随机化
    randomize_dynamics=True,   # 动力学随机化
    randomize_on_reset=True,   # 每个 episode 开始时随机化
)

env = GymWrapper(env)
```

### 7.3 视觉随机化

| Modder | 随机化内容 | Sim2Real 作用 |
|--------|-----------|--------------|
| `TextureModder` | 颜色、纹理、材质 | 真实物体外观变化 |
| `CameraModder` | 相机位置、朝向、FOV | 相机安装误差、内参偏差 |
| `LightingModder` | 光源位置、方向、强度 | 真实光照条件变化 |

### 7.4 动力学随机化（最重要）

见 [mjmod.py](../utils/mjmod.py) 中的 `DynamicsModder`。

| 参数组 | 参数 | 现实含义 |
|--------|------|---------|
| **Opt** | `density`, `viscosity` | 空气阻力、介质特性 |
| **Body** | `mass`, `inertia` | 负载质量、惯性张量 |
| **Geom** | `friction`, `solref`, `solimp` | 接触摩擦、软接触模型 |
| **Joint** | `damping`, `frictionloss`, `armature` | 关节阻尼、静摩擦、电机惯量 |

### 7.5 实战练习

**练习 1：对 Lift 任务的动力学随机化**

```python
from robosuite.wrappers import DomainRandomizationWrapper

env = DomainRandomizationWrapper(
    env,
    randomize_color=False,
    randomize_camera=False,
    randomize_lighting=False,
    randomize_dynamics=True,
    dynamics_randomization_args={
        "randomize_friction": True,
        "friction_perturbation_ratio": 0.5,
        "randomize_mass": True,
        "mass_perturbation_ratio": 0.5,
    },
    randomize_on_reset=True,
)
```

训练并观察：
- 训练速度是否变慢？
- 最终成功率是否下降？
- 测试时在没有随机化的环境中是否仍有效？

**练习 2：视觉域随机化**

```python
env = DomainRandomizationWrapper(
    env,
    randomize_color=True,
    randomize_camera=True,
    randomize_lighting=True,
    randomize_dynamics=False,
)
```

对比：
- 使用图像观测训练
- 关闭视觉随机化 vs 开启视觉随机化
- 在固定纹理下测试泛化能力

### 7.6 域随机化的调参经验

1. **从小到大**：先小幅度随机化，观察训练是否稳定，再逐步增大
2. **分阶段启用**：先开动力学，再开视觉，最后全开
3. **监控训练曲线**：随机化过强会导致奖励无法上升
4. **保留评估环境**：用不随机化的环境评估真实性能
5. **真实数据校准**：如果有真实数据，可缩小随机化范围

---

## 第八章：演示初始化 RL

### 8.1 为什么需要演示？

纯 RL 从初始状态开始随机探索，对于精细操作任务：

- 初始状态离目标很远
- 随机动作几乎不可能成功
- 稀疏奖励下策略永远看不到成功信号

**演示初始化 RL** 用人类演示构造更好的初始状态分布。

### 8.2 核心思想

从演示轨迹中采样某个中间状态作为 episode 起点：

```
演示轨迹:  s0 --a0--> s1 --a1--> s2 --...--> sN (成功)

普通 RL:  总是从 s0 开始
演示初始化 RL:  可以从 s75、s50、s25 甚至 s0 开始
```

这是一种**课程学习（Curriculum Learning）**：先从简单状态（接近成功）开始，逐步回退到完整任务。

### 8.3 robosuite 中的 DemoSamplerWrapper

见 [demo_sampler_wrapper.py](../wrappers/demo_sampler_wrapper.py)。

```python
from robosuite.wrappers import DemoSamplerWrapper, GymWrapper

env = suite.make("Lift", robots="Panda", ...)

env = DemoSamplerWrapper(
    env,
    demo_path="path/to/demos",
    sampling_schemes=("reverse", "random"),
    scheme_ratios=(0.9, 0.1),
    open_loop_initial_window_width=25,
    open_loop_window_increment=25,
    open_loop_increment_freq=100,
)

env = GymWrapper(env)
```

### 8.4 采样方案对比

| 方案 | 采样范围 | 适用场景 |
|------|---------|---------|
| `random` | 环境默认初始状态 | 防止过拟合，保证泛化 |
| `uniform` | 演示任意状态 | 任务难度适中，演示质量参差 |
| `reverse` | 从末尾向开头扩展 | 结果导向任务（如 Lift、Place） |
| `forward` | 从开头向末尾扩展 | 过程导向任务（如轨迹跟踪） |

### 8.5 反向课程学习的过程

```
阶段 1: window=[N-25, N]   只从最后 25 步采样（最简单，接近成功）
阶段 2: window=[N-50, N]   从最后 50 步采样
阶段 3: window=[N-75, N]   从最后 75 步采样
阶段 4: window=[0, N]      从整个演示采样（完整难度）
```

### 8.6 收集演示

```bash
python robosuite/scripts/collect_human_demonstrations.py \
    --environment Lift \
    --robots Panda \
    --controller OSC_POSE \
    --episodes 50 \
    --hz 20
```

### 8.7 与模仿学习的结合

演示初始化 RL 常与模仿学习结合：

1. **Behavior Cloning (BC)**：直接用演示训练策略
2. **DAgger**：迭代收集新演示并训练
3. **RL fine-tune**：用 BC 初始化策略，再用 RL 优化

---

## 第九章：传感器建模与延迟

### 9.1 真实传感器的问题

仿真中传感器通常是**确定性的、无延迟的**。真实传感器则存在：

- **噪声**：位置、速度、力矩测量都有噪声
- **延迟**：从采样到控制器的传输延迟
- **异步采样**：不同传感器采样率不同
- **丢包/ dropout**：通信不可靠

### 9.2 robosuite 的 Observable API

见 [observables.py](../utils/observables.py)。

每个观测经过三段管线：

```
真值数据
    ↓
sensor()       # 采样
    ↓
corrupter()    # 添加噪声/损坏
    ↓
delayer()      # 添加延迟
    ↓
filter()       # 滤波
    ↓
最终观测
```

### 9.3 配置传感器损坏

```python
import numpy as np

# 给关节位置观测加高斯噪声
def gaussian_noise(value):
    return value + np.random.normal(0, 0.01, size=value.shape)

env.modify_observable(
    observable_name="robot0_joint_pos",
    attribute="corrupter",
    modifier=gaussian_noise,
)

# 设置观测延迟
env.modify_observable(
    observable_name="robot0_joint_pos",
    attribute="delayer",
    modifier=lambda value: (value, 0.02),  # 延迟 20ms
)

# 设置采样率
env.modify_observable(
    observable_name="robot0_joint_pos",
    attribute="sampling_rate",
    modifier=50,  # 50Hz
)
```

### 9.4 为什么要建模传感器问题？

1. **让策略对噪声鲁棒**：真实传感器一定有噪声
2. **测试延迟补偿能力**：大延迟下策略仍要稳定
3. **模拟多传感器融合**：不同模态采样率不同

---

## 第十章：完整训练流程与调参经验

### 10.1 推荐入门流程

```
步骤 1: 跑 demo_random_action.py，确认环境正常
步骤 2: 跑 demo_control.py，理解不同控制器行为
步骤 3: 用 GymWrapper + SAC 训练 Lift（reward_shaping=True）
步骤 4: 对比 reward_shaping=True vs False
步骤 5: 尝试 Door / Stack 等更难任务
步骤 6: 加入 DomainRandomizationWrapper
步骤 7: 加入 DemoSamplerWrapper
步骤 8: 迁移到真实机器人（如有条件）
```

### 10.2 推荐超参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| 控制器 | `OSC_POSE` | 任务空间控制，学习快 |
| `control_freq` | 20 Hz | 常用频率 |
| `horizon` | 200-400 | 根据任务复杂度调整 |
| `reward_shaping` | `True`（初学） | 提供稠密信号 |
| SAC `learning_rate` | 3e-4 | 标准值 |
| SAC `buffer_size` | 1e6 | 操作任务常用 |
| SAC `batch_size` | 256 | 标准值 |
| SAC `tau` | 0.005 | 软更新系数 |
| SAC `gamma` | 0.99 | 折扣因子 |

### 10.3 调试检查清单

| 现象 | 可能原因 | 解决方案 |
|------|---------|---------|
| 奖励不上升 | 奖励太稀疏 / 探索不够 | 开 reward_shaping，增大熵系数 |
| 训练不稳定 | 奖励量级过大 / 观测未归一化 | 归一化观测，缩放奖励 |
| 成功率低但奖励高 | Reward hacking | 检查奖励设计，减少不必要分量 |
| 仿真中成功率高，真实失败 | Sim2Real 差距大 | 加域随机化，加传感器噪声 |
| 训练极慢 | 渲染未关 / 图像观测 | 关渲染，用低维状态 |

### 10.4 学习曲线解读

```
奖励曲线
  │
  │       理想：快速上升，最终稳定在高值
  │      /
  │     /
  │    /
  │___/
  └────────────────────── 时间步

  │
  │       过拟合：训练奖励高，测试奖励低
  │      /\___
  │     /      \
  │____/        \___
  └────────────────────── 时间步

  │
  │       学习失败：奖励几乎不上升
  │_________________
  └────────────────────── 时间步
```

---

## 第十一章：常见问题与解决方案

### Q1: 为什么策略在仿真中表现好，到真实机器人就不行？

**原因**：Sim2Real 差距。

**解决方案**：
- 增加动力学随机化
- 增加视觉随机化（若用图像观测）
- 建模传感器噪声和延迟
- 在真实机器人上做少量 fine-tune
- 使用更精确的机器人模型（系统辨识）

### Q2: 奖励曲线上升但成功率不上升？

**原因**：可能是 reward hacking 或奖励设计不合理。

**解决方案**：
- 检查策略实际行为
- 减少不必要的过程奖励
- 增加稀疏成功奖励权重
- 用视频记录 episode 分析

### Q3: 训练特别慢怎么办？

**排查**：
1. 是否开启了渲染？`has_renderer=False`
2. 是否使用了图像观测？改为低维状态
3. 是否单线程？使用向量化环境
4. 控制器是否太复杂？尝试 `OSC_POSITION`

### Q4: 稀疏奖励能学出来吗？

**可以，但需要技巧**：
- 使用 HER（Hindsight Experience Replay）
- 使用演示初始化 RL
- 使用好奇心驱动探索（intrinsic motivation）
- 缩短 horizon

### Q5: 该用 SAC 还是 PPO？

| 情况 | 推荐 |
|------|------|
| 样本效率优先 | SAC |
| 稳定性优先 / 容易调参 | PPO |
| 需要并行环境 | PPO |
| 连续精细控制 | SAC |

---

## 扩展阅读

### 论文

1. **robosuite 框架论文**：Zhu et al., "robosuite: A Modular Simulation Framework and Benchmark for Robot Learning", arXiv:2009.12293
2. **SAC**：Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor", ICML 2018
3. **PPO**：Schulman et al., "Proximal Policy Optimization Algorithms", arXiv:1707.06347
4. **Domain Randomization**：Tobin et al., "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World", IROS 2017
5. **Dactyl / Reverse Curriculum**：OpenAI, "Learning Dexterous In-Hand Manipulation", IJRR 2019
6. **HER**：Andrychowicz et al., "Hindsight Experience Replay", NeurIPS 2017

### 相关项目

- [robomimic](https://arise-initiative.github.io/robomimic-web/)：模仿学习框架
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)：主流 RL 库
- [CleanRL](https://docs.cleanrl.dev/)：简洁的 RL 实现
- [MuJoCo 文档](http://www.mujoco.org/book/)：物理引擎参考

### robosuite 官方资源

- [官方文档](https://robosuite.ai/docs/overview.html)
- [Demos](../demos/)
- [Sim2Real 文档](../algorithms/sim2real.md)

---

## 结语

强化学习和 Sim2Real 是机器人学习中最具挑战性也最有价值的方向之一。robosuite 提供了一个完整的平台，让你可以从仿真开始，逐步理解：

- 如何设计奖励函数
- 如何选择控制器
- 如何接入 RL 算法
- 如何缩小仿真与现实的差距
- 如何利用人类演示加速学习

**最重要的建议**：不要只读代码和文档，一定要动手跑实验。RL 的直觉只能在调参、观察训练曲线和分析策略行为中建立。

祝你学习愉快！
