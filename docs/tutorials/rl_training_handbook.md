# robosuite RL 训练实战手册

> 配合 [`rl_sim2real_tutorial.md`](./rl_sim2real_tutorial.md) 阅读的理论篇，本文档聚焦"动手跑通训练"的工程细节。
> 对应训练脚本：[`train_rl_sb3.py`](../../robosuite/demos/train_rl_sb3.py)

---

## 目录

1. [环境准备](#一环境准备)
2. [训练脚本结构详解](#二训练脚本结构详解)
3. [超参数理解与调参](#三超参数理解与调参)
4. [训练时间估算](#四训练时间估算)
5. [TensorBoard 可视化](#五tensorboard-可视化)
6. [常见错误与解决方案](#六常见错误与解决方案)
7. [性能优化清单](#七性能优化清单)

---

## 一、环境准备

### 1.1 conda 环境

本项目使用独立的 conda 环境 `robosuite`，所有操作必须指定该环境：

```bash
# 安装依赖
conda run -n robosuite pip install <package>

# 运行脚本
conda run -n robosuite python <script>.py
```

**不要用 `base` 环境**，否则会因环境隔离导致 `ModuleNotFoundError`。

### 1.2 所需依赖

| 包 | 用途 | 安装方式 |
|----|------|---------|
| `robosuite` | 仿真框架 | `pip install -e .`（项目根目录） |
| `stable-baselines3` | RL 算法库 | `pip install stable-baselines3` |
| `torch`（CUDA 版） | 神经网络后端 | 随 SB3 自动安装 |
| `tensorboard` | 训练曲线可视化 | `pip install tensorboard` |
| `gymnasium` | Gym API 标准 | 随 SB3 自动安装 |

或直接用项目提供的：

```bash
conda run -n robosuite pip install -r requirements-extra.txt
```

### 1.3 硬件参考

本手册以 **4060 Ti 16GB** 为基准：

- GPU 显存：16GB（Lift 低维任务用不到 1GB，绰绰有余）
- 瓶颈：**CPU 仿真**（MuJoCo 单线程），不是 GPU
- 核心结论：低维状态 RL 的 GPU 利用率低是正常现象

---

## 二、训练脚本结构详解

[`train_rl_sb3.py`](../../robosuite/demos/train_rl_sb3.py) 分为 7 个步骤：

```
第一步：定义环境工厂 make_env()
    ↓
第二步：创建向量化环境 SubprocVecEnv（8 并行）
    ↓
第三步：观测与奖励归一化 VecNormalize
    ↓
第四步：创建 SAC 算法实例
    ↓
第五步：训练 50 万步
    ↓
第六步：保存模型 + 归一化参数
    ↓
第七步：测试策略
```

### 2.1 为什么需要环境工厂函数？

```python
def make_env():
    env = suite.make("Lift", robots="Panda", ...)
    env = GymWrapper(env)
    return Monitor(env)
```

向量化环境需要 **N 个独立的环境实例**，每个实例必须独立创建。`SubprocVecEnv` 接收一个函数列表，每个函数返回一个环境实例。

### 2.2 为什么需要 GymWrapper？

robosuite 原生返回 `OrderedDict` 观测和 4 元组 `step()`，而 SB3 需要：

| 项目 | robosuite 原生 | Gym API（SB3 要求） |
|------|---------------|-------------------|
| `obs` | `OrderedDict` | `np.array`（一维） |
| `step()` 返回 | `(obs, reward, done, info)` | `(obs, reward, terminated, truncated, info)` |
| `observation_space` | 无显式定义 | `gym.spaces.Box` |
| `action_space` | `env.action_spec` | `gym.spaces.Box` |

`GymWrapper` 自动完成这个转换。

### 2.3 为什么需要 Monitor？

**没有 `Monitor`，TensorBoard 中就不会出现 `rollout/ep_rew_mean`（奖励曲线）。**

`Monitor` 在每个 episode 结束时，把该 episode 的总奖励和步数写入 `info["episode"]`。SB3 据此计算滑动平均值，记录到 TensorBoard 的 `rollout/ep_rew_mean` 和 `rollout/ep_len_mean`。

包装顺序必须是：`suite.make → GymWrapper → Monitor`（Monitor 在最外层）。

### 2.4 为什么需要 `if __name__ == "__main__":`？

`SubprocVecEnv` 的子进程会重新 import 主模块来加载 `make_env` 函数。如果顶层有 `env = SubprocVecEnv(...)`，子进程 import 时会再次执行这行，导致无限递归创建子进程。

**正确做法**：把所有训练逻辑放进 `main()` 函数，用 `if __name__ == "__main__":` 保护。

---

## 三、超参数理解与调参

### 3.1 环境参数

| 参数 | 值 | 作用 | 调参建议 |
|------|----|----|---------|
| `control_freq` | 20 | 控制频率 Hz，每步 50ms | 高=精细但样本多，低=粗糙 |
| `horizon` | 200 | 每 episode 最大步数 | 200 步 = 10 秒模拟时间 |
| `reward_shaping` | True | 稠密奖励 | 新手必开，关闭后几乎学不会 |
| `use_camera_obs` | False | 用低维状态而非图像 | 新手必关，图像 RL 慢 10-100 倍 |

### 3.2 SAC 超参数

| 参数 | 值 | 含义 | 调参建议 |
|------|----|----|---------|
| `learning_rate` | 3e-4 | 网络学习率 | 太大（1e-3）学崩，太小（1e-5）极慢。3e-4 是 SAC 黄金值，一般不动 |
| `buffer_size` | 200000 | 经验回放池大小 | 存历史 (s,a,r,s')。太小忘旧经验，太大占内存 |
| `batch_size` | 256 | 每次更新采样数 | 越大越稳定但越慢。256 是标配 |
| `tau` | 0.005 | 目标网络软更新系数 | 越大更新越快但不稳定，越小越稳定但慢 |
| `gamma` | 0.99 | 折扣因子 | 0.99=重视长期，0.9=只关心近期 |
| `net_arch` | [256, 256] | 网络结构 | 简单任务 [128,128]，复杂任务 [400,300] |

### 3.3 网络结构选择

```python
policy_kwargs=dict(net_arch=[256, 256])
```

| 任务难度 | 推荐结构 | 说明 |
|---------|---------|------|
| 简单（Lift） | [128, 128] 或 [256, 256] | 小任务用小网络，加速训练 |
| 中等（Stack、PickPlace） | [256, 256] | 默认配置 |
| 复杂（Assembly、Tool Use） | [400, 300] | 需要更强表达能力 |

---

## 四、训练时间估算

### 4.1 计算公式

```
预计剩余时间（秒） = (总目标步数 - 已完成步数) / fps
```

### 4.2 实例

从终端日志读取：

```
fps = 131
total_timesteps = 36800
目标 = 500000

剩余步数 = 500000 - 36800 = 463200
剩余时间 = 463200 / 131 ≈ 3536 秒 ≈ 59 分钟
```

### 4.3 不同配置的速度参考

| 配置 | 典型 fps | 50 万步耗时 |
|------|---------|------------|
| 单环境 | 30-60 | 2-5 小时 |
| 8 环境 DummyVecEnv | 100-200 | 40-80 分钟 |
| 8 环境 SubprocVecEnv | 300-600 | 15-30 分钟 |

### 4.4 瓶颈分析

```
RL 训练时间 = CPU 仿真（70-80%）+ GPU 计算（20-30%）
```

对于低维状态的 robosuite：
- **MuJoCo 物理仿真占 70-80% 时间**，单线程 CPU
- 网络更新占 20-30%，GPU 轻松搞定
- GPU 利用率低是**正常现象**，因为网络太小（256×256 MLP）

---

## 五、TensorBoard 可视化

### 5.1 启用方式

训练脚本中已配置：

```python
model = SAC(..., tensorboard_log="./logs/sac_lift/")
```

### 5.2 启动 TensorBoard

训练运行后，另开一个终端：

```bash
conda activate robosuite
tensorboard --logdir=./logs/sac_lift/ --port=6006
```

浏览器打开 `http://localhost:6006`。

### 5.3 关键曲线

| 曲线 | 含义 | 健康表现 |
|------|------|---------|
| `rollout/ep_rew_mean` | **平均回合奖励（最重要）** | 持续上升 |
| `rollout/ep_len_mean` | 平均回合长度 | 下降可能意味着提前成功 |
| `train/ent_coef` | 熵系数（探索强度） | 自动调节，趋于稳定 |
| `train/actor_loss` | 策略损失 | 负值正常 |
| `train/critic_loss` | 价值损失 | 趋于稳定 |
| `train/learning_rate` | 学习率 | 固定不变 |

### 5.4 Lift 任务奖励参考

| ep_rew_mean | 含义 |
|------------|------|
| 0 - 0.5 | 还没学会接近物体 |
| 0.5 - 1.5 | 会接近，偶尔抓到 |
| 1.5 - 2.0 | 基本学会抓取 |
| **2.0 - 2.25** | 稳定抬起，任务成功 |

### 5.5 异常信号

| 现象 | 可能原因 | 解决方案 |
|------|---------|---------|
| 曲线长期平的 | 步数不够 / 种子差 | 跑更多步数 |
| 曲线上升后突然崩 | 学习率太大 / 奖励设计问题 | 降低学习率 |
| 曲线剧烈震荡 | batch 太小 / 学习率太大 | 增大 batch_size 或降低学习率 |

---

## 六、常见错误与解决方案

### 6.1 ModuleNotFoundError: No module named 'stable_baselines3'

**原因**：装错 conda 环境了（装到了 base，运行时用的是 robosuite）。

**解决**：

```bash
conda run -n robosuite pip install stable-baselines3
```

### 6.2 RuntimeError: An attempt has been made to start a new process...

**原因**：`SubprocVecEnv` 的子进程会重新 import 主模块，顶层代码没放在 `if __name__ == "__main__":` 保护块中，导致无限递归。

**解决**：把环境创建和训练逻辑包进 `main()` 函数：

```python
def main():
    env = SubprocVecEnv([make_env for _ in range(8)])
    # ... 训练逻辑 ...

if __name__ == "__main__":
    main()
```

### 6.3 GPU 利用率低

**原因**：低维状态输入下网络太小（256×256 MLP），计算量极小，GPU 大部分时间在等 CPU 仿真。

**结论**：这是正常现象，无法通过配置解决。瓶颈在 CPU 仿真，不在 GPU。

### 6.4 训练超慢（fps < 50）

**原因**：没用向量化环境，或用了 `DummyVecEnv`（单进程）。

**解决**：换 `SubprocVecEnv`（多进程），见 [`train_rl_sb3.py`](../../robosuite/demos/train_rl_sb3.py)。

### 6.5 测试时效果差

**原因**：没加载 `VecNormalize` 参数，策略"看不懂"未归一化的观测。

**解决**：测试时必须加载归一化参数：

```python
env = VecNormalize.load("vec_normalize.pkl", env)
env.training = False
env.norm_reward = False
```

### 6.6 TensorBoard 看不到 ep_rew_mean 曲线

**原因**：缺少 `Monitor` 包装器。SB3 通过 `Monitor` 追踪每个 episode 的结束和总奖励，没有它就无法计算 `ep_rew_mean`。

**解决**：在 `make_env()` 中 `GymWrapper` 之上包一层 `Monitor`：

```python
from stable_baselines3.common.monitor import Monitor

def make_env():
    env = suite.make("Lift", robots="Panda", ...)
    env = GymWrapper(env)
    env = Monitor(env)      # 关键！没有它 TensorBoard 不会显示 ep_rew_mean
    return env
```

---

## 七、性能优化清单

### 7.1 优化优先级

| 优先级 | 优化项 | 效果 | 难度 |
|--------|-------|------|------|
| 1 | `SubprocVecEnv` 替代单环境 | fps 提升 3-10 倍 | 低 |
| 2 | 关闭所有渲染 | fps 提升 2-3 倍 | 低 |
| 3 | `VecNormalize` 归一化 | 收敛速度提升，稳定性提升 | 低 |
| 4 | 增加训练步数 | 效果直接提升 | 低 |
| 5 | 缩小网络结构 | 单步更新加速 | 中 |
| 6 | 图像观测 + CnnPolicy | 能解决视觉任务，但慢 10-100 倍 | 高 |

### 7.2 推荐配置（4060 Ti 16GB）

```python
# 环境
NUM_ENVS = 8
control_freq = 20
horizon = 200
reward_shaping = True
use_camera_obs = False

# SAC
learning_rate = 3e-4
buffer_size = 200000
batch_size = 256
net_arch = [256, 256]
device = "cuda"
total_timesteps = 500000
```

### 7.3 向量化环境对比

| 类型 | 原理 | 优点 | 缺点 | 适用场景 |
|------|------|------|------|---------|
| `DummyVecEnv` | 单进程内循环 | 简单稳定，易调试 | 没真正多核，fps 低 | 调试、验证 |
| `SubprocVecEnv` | 多进程并行 | 真正多核，fps 高 | 需 `if __name__` 保护，IPC 开销 | 正式训练 |

---

## 八、可视化评估

训练完成后，用 [`evaluate_rl.py`](../../robosuite/demos/evaluate_rl.py) 在可视化窗口中观看机器人执行任务。

### 8.1 运行方式

```bash
conda run -n robosuite python robosuite/demos/evaluate_rl.py
```

会弹出 MuJoCo 可视化窗口，实时渲染机器人抓取立方体的过程。

### 8.2 评估脚本做了什么

1. 创建带渲染窗口的环境（`has_renderer=True`）
2. 加载训练时的 `VecNormalize` 参数（必须加载，否则策略看不懂观测）
3. 加载训练好的 SAC 模型
4. 运行 5 个 episode，每个 episode 实时渲染
5. 统计平均奖励、成功率

### 8.3 关键注意事项

| 项目 | 说明 |
|------|------|
| **环境参数** | 除 `has_renderer=True` 外，必须与训练时完全一致 |
| **归一化参数** | 必须用 `VecNormalize.load()` 加载，否则效果极差 |
| **测试模式** | `env.training = False`，`env.norm_reward = False` |
| **确定性推理** | `model.predict(obs, deterministic=True)`，用策略均值而非采样 |

### 8.4 成功率判断

Lift 任务奖励 >= 2.0 通常意味着成功抬起立方体：

| 奖励范围 | 含义 |
|---------|------|
| 0 - 0.5 | 没学会接近物体 |
| 0.5 - 1.5 | 会接近，偶尔抓到 |
| 1.5 - 2.0 | 基本学会抓取 |
| 2.0 - 2.25 | 稳定抬起，任务成功 |

---

## 附录：完整运行流程

```bash
# 1. 激活环境
conda activate robosuite

# 2. 启动训练
python robosuite/demos/train_rl_sb3.py

# 3. 另开终端启动 TensorBoard
conda run -n robosuite tensorboard --logdir=./logs/sac_lift/ --port=6006

# 4. 浏览器打开
# http://localhost:6006

# 5. 训练完成后，可视化评估
conda run -n robosuite python robosuite/demos/evaluate_rl.py
```
