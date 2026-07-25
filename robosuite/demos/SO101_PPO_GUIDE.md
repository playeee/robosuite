# SO101 Lift + PPO 算法实战指南

本指南是 [`SO101_RL_ALGORITHM_GUIDE.md`](./SO101_RL_ALGORITHM_GUIDE.md) 的 PPO 专题扩展，专注于把 [`train_rl_sb3_so101_realistic.py`](./train_rl_sb3_so101_realistic.py) 从 SAC 改造成 PPO 的完整流程。

阅读本指南前，建议先读完通用指南的第 1~3 节，了解算法分类与通用注意事项。

---

## 1. 为什么选 PPO？

PPO（Proximal Policy Optimization）是 on-policy 算法，与当前默认的 SAC（off-policy）有本质差异。选 PPO 的理由：

| 优势 | 劣势 |
|------|------|
| 训练过程更稳定，不易突然崩溃 | 样本效率低，需要更多训练步数 |
| 超参数鲁棒，工业界默认算法 | 依赖并行环境数（`n_envs`） |
| 不需要 replay buffer，内存占用低 | 探索能力依赖 `ent_coef`，不如 SAC 的最大熵 |
| 策略更新可解释性强（clip 机制） | 每次更新后丢弃数据，浪费样本 |

**SO101 Lift 任务适用性**：
- ✓ 动作连续，PPO 支持
- ✓ horizon=200 较短，on-policy 也能跑得动
- ✓ `realistic_state` 模式有 shaping 奖励，缓解了 PPO 在纯稀疏任务上的劣势
- ⚠ 500K 步可能不够，建议 1M~3M
- ⚠ 8 个环境偏少，建议 16

---

## 2. PPO 与 SAC 的核心差异

理解这些差异是改造的前提：

### 2.1 数据使用方式

```
SAC（off-policy）：
  env → replay buffer（200K）→ 随机采样 batch（256）→ 更新策略 → 保留 buffer

PPO（on-policy）：
  env → 收集 n_steps × n_envs 个样本 → 更新策略 n_epochs 轮 → 丢弃全部样本
```

**关键含义**：
- SAC 一个样本可以被多次学习；PPO 一个样本只在一个 update 里被学 `n_epochs` 轮，然后丢弃。
- PPO 必须保证 `n_steps × n_envs` 足够大，否则每次更新的样本太少，策略会震荡。

### 2.2 探索机制

| 机制 | SAC | PPO |
|------|-----|-----|
| 探索来源 | 最大熵（内置，自动） | 策略分布的熵（`ent_coef` 控制） |
| 探索强度 | 训练后期自动衰减 | 需要手动调 `ent_coef` |
| 状态依赖 | 是（state-dependent） | 否（高斯策略） |

**实务影响**：稀疏奖励任务下，PPO 的 `ent_coef` 不能设 0，否则策略早早收敛到“不动最安全”。

### 2.3 网络结构

- SAC：actor 和 critic 共享 `net_arch=[256, 256]`
- PPO：SB3 ≥ 1.6 推荐 **分离结构** `net_arch=dict(pi=[256, 256], vf=[256, 256])`

旧的列表式 `net_arch=[256, 256]` 在 PPO 里仍兼容，但会触发 deprecation warning。

---

## 3. 完整改造代码

下面是完整的 PPO 改造代码片段，按修改位置组织。建议直接复制对应片段替换原脚本。

### 3.1 顶部 import

```python
# 原代码：
# from stable_baselines3 import SAC
# 改为：
from stable_baselines3 import PPO
```

### 3.2 配置开关

```python
# 原代码：
# MODEL_PATH = "lift_so101_sac_realistic.zip"
# VEC_NORMALIZE_PATH = "vec_normalize_so101.pkl"
# 改为：
MODEL_PATH = "lift_so101_ppo_realistic.zip"
VEC_NORMALIZE_PATH = "vec_normalize_so101_ppo.pkl"

# 原代码：
# NUM_ENVS = 8
# TOTAL_TIMESTEPS = 500_000
# 改为：
NUM_ENVS = 16                  # PPO 受益于更多并行环境
TOTAL_TIMESTEPS = 2_000_000    # PPO 样本效率低，建议 1M~3M
```

### 3.3 超参数与模型构造

把原脚本的整段 `sac_kwargs` + `SAC.load/SAC("MlpPolicy", ...)` 替换为：

```python
# =============================================================================
# PPO 超参数
# =============================================================================
# 与 SAC 的关键区别：
#   - 没有 buffer_size、tau、use_sde（这些是 off-policy / SAC 专属）
#   - 新增 n_steps、n_epochs、gae_lambda、clip_range、ent_coef、vf_coef、max_grad_norm
#   - net_arch 用 dict 分离 actor / critic
#
# 整除校验：
#   n_steps * n_envs = 2048 * 16 = 32768
#   batch_size = 64，32768 / 64 = 512 ✓
# =============================================================================
ppo_kwargs = dict(
    verbose=0,
    tensorboard_log=LOG_DIR,
    learning_rate=3e-4,
    n_steps=2048,           # 每个 env 收集 2048 步后做一次策略更新
    batch_size=64,          # 从 n_steps*n_envs 中随机切片，必须能整除
    n_epochs=10,            # 每次 rollout 重复训练 10 轮
    gamma=0.99,
    gae_lambda=0.95,        # GAE 优势函数的 λ，0.95 是经典值
    clip_range=0.2,         # PPO clip 范围，0.2 是原论文值，不要轻易改
    ent_coef=0.01,          # 熵正则系数，稀疏奖励任务建议 0.01~0.02
    vf_coef=0.5,            # value function loss 权重
    max_grad_norm=0.5,      # 梯度裁剪，防止训练发散
    policy_kwargs=dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),  # actor/critic 分离
    ),
    device="auto",
)

if RESUME and os.path.exists(MODEL_PATH):
    print(f"加载已有模型：{MODEL_PATH}")
    model = PPO.load(MODEL_PATH, env=env, **ppo_kwargs)
else:
    if RESUME:
        print(f"[警告] 未找到模型 {MODEL_PATH}，将从头训练")
    model = PPO("MlpPolicy", env, **ppo_kwargs)
```

### 3.4 最终保存路径

```python
# 原代码：
# final_model_path = "lift_so101_sac_realistic"
# 改为：
final_model_path = "lift_so101_ppo_realistic"
```

### 3.5 其余部分无需改动

以下部分 **完全复用**，不要动：
- `make_env()` 函数（环境构造、Wrapper、DomainRandomization）
- `VecNormalize` 包装逻辑
- `eval_env` 创建逻辑
- 所有 Callback（`TrainingLoggerCallback`、`RewardBreakdownCallback`、`EvalCallback`、`CheckpointCallback`、`RolloutCollectorCallback`）
- 测试循环

---

## 4. PPO 超参数详解

按重要性排序：

### 4.1 `n_steps`（每 env rollout 长度）⭐⭐⭐⭐⭐

- **含义**：每个并行环境收集多少步后才触发一次策略更新。
- **影响**：每次 update 的总样本数 = `n_steps * n_envs`。太小 → advantage 估计噪声大；太大 → 更新频率低，学习慢。
- **SO101 推荐**：`2048`。
  - horizon=200，2048 步 ≈ 10 个 episode，advantage 估计够稳定。
  - 常见误用：`n_steps=128`，每次只有 128*16=2048 步，太少，PPO 会震荡。
- **必须满足**：`n_steps * n_envs % batch_size == 0`。

### 4.2 `batch_size` ⭐⭐⭐⭐

- **含义**：每次 SGD 的 mini-batch 大小，从 `n_steps * n_envs` 中随机切片。
- **影响**：太小 → 梯度噪声大；太大 → 每次更新太慢、容易陷入局部最优。
- **SO101 推荐**：`64` 或 `128`。
- **整除校验**：`2048 * 16 = 32768`，`32768 / 64 = 512` ✓。

### 4.3 `n_epochs` ⭐⭐⭐⭐

- **含义**：每次 rollout 数据被重复训练多少轮。
- **影响**：太小（1~2）→ 样本浪费；太大（>20）→ 过拟合当前 rollout，策略崩溃。
- **SO101 推荐**：`10`（默认）。
  - 稀疏奖励任务可降到 `3~5`，避免在“空奖励 rollout”上反复强化错误行为。

### 4.4 `learning_rate` ⭐⭐⭐⭐

- **含义**：Adam 优化器的学习率。
- **影响**：PPO 比 SAC 对 lr 更敏感。过大 → 策略更新激进，clip 失效；过小 → 收敛慢。
- **SO101 推荐**：`3e-4` 起步，若训练发散降到 `1e-4`。
- **进阶**：用线性衰减避免后期震荡：
  ```python
  from stable_baselines3.common.utils import get_linear_fn
  lr_schedule = get_linear_fn(3e-4, 1e-5, 1.0)  # 从 3e-4 衰减到 1e-5
  ppo_kwargs["learning_rate"] = lr_schedule
  ```

### 4.5 `clip_range` ⭐⭐⭐

- **含义**：新旧策略概率比的裁剪范围，`0.2` 表示新旧策略 ratio 限制在 `[0.8, 1.2]`。
- **影响**：过小 → 学习慢；过大 → 策略更新太激进，训练发散。
- **SO101 推荐**：`0.2`（原论文值），**不要轻易改**。
- **进阶**：也可线性衰减：
  ```python
  clip_schedule = get_linear_fn(0.2, 0.05, 1.0)
  ppo_kwargs["clip_range"] = clip_schedule
  ```

### 4.6 `ent_coef`（熵正则系数）⭐⭐⭐⭐

- **含义**：策略熵的权重，鼓励探索。
- **影响**：
  - `0`：纯利用，稀疏奖励下学不出来。
  - `0.01`：经典值，平衡探索与利用。
  - `0.05+`：强探索，但策略可能不收敛。
- **SO101 推荐**：`0.01` 起步。若 100K 步后 `reward_reach` 不上升，调到 `0.02`。

### 4.7 `gae_lambda` ⭐⭐⭐

- **含义**：Generalized Advantage Estimation 的 λ，控制 bias-variance tradeoff。
  - `0`：高 bias、低 variance（类似 TD）。
  - `1`：低 bias、高 variance（类似 Monte Carlo）。
- **SO101 推荐**：`0.95`（经典值），几乎不用改。

### 4.8 `gamma` ⭐⭐⭐

- **含义**：折扣因子。
- **SO101 推荐**：`0.99`（与 SAC 一致）。horizon=200，`0.99^200 ≈ 0.13`，远期奖励仍有信号。

### 4.9 `vf_coef` ⭐⭐

- **含义**：value function loss 在总 loss 中的权重。
- **SO101 推荐**：`0.5`（默认）。若 critic 学不准（tensorboard 里 `explained_variance` 低），可调到 `1.0`。

### 4.10 `max_grad_norm` ⭐⭐

- **含义**：梯度裁剪阈值。
- **SO101 推荐**：`0.5`（默认）。训练发散时改为 `0.3`。

### 4.11 `policy_kwargs.net_arch` ⭐⭐⭐

- **含义**：actor / critic 网络结构。
- **SO101 推荐**：
  ```python
  dict(pi=[256, 256], vf=[256, 256])
  ```
  - 观测维度约 30（关节 + eef + rel_pos），`[256, 256]` 容量足够。
  - 若训练慢但稳定，可减到 `[128, 128]` 加速。
  - 若欠拟合，加到 `[512, 512]`，但要配合降 `learning_rate`。

---

## 5. 完整超参数推荐配置

按任务阶段给出三套配置：

### 5.1 快速验证（先跑通）

```python
ppo_kwargs = dict(
    verbose=0,
    tensorboard_log=LOG_DIR,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    device="auto",
)
# NUM_ENVS = 8, TOTAL_TIMESTEPS = 500_000
```

### 5.2 正式训练（推荐）

```python
ppo_kwargs = dict(
    verbose=0,
    tensorboard_log=LOG_DIR,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    device="auto",
)
# NUM_ENVS = 16, TOTAL_TIMESTEPS = 2_000_000
```

### 5.3 稀疏奖励强化（探索不足时）

```python
from stable_baselines3.common.utils import get_linear_fn

ppo_kwargs = dict(
    verbose=0,
    tensorboard_log=LOG_DIR,
    learning_rate=get_linear_fn(3e-4, 1e-5, 1.0),  # 线性衰减
    n_steps=4096,            # 加大 rollout，advantage 更准
    batch_size=128,
    n_epochs=5,              # 降 n_epochs，避免在空奖励 rollout 上过拟合
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=get_linear_fn(0.2, 0.05, 1.0),      # clip 衰减
    ent_coef=0.02,           # 加大探索
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    device="auto",
)
# NUM_ENVS = 16, TOTAL_TIMESTEPS = 3_000_000
```

---

## 6. 训练监控与调参

### 6.1 Tensorboard 关键指标

启动：
```bash
conda run -n robosuite tensorboard --logdir logs/sac_lift_so101_realistic --port 6006
```

重点看：

| 指标 | 健康范围 | 异常处理 |
|------|----------|----------|
| `rollout/ep_rew_mean` | 持续上升 | 死平 → 见 6.2 |
| `rollout/ep_len_mean` | 100~200 | 始终 200 → 策略学不会完成任务 |
| `train/entropy_loss` | 缓慢下降 | 快速归零 → `ent_coef` 太小 |
| `train/policy_gradient_loss` | 在 0 附近震荡 | 持续大正/大负 → `learning_rate` 过大 |
| `train/value_loss` | 持续下降 | 不降反升 → critic 学不准，调 `vf_coef` |
| `train/explained_variance` | 0.5~0.9 | <0 → critic 完全没用，检查 `gamma` |
| `train/clip_fraction` | 0.1~0.3 | >0.5 → `clip_range` 太小或 lr 过大 |

### 6.2 `reward_reach` 不上升的排查

`RewardBreakdownCallback` 每 10K 步打印各分量均值。若 100K 步后 `reward_reach` 仍接近 0：

1. **检查探索**：`ent_coef` 是否太小？调到 `0.02`。
2. **检查归一化**：`vec_normalize_so101_ppo.pkl` 是否用了旧的（被污染的）？删除重训。
3. **检查 `n_steps`**：是否太小（如 128）？调到 `2048`。
4. **检查 lr**：是否过大？`3e-4 → 1e-4`。
5. **用 `diagnose_so101_reward.py` 分析 rollout**，看策略在哪个阶段卡住。

### 6.3 训练发散的排查

症状：`rollout/ep_rew_mean` 突然大幅下跌或变 NaN。

1. 降 `learning_rate`：`3e-4 → 1e-4`。
2. 降 `n_epochs`：`10 → 3`。
3. 降 `max_grad_norm`：`0.5 → 0.3`。
4. 检查 `clip_fraction` 是否 >0.5，是则降 `learning_rate`。

---

## 7. 常见问题（FAQ）

**Q1：PPO 报 `ValueError: batch_size must be smaller than or equal to n_steps * n_envs`？**
A：`batch_size > n_steps * n_envs`。检查：
- `n_steps=2048, n_envs=16, batch_size=64` → `32768 / 64 = 512` ✓
- `n_steps=2048, n_envs=8, batch_size=4096` → `16384 / 4096 = 4` ✓ 但 batch 太大
- `n_steps=128, n_envs=8, batch_size=256` → `1024 / 256 = 4` ✓ 但 rollout 太小

**Q2：PPO 训练比 SAC 慢很多？**
A：正常。PPO 样本效率低。可：
- 加大 `NUM_ENVS`（8 → 16 → 32）。
- 减小 `n_steps`（但不要 <1024）。
- 用 `device="cuda"`（若有 GPU）。

**Q3：PPO 训练 500K 步还是 0 成功？**
A：PPO 通常需要 1M+。若 1M 仍 0 成功：
1. 先用 SAC 在相同环境跑通，确认环境和奖励设计没问题。
2. 检查 `ent_coef` 是否够大（`0.02`）。
3. 用 `diagnose_so101_reward.py` 分析 rollout。

**Q4：`explained_variance` 是负数？**
A：critic 完全没学到东西。可能原因：
- `vf_coef` 太小 → 调到 `1.0`。
- reward 量级过大 → 确认 `VecNormalize(norm_reward=True)` 已开。
- `gamma` 过大 → 确认是 `0.99`。

**Q5：能不能复用 SAC 训好的 `vec_normalize_so101.pkl`？**
A：技术上可以（格式通用），但 **不建议**。SAC 和 PPO 的 rollout 分布不同，归一化统计量会被污染。建议用新文件名 `vec_normalize_so101_ppo.pkl` 从头训。

**Q6：PPO 的 `ent_coef` 训练后期要不要衰减？**
A：通常不需要。PPO 的熵会随策略变确定而自然下降。若发现后期探索过度（策略不收敛），可手动衰减：
```python
ent_schedule = get_linear_fn(0.01, 0.001, 1.0)
ppo_kwargs["ent_coef"] = ent_schedule
```

**Q7：`clip_fraction` 是什么？多少算正常？**
A：被 clip 的样本比例。`0.1~0.3` 健康，说明 clip 在起作用但不过度。`>0.5` 说明策略更新太激进，降 `learning_rate`。

**Q8：PPO 能用 `use_sde=True` 吗？**
A：可以。SB3 的 PPO 支持 `use_sde=True`（state-independent exploration），效果类似 SAC 的状态无关熵。稀疏奖励任务可尝试，但通常 `ent_coef` 足够。

---

## 8. 进阶技巧

### 8.1 学习率与 clip 衰减

长训练（>1M 步）后期策略接近收敛，需要降 lr 和 clip 防止震荡：

```python
from stable_baselines3.common.utils import get_linear_fn

ppo_kwargs = dict(
    # ... 其他参数 ...
    learning_rate=get_linear_fn(3e-4, 1e-5, 1.0),  # 3e-4 → 1e-5
    clip_range=get_linear_fn(0.2, 0.05, 1.0),       # 0.2 → 0.05
    # ...
)
```

### 8.2 自定义回调：早停 + 自动调参

```python
class PPOEarlyStopCallback(BaseCallback):
    """连续 N 次评估无提升则停止训练。"""
    def __init__(self, patience=10, min_delta=0.1):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.best_reward = -np.inf

    def _on_step(self):
        # 从 EvalCallback 同步的评估结果
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
            if mean_reward > self.best_reward + self.min_delta:
                self.best_reward = mean_reward
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    print(f"[EarlyStop] {self.patience} 次评估无提升，停止训练")
                    return False
        return True
```

### 8.3 多环境并行加速

PPO 强依赖 `n_envs`，但 `SubprocVecEnv` 在环境数 >32 时通信开销会显著增加。若需要更多环境：

```python
# 方案1：单机多进程，建议 n_envs <= 32
env = SubprocVecEnv([make_env for _ in range(16)])

# 方案2：若 SO101 仿真本身是瓶颈，可降低 horizon 释放 CPU
# （但 horizon 过小会导致任务不可解，建议 >= 150）
```

### 8.4 与 VecNormalize 配合

PPO 对 reward scale 敏感，`VecNormalize(norm_reward=True)` 必须开。注意：

- 训练时 `env.training = True`，归一化统计量持续更新。
- 评估时 `env.training = False, env.norm_reward = False`，使用冻结的统计量并看真实 reward。
- 脚本末尾的测试循环已经正确处理了这两点，无需改动。

### 8.5 复用 SAC 的环境 Wrapper

`SO101LiftObservationWrapper`（拼 eef→cube 相对位置）和 `SO101LiftRewardShapingWrapper`（shaping 奖励）都是 `gym.Wrapper`，与算法无关，**PPO 直接复用**。这两个 Wrapper 是 PPO 在稀疏奖励下能学出来的关键，不要删。

---

## 9. PPO vs SAC 实测预期

在 SO101 Lift `realistic_state` 任务上的经验对比：

| 指标 | SAC | PPO |
|------|-----|-----|
| 首次出现成功 episode | ~200K 步 | ~800K 步 |
| 50% 成功率 | ~400K 步 | ~1.5M 步 |
| 90% 成功率 | ~500K 步 | ~2.5M 步 |
| 训练 wall-clock（16 envs） | 较快 | 较慢（同等步数） |
| 训练稳定性 | 偶有崩溃 | 很稳定 |
| 超参数敏感度 | 低 | 中 |
| 显存占用 | 高（buffer） | 低 |

**结论**：
- 想快速出结果 → 用 SAC。
- 想要稳定训练 + 工业界通用方案 → 用 PPO，但接受更长的训练时间。
- 想做算法对比研究 → 两个都跑，用相同的 `EvalCallback` 配置对比。

---

## 10. 完整 Checklist

按顺序逐项打勾，确保改造无误：

- [ ] `from stable_baselines3 import SAC` → `from stable_baselines3 import PPO`
- [ ] `MODEL_PATH` 改为 `lift_so101_ppo_realistic.zip`
- [ ] `VEC_NORMALIZE_PATH` 改为 `vec_normalize_so101_ppo.pkl`（建议）
- [ ] `NUM_ENVS` 从 8 调到 16
- [ ] `TOTAL_TIMESTEPS` 从 500K 调到 2M
- [ ] 删除 `sac_kwargs`，替换为 `ppo_kwargs`
- [ ] 删除 `buffer_size`、`tau`、`use_sde`
- [ ] 新增 `n_steps=2048`、`n_epochs=10`、`gae_lambda=0.95`、`clip_range=0.2`、`ent_coef=0.01`、`vf_coef=0.5`、`max_grad_norm=0.5`
- [ ] `policy_kwargs.net_arch` 从 `[256, 256]` 改为 `dict(pi=[256, 256], vf=[256, 256])`
- [ ] `SAC.load(...)` → `PPO.load(...)`
- [ ] `SAC("MlpPolicy", ...)` → `PPO("MlpPolicy", ...)`
- [ ] `final_model_path` 改为 `lift_so101_ppo_realistic`
- [ ] 验证 `n_steps * n_envs % batch_size == 0`（2048*16=32768, 32768%64=0 ✓）
- [ ] 第一次训练 `RESUME = False`
- [ ] 训练前 10K 步检查 `RewardBreakdownCallback` 输出有 `reward_reach` 信号
- [ ] Tensorboard 看 `train/explained_variance > 0`、`train/clip_fraction` 在 0.1~0.3

打完勾即可放心训练。
