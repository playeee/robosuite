# SO101 Lift 强化学习算法使用教学指南

本指南面向 [`train_rl_sb3_so101_realistic.py`](./train_rl_sb3_so101_realistic.py) 训练脚本，讲解如何把当前使用的 **SAC** 替换为其他 Stable-Baselines3（SB3）算法（如 **PPO**、**TD3**、**DDPG**），并指出切换时必须注意的陷阱。

---

## 1. 先看懂本任务的特点

在选算法之前，必须先认清 SO101 Lift 这个任务本身：

| 维度 | 取值 | 对算法的影响 |
|------|------|--------------|
| 动作空间 | **连续**（5 关节位置 + 1 夹爪，约 6 维） | 排除 DQN 系列，只能用连续控制算法 |
| 观测空间 | 向量（关节 + eef + eef→cube 相对位置） | 用 `MlpPolicy` 即可，不需要 CNN |
| Episode 长度 | `horizon=200` | 较短，on-policy 也能跑得动 |
| 奖励密度 | `realistic_state` 模式下稀疏（成功 +2.25），但 Wrapper 补了 shaping | 介于纯稀疏与稠密之间 |
| 训练步数 | 500K | off-policy 够用；on-policy 偏少 |
| 并行环境 | `NUM_ENVS=8`（`SubprocVecEnv`） | on-policy 受益更大，可再加 |
| 归一化 | `VecNormalize(norm_obs=True, norm_reward=True)` | 所有算法通用，但 on-policy 对 reward scale 更敏感 |

**结论**：本任务动作连续、长度短、有 shaping 信号，SAC / TD3 / DDPG / PPO / A2C 都可以跑，但表现差异很大。

---

## 2. SB3 算法分类与本任务适用性

SB3 内置的连续控制算法按更新方式分两类：

### 2.1 Off-policy（基于 replay buffer）
- **SAC**（当前默认）：最大熵强化学习，自带探索，样本效率最高，超参数最不敏感。**首选**。
- **TD3**：SAC 的“无熵”版本，确定性策略，需要外部探索噪声。比 SAC 稳定但探索弱。
- **DDPG**：TD3 的前身，单 Q 网络、无目标策略平滑，训练易发散。**不推荐**，用 TD3 替代。

特点：
- 使用 replay buffer，1 个环境也能训练（`SubprocVecEnv` 主要用于加速采样）。
- `batch_size` 从 buffer 随机采样，与 `n_envs` 无关。
- 样本效率高，500K 步通常够 SO101 Lift 收敛。

### 2.2 On-policy（基于当前策略 rollout）
- **PPO**：最常用的 on-policy 算法，超参数多但鲁棒。
- **A2C**：PPO 的简化版（同步 Advantage Actor-Critic），性能通常不如 PPO，**不推荐**。

特点：
- **没有 replay buffer**，每收集 `n_steps * n_envs` 步就更新一次策略然后丢弃。
- **强依赖并行环境数**：`n_envs` 越多越稳定。SO101 Lift 建议至少 8，可加到 16~32。
- 样本效率低，500K 步可能不够，建议 1M~3M。
- `batch_size` 必须 **整除** `n_steps * n_envs`。

### 2.3 不适用的算法
- **DQN / DDPG-Discrete / SAC-Discrete**：只能处理离散动作空间，SO101 是连续控制，**不可用**。
- **A2C**：理论上可跑，但 PPO 是其严格改进，没有理由选它。

---

## 3. 切换算法时的通用注意事项

无论换哪种算法，下面这些点都必须检查：

### 3.1 模型文件不能跨算法 load
脚本里的 `RESUME` 机制：
```python
model = SAC.load(MODEL_PATH, env=env, **sac_kwargs)
```
`MODEL_PATH = "lift_so101_sac_realistic.zip"` 是 **SAC 专用** 的。换算法后：
1. **必须改 `MODEL_PATH`**，例如 `lift_so101_ppo_realistic.zip`，否则 `PPO.load("..._sac_...")` 会直接报错。
2. 第一次训练时把 `RESUME = False`，跑出 `.zip` 后再改回 `True`。
3. `VEC_NORMALIZE_PATH`（`vec_normalize_so101.pkl`）跨算法 **可以复用**，因为它只存观测/奖励的统计量，与算法无关。但建议不同算法用不同文件，避免归一化统计量被另一个算法的训练数据污染。

### 3.2 `sac_kwargs` 里有些参数是 SAC 专属
当前脚本：
```python
sac_kwargs = dict(
    verbose=0,
    tensorboard_log=LOG_DIR,
    learning_rate=3e-4,
    buffer_size=200_000,     # 仅 off-policy
    batch_size=256,
    tau=0.005,               # 仅 off-policy（soft target update）
    gamma=0.99,
    policy_kwargs=dict(net_arch=[256, 256], use_sde=False),  # use_sde 仅 SAC
    device="auto",
)
```
- `buffer_size`、`tau`、`use_sde` 都是 **SAC 专属**，传给 PPO 会直接 `TypeError`。
- `learning_rate`、`gamma`、`batch_size`、`policy_kwargs.net_arch`、`device`、`verbose`、`tensorboard_log` 是通用参数。

### 3.3 `VecNormalize` 通用但语义有差异
- `norm_obs=True`：所有算法都建议开。
- `norm_reward=True`：
  - Off-policy：通常开（SAC 默认配置就是这样）。
  - On-policy（PPO）：也建议开，但要注意 PPO 的 `clip_range` 是在 **归一化后的 reward** 上算的，所以 `clip_range=0.2` 通常不用改。
- `eval_env` 的 `norm_reward=False` 必须保持，评估时要看真实奖励。

### 3.4 `Callback` 全部通用
`TrainingLoggerCallback`、`RewardBreakdownCallback`、`EvalCallback`、`CheckpointCallback`、`RolloutCollectorCallback` 都基于 SB3 的 `BaseCallback`，与算法无关，**直接复用即可**。

### 3.5 `policy_kwargs.net_arch` 的结构差异
- **SAC / TD3 / DDPG**：actor 和 critic 共享同一 `net_arch=[256, 256]` 即可。
- **PPO**：SB3 ≥ 1.6 推荐用 **分离结构**：
  ```python
  policy_kwargs=dict(
      net_arch=dict(pi=[256, 256], vf=[256, 256]),  # actor/critic 分开
  )
  ```
  旧的列表式 `net_arch=[256, 256]` 仍兼容但已 deprecated。

### 3.6 `n_envs` 的影响
- SAC：8 个环境主要为了加速采样，少了也能跑。
- PPO：`n_envs` 直接决定每次更新的 batch 大小，**太少会导致策略更新方差大、训练不稳**。建议 8~16。

---

## 4. PPO 改造完整示例

PPO 是 on-policy，与 SAC 差异最大，这里给出完整改造步骤。

### 4.1 关键差异速查

| 项 | SAC | PPO |
|----|-----|-----|
| replay buffer | 有（`buffer_size`） | 无 |
| 每次更新数据来源 | buffer 随机采样 | 当前策略 rollout |
| `n_steps` | 无 | **必须设**（每 env rollout 长度） |
| `batch_size` | 任意（从 buffer 采） | 必须 **整除** `n_steps * n_envs` |
| `tau` | 0.005（soft update） | 无 |
| `use_sde` | 有 | 无 |
| 探索机制 | 最大熵（内置） | action 分布的熵正则（`ent_coef`） |
| 训练步数 | 500K 够 | 建议 1M~3M |
| `n_envs` | 8 够 | 建议 8~16 |

### 4.2 修改清单

#### Step 1：改 import
```python
from stable_baselines3 import SAC  # 删掉
from stable_baselines3 import PPO  # 改成
```

#### Step 2：改模型路径
```python
MODEL_PATH = "lift_so101_sac_realistic.zip"           # 删掉
MODEL_PATH = "lift_so101_ppo_realistic.zip"           # 改成
VEC_NORMALIZE_PATH = "vec_normalize_so101.pkl"
# 建议归一化文件也分开，避免污染
VEC_NORMALIZE_PATH = "vec_normalize_so101_ppo.pkl"
```

#### Step 3：改超参数
把 `sac_kwargs` 整段替换为 `ppo_kwargs`：
```python
ppo_kwargs = dict(
    verbose=0,
    tensorboard_log=LOG_DIR,
    learning_rate=3e-4,
    n_steps=2048,           # 每个 env 收集 2048 步才更新一次
    batch_size=64,          # 必须 <= n_steps * n_envs，建议能整除
    n_epochs=10,            # 每次 rollout 重复训练 10 轮
    gamma=0.99,
    gae_lambda=0.95,        # GAE 参数
    clip_range=0.2,         # PPO clip
    ent_coef=0.01,          # 熵正则，鼓励探索（稀疏奖励任务可调到 0.02）
    vf_coef=0.5,            # value function loss 权重
    max_grad_norm=0.5,      # 梯度裁剪
    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    device="auto",
)
```

#### Step 4：改模型构造
```python
if RESUME and os.path.exists(MODEL_PATH):
    print(f"加载已有模型：{MODEL_PATH}")
    model = PPO.load(MODEL_PATH, env=env, **ppo_kwargs)
else:
    if RESUME:
        print(f"[警告] 未找到模型 {MODEL_PATH}，将从头训练")
    model = PPO("MlpPolicy", env, **ppo_kwargs)
```

#### Step 5：调大总步数
PPO 样本效率低，500K 通常不够：
```python
TOTAL_TIMESTEPS = 2_000_000   # PPO 建议 1M~3M
```

#### Step 6（可选）：增加并行环境
```python
NUM_ENVS = 16   # PPO 受益于更多环境
```
注意 `n_steps * n_envs` 是每次更新的总样本数。`2048 * 16 = 32768`，`batch_size=64` 整除，OK。

### 4.3 PPO 常见坑

1. **`batch_size` 不能整除 `n_steps * n_envs`** → SB3 会报错。
   - 检查：`2048 * 8 = 16384`，`16384 / 64 = 256` ✓
2. **`n_steps` 太小**（如 128）→ 每次 rollout 太短， advantage 估计噪声大，训练发散。
   - 建议 `n_steps >= 1024`，SO101 horizon=200，用 2048 就是 10 个 episode，合适。
3. **`ent_coef` 设 0** → 稀疏奖励下探索不足，策略早早收敛到“不动最安全”。
   - 建议 `ent_coef=0.01~0.02`。
4. **`learning_rate` 过大** → PPO 比 SAC 对 lr 更敏感，`3e-4` 通常 OK，但若发散可降到 `1e-4` 或用 `lr_schedule` 线性衰减。
5. **`clip_range` 过大** → 策略更新太激进，`0.2` 是经典值，不要轻易改。
6. **没调 `n_epochs`** → 默认 10，对稀疏奖励任务可降到 3~5，避免过拟合当前 rollout。

---

## 5. TD3 改造示例

TD3 与 SAC 同属 off-policy，改造比 PPO 简单很多。

### 5.1 关键差异

| 项 | SAC | TD3 |
|----|-----|-----|
| 探索 | 最大熵（内置） | 外部高斯噪声（`action_noise`） |
| Critic | 单 Q（双 Q 估计） | 双 Q（min） |
| `tau` | 0.005 | 0.005（同） |
| `use_sde` | 有 | 无 |
| `target_entropy` | 有 | 无 |
| `ent_coef` | 有 | 无 |

### 5.2 修改清单

```python
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise

MODEL_PATH = "lift_so101_td3_realistic.zip"

# TD3 必须显式提供探索噪声，否则策略是确定性的，几乎不探索
n_actions = 6  # SO101: 5 关节 + 1 夹爪，需根据 env.action_space 确认
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

td3_kwargs = dict(
    verbose=0,
    tensorboard_log=LOG_DIR,
    learning_rate=3e-4,
    buffer_size=200_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    action_noise=action_noise,        # TD3 必须有
    policy_kwargs=dict(net_arch=[256, 256]),
    device="auto",
)

if RESUME and os.path.exists(MODEL_PATH):
    model = TD3.load(MODEL_PATH, env=env, **td3_kwargs)
else:
    model = TD3("MlpPolicy", env, **td3_kwargs)
```

### 5.3 TD3 注意点

1. **`action_noise` 必须设**，否则 TD3 几乎不探索，稀疏奖励任务直接学不出来。
   - `sigma=0.1` 是经验值，过小探索不够，过大策略抖动。
2. **`action_noise` 的维度必须匹配 `env.action_space`**，不要硬编码 6，应该从 env 拿：
   ```python
   n_actions = env.action_space.shape[0]  # 在 VecNormalize 包装后取
   ```
3. TD3 没有 `use_sde`，去掉它。
4. TD3 的 `policy_delay=2`（默认）：critic 更新 2 次，actor 才更新 1 次，通常不用改。

---

## 6. 算法对比与推荐

| 算法 | 样本效率 | 训练稳定性 | 探索能力 | 调参难度 | SO101 Lift 推荐 |
|------|----------|------------|----------|----------|----------------|
| **SAC** | ★★★★★ | ★★★★★ | ★★★★★（内置熵） | ★ | **首选**，当前默认 |
| **TD3** | ★★★★ | ★★★★ | ★★（依赖噪声） | ★★ | 次选，想用确定性策略时 |
| **DDPG** | ★★★ | ★★ | ★★ | ★★★ | 不推荐，用 TD3 替代 |
| **PPO** | ★★ | ★★★★ | ★★★（ent_coef） | ★★ | 可选，需要更多步数 |
| **A2C** | ★ | ★★★ | ★★ | ★ | 不推荐，PPO 严格更优 |

**实务建议**：
- **第一次跑通**：用 SAC（当前配置），它能最快验证环境和奖励设计是否正确。
- **想试 on-policy**：用 PPO，但把 `TOTAL_TIMESTEPS` 调到 2M，`NUM_ENVS` 调到 16。
- **想试确定性策略**：用 TD3，但务必设 `action_noise`。
- **不要用** DDPG、A2C、DQN 系列。

---

## 7. 验证算法是否正常工作

换算法后，按下面顺序检查：

### 7.1 训练前 10K 步
- **看 tensorboard**：`reward` 曲线应该有变化，不是死平。
- **看终端 `RewardBreakdownCallback`**：各奖励分量均值不应全为 0，至少 `reward_reach` 应该随训练上升。
- **若全为 0**：说明策略完全没探索，检查 `ent_coef`（PPO）、`action_noise`（TD3）。

### 7.2 训练 100K 步
- `reward_reach` 应该明显上升（策略学会靠近 cube）。
- `reward_z_float`（悬浮惩罚）应该被触发并逐渐减小。
- 若 `reward_reach` 不升：检查 `VecNormalize` 是否正常、`learning_rate` 是否过大。

### 7.3 训练 500K 步
- SAC：应该出现成功 episode。
- PPO：可能需要到 1M 才出现。
- 若始终 0 成功：用 `diagnose_so101_reward.py` 分析 rollout，看策略在哪个阶段卡住。

### 7.4 评估环境
`EvalCallback` 会定期在 `eval_env` 上跑 5 个 episode，看 `eval/mean_reward` 曲线：
- 持续上升 → 正常。
- 震荡剧烈 → `learning_rate` 过大或 `batch_size` 过小。
- 死平 → 策略没学到东西，回到 7.1 排查。

---

## 8. 常见问题（FAQ）

**Q1：换算法后 `RESUME=True` 报错 `KeyError: '..._class'`？**
A：跨算法 load 了。改 `MODEL_PATH` 为新算法专用文件名，第一次 `RESUME=False` 跑出 `.zip` 再改回 `True`。

**Q2：PPO 报 `batch_size must be <= n_steps * n_envs`？**
A：检查 `n_steps * n_envs >= batch_size`，且 `n_steps * n_envs % batch_size == 0`。

**Q3：TD3 训练 500K 步还是 0 成功？**
A：`action_noise` 没设或太小。稀疏奖励任务 `sigma` 可调到 0.2。

**Q4：PPO 训练比 SAC 慢很多？**
A：正常。PPO 样本效率低，但每次更新轻。整体 wall-clock 时间不一定更长，取决于 `n_envs`。可加大 `NUM_ENVS` 加速。

**Q5：能不能用同一个 `VecNormalize` 文件跨算法？**
A：技术上可以（格式通用），但建议分开。不同算法的 rollout 分布不同，归一化统计量会被互相污染，影响后续训练。

**Q6：换算法后 reward 量级变了，`VecNormalize` 要重训吗？**
A：`norm_reward=True` 会自动适应 reward 量级，不用重训。但如果换算法后 reward 量级剧烈变化（如从 ±1 变成 ±100），建议删掉旧 `vec_normalize.pkl` 重新开始。

**Q7：SAC 的 `use_sde=True` 在 TD3/PPO 里有对应物吗？**
A：没有。`use_sde` 是 SAC 专属的状态无关探索。其他算法用 `action_noise`（TD3）或 `ent_coef`（PPO）实现探索。

---

## 9. 一键切换清单（Checklist）

换算法时按这个清单逐项打勾：

- [ ] 改 `from stable_baselines3 import XXX`
- [ ] 改 `MODEL_PATH` 为新算法专用文件名
- [ ] （建议）改 `VEC_NORMALIZE_PATH` 为新算法专用文件名
- [ ] 改 `sac_kwargs` → `xxx_kwargs`，删除算法专属参数
- [ ] 改 `SAC.load` / `SAC("MlpPolicy", ...)` → `XXX.load` / `XXX("MlpPolicy", ...)`
- [ ] PPO 专属：设 `n_steps`、`n_epochs`、`gae_lambda`、`clip_range`、`ent_coef`
- [ ] TD3 专属：设 `action_noise`
- [ ] 检查 `policy_kwargs.net_arch` 结构（PPO 用 dict，SAC/TD3 用 list）
- [ ] PPO：检查 `n_steps * n_envs % batch_size == 0`
- [ ] PPO：调大 `TOTAL_TIMESTEPS`（1M~3M）
- [ ] 第一次跑 `RESUME = False`
- [ ] 训练前 10K 步看 `RewardBreakdownCallback` 输出，确认有探索信号

按这个清单走，基本能避开所有常见坑。
