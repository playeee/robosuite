# SO101 训练/评估诊断工具与奖励优化工作流

本指南介绍为 SO101 Lift 任务新增的“奖励分解诊断”工具链：它们能做什么、怎么用，以及如何用它们来定位并修复环境/奖励函数的问题。文末用一个**真实案例**（机械臂“抬起来就不下来、奖励却 85 分”）完整演示整个排查过程。

---

## 1. 为什么需要这些工具

强化学习训练里，“总奖励 85 分”是一个**没有诊断价值**的数字——它无法告诉你：

- 这 85 分是策略真正抓起 cube 拿到的，还是某个奖励分项在白送分（reward hacking）？
- 机械臂为什么悬浮在桌面上方不肯下来？
- 接近 cube 的信号有没有传到策略？

新增的工具把奖励拆成 **10 个分量**逐步记录，并和末端/cube 的位置轨迹对照，让你能直接回答“策略拿到的是什么奖励、它在做什么动作”。

### 奖励分量清单

所有工具共用 `so101_realistic/wrappers.py` 中的 `REWARD_COMPONENTS`：

| 分量 | 类型 | 含义 |
|---|---|---|
| `reward_reach` | 正 | 末端接近 cube（tanh 映射） |
| `reward_grasp` | 正 | 抓住 cube（二值里程碑） |
| `reward_lift` | 正 | cube 被抬起（相对静止高度） |
| `reward_smooth` | 负 | 动作平滑度惩罚 |
| `reward_vel` | 负 | 关节速度惩罚 |
| `reward_ee_vel` | 负 | 末端速度惩罚 |
| `reward_z_float` | 负 | 末端悬浮惩罚（eef 高于 cube） |
| `reward_gripper_move` | 正/负 | 夹爪运动 |
| `original_reward` | 总量 | robosuite 原始奖励（成功=2.25） |
| `shaped_reward` | 总量 | 合成后的最终单步奖励 |

> 改奖励分项时只需改 `wrappers.py` 里这一处常量，所有工具自动同步。

---

## 2. 工具一览

| 工具 | 何时用 | 输出 |
|---|---|---|
| `diagnose_so101_reward.py` | 训练后/改奖励后，定位“奖励从哪来、策略在做什么” | 终端报告 + JSON + PNG |
| `visualize_rollout_so101.py` | 想边看渲染边留轨迹 | 渲染窗口 + npz + PNG（含奖励分解堆叠图） |
| `RewardBreakdownCallback` | 训练**过程中**实时监控分量 | 终端 + tensorboard |
| `analyze_rollouts` | 批量统计已保存的 npz 轨迹 | 终端报告 + JSON |
| `eval_so101_visual.py` | 快速看策略跑得怎样 | 渲染窗口 + 每集总奖励 |

---

## 3. 诊断脚本 `diagnose_so101_reward.py`（核心工具）

### 3.1 运行

```bash
conda run -n robosuite python robosuite/demos/diagnose_so101_reward.py
```

可选环境变量覆盖默认配置：

```bash
MODEL_PATH=xxx.zip VEC_NORMALIZE_PATH=xxx.pkl N_EPISODES=5 \
  conda run -n robosuite python robosuite/demos/diagnose_so101_reward.py
```

> 注意：脚本默认从**项目根目录**加载 `lift_so101_sac_realistic.zip` 和 `vec_normalize_so101.pkl`，所以请在项目根目录下运行。

### 3.2 输出位置

- `./logs/sac_lift_so101_realistic/diagnose/reward_breakdown_report.json`
- `./logs/sac_lift_so101_realistic/diagnose/reward_breakdown_epNN.png`

### 3.3 怎么读报告

报告分三块：

**① 初始状态** — episode 开始时桌面、末端、cube 的位置关系。如果初始 eef 就高出 cube 很多，说明机器人初始姿态偏高。

**② 奖励分量累计** — 整个 episode 求和。重点看自动标注的警告：
- `reward_lift` 高但 `original_reward≈0` → **白送分（抬升基线写错）**
- `reward_z_float` 很负 → 末端悬浮严重
- `reward_reach≈0` 且 `original_reward≈0` → 接近信号丢失
- `reward_grasp=0` → 全程未抓取

**③ 位置/距离轨迹** — eef 和 cube 的高度随时间变化、3D 距离最小值、进入 5cm/2cm 的步数、抓取步数、cube 相对静止位置的最大抬升。这是判断“策略到底有没有去碰 cube”的硬指标。

### 3.4 PNG 三联图

1. **奖励分解堆叠面积图**：上方堆叠=正信号（reach/grasp/lift），下方堆叠=惩罚。一眼看出哪个分量在主导奖励。
2. **末端与 cube 高度轨迹**：绿线=成功阈值（+4cm）。看 cube 线有没有被抬过绿线。
3. **末端到 cube 的 3D 距离**：橙线 5cm（接近）、红线 2cm（抓取范围）。

---

## 4. 训练时实时监控：`RewardBreakdownCallback`

已接入训练脚本，无需额外配置。训练时每 10000 步打印：

```
[RewardBreakdown] step 50,000 | 统计 80000 个 env-step
  reward_reach       :  +0.0312   reward_grasp    :  +0.0000
  reward_lift        :  +0.5421   reward_smooth   :  -0.0061
  ...
  original_reward    :  +0.0000   shaped_reward   :  +0.4860
  ⚠ 警告：original_reward≈0 但 reward_lift 偏高 → 疑似抬升奖励基线 reward hacking
```

同时写入 tensorboard 的 `reward_breakdown/` 命名空间，可用 `tensorboard --logdir logs/sac_lift_so101_realistic` 查看。

**用法**：训练前 5 万步盯紧这个输出。如果 `original_reward` 一直是 0 而 `reward_lift` 在涨，立刻停掉训练去查奖励基线——不用等 50 万步跑完再发现策略学废了。

---

## 5. 可视化留痕：`visualize_rollout_so101.py`

```bash
conda run -n robosuite python robosuite/demos/visualize_rollout_so101.py
```

相比诊断脚本，它会**开渲染窗口**实时看动作，并把每条轨迹存成 npz（现在包含全部奖励分量）+ PNG（5 联图，最上面是奖励分解堆叠图）。

输出：`./logs/sac_lift_so101_realistic/visual_rollouts/`

---

## 6. 批量统计：`analyze_rollouts`

训练脚本测试段结束时会自动调用，也可以单独跑：

```python
from so101_realistic import analyze_rollouts
analyze_rollouts("logs/sac_lift_so101_realistic/test_rollouts", save_report=True)
```

现在当 npz 含奖励分量时，报告会多出“奖励分量分解”一节，对每条轨迹的分量累计值求均值±标准差，并自动标注可疑模式。旧轨迹（没存分量的）会显示 `[跳过]`，兼容性好。

---

## 7. 用这些工具优化环境/奖励的工作流

这是一个**通用排查循环**，遇到“策略表现奇怪”时按此走：

```
1. 跑 diagnose_so101_reward.py，拿到奖励分解 + 位置轨迹
        ↓
2. 看哪个分量“不该高却高”或“该有却没有”
        ↓
3. 对照 wrappers.py 里该分量的公式，找参数/基线 bug
        ↓
4. 改 wrappers.py（只改一处），重跑 diagnose 对比“修复前 vs 修复后”
        ↓
5. 分量合理后，从头重训（RESUME=False），训练中用 RewardBreakdownCallback 盯
        ↓
6. 训练完再跑 diagnose 确认策略行为正确
```

### 判读速查表

| 现象 | 可能原因 | 该调什么 |
|---|---|---|
| `reward_lift` 高但 `original_reward=0` | 抬升基线用桌面高度，cube 不动也送分 | 基线改用 cube 静止高度 |
| `reward_reach≈0` | `reach_tanh_scale` 过大，远距离无梯度 | 调小 `reach_tanh_scale`（如 15→5） |
| `reward_z_float` 很负且持续 | 末端悬浮，但罚得不够拉不回来 | 调小阈值、调大权重（0.12→0.06, 0.30→0.40） |
| `reward_grasp=0` 全程 | 从未接触 cube | 检查 cube 放置范围是否在可达工作空间内 |
| `reward_ee_vel` 很负 | 末端速度惩罚过大，连正常运动都罚 | 调小 `w_ee_vel`（如 0.50→0.10） |
| eef 初始就高出 cube 很多 | 机器人初始姿态偏高 | 检查 `init_qpos` / `initialization_noise` |

---

## 8. 真实案例：修复“奖励 85 分却任务全失败”

### 8.1 现象

```
Episode 1: 步数=200, 奖励=85.98
Episode 2: 步数=200, 奖励=84.77
...
平均奖励：85.60
```

机械臂“一开始抬起来就不下来，没接触物体、悬浮在桌上”，任务全失败。

### 8.2 诊断（跑 `diagnose_so101_reward.py`）

奖励分解暴露了根因：

```
reward_lift        :  +109.357   ← ⚠ 白送分？cube没被抬起却拿到抬升奖励（基线写错）
reward_z_float     :   -15.979   ← ⚠ 末端悬浮严重
reward_reach       :    +0.167   ← 接近信号过弱
reward_grasp       :    +0.000   ← 全程未抓取
original_reward    :    +0.000   ← 任务从未成功
shaped_reward      :   +92.3
```

**85 分里 109 分是 `reward_lift` 白送的**——cube 坐在桌上一动不动就拿抬升奖励。

### 8.3 定位 bug

`wrappers.py` 原代码用桌面中心高度当抬升基线：

```python
table_height = base_env.model.mujoco_arena.table_offset[2]   # = 0.8（桌面【中心】）
lift_height  = cube_height - table_height
```

但 cube 静止放在桌上时中心 z≈0.822（比 `table_offset[2]=0.8` 高约 2cm），所以 cube 不动就有 `r_lift ≈ 0.55/步`。叠加 `reach_tanh_scale=15`（过陡，dist>5cm 信号归零）和 `z_float` 罚得太轻，策略学到“把臂举高悬停”这个局部最优。

### 8.4 修复（`wrappers.py`）

1. `reset` 时记录 `_cube_rest_z`（cube 静止高度），抬升奖励只算 `max(0, cube_z - rest_z)` → cube 不动 `r_lift=0`
2. `reach_tanh_scale` 15→5（拉长接近梯度）
3. `z_float` 阈值 0.12→0.06、权重 0.30→0.40（强化悬浮惩罚）
4. `w_ee_vel` 0.50→0.10（不惩罚正常下降运动）

### 8.5 验证（重跑 `diagnose_so101_reward.py` 对比）

| 分量 | 修复前 | 修复后 |
|---|---|---|
| `reward_lift` | +109.4 | **0.0** ✓ |
| `reward_reach` | +0.17 | +4.86 |
| `reward_z_float` | −16.0 | −26.4（罚得更重，但不再被白送分盖过）|
| 总奖励 | +92.3（假成功） | −22.8（正确反映失败）|

修复后“悬停在 1.2m”变成**负奖励**，而“靠近→抓取→抬起”才会拿到正奖励——奖励地形终于指向真正的任务。

### 8.6 重训

⚠ 修复只改奖励信号，**不会改变已训练策略的行为**。必须 `RESUME=False` 从头重训：

```bash
conda run -n robosuite python robosuite/demos/train_rl_sb3_so101_realistic.py
```

重训时用 `RewardBreakdownCallback` 盯前 5 万步，确认 `original_reward` 开始出现非零（说明策略真去抓 cube 了），而不是 `reward_lift` 单方面虚高。

---

## 9. 原理视角：从案例到一般方法论

§8 的案例看似是「调几个系数」，背后其实有一套**从梯度反推根因**的方法论。
本节把案例抽象为可复用的诊断流程，并指向主指南的深层原理章节。

### 9.1 案例的两层根因（梯度视角）

§8 的「悬浮不动」其实叠加了**两类独立的奖励设计失误**：

| 失误类型 | 表现 | 梯度本质 |
|---|---|---|
| **正信号虚高** | `reward_lift=+109` 但 `original_reward=0` | 基线选错造成假山头，策略梯度指向"举高臂"而非"抬 cube" |
| **正梯度饱和** | `reward_reach=+0.17`，几乎无信号 | tanh 系数 k=5 过大，dist=0.4 处梯度仅 -0.18，被 z_float 噪声淹没 |

第一类用「奖励分量分解」一眼能看出（lift 虚高 + original=0）。
第二类更隐蔽——`reward_reach` 不为 0，看起来"有信号"，但梯度不足以驱动策略。

### 9.2 为什么"加大 z_float 惩罚"不能破局

案例中最初尝试把 `w_z_float` 从 0.30 提到 0.40，结果策略仍悬浮。原因在
[`reward_function_guide.md` §9.4.3](./reward_function_guide.md#943-为什么不能只加大负惩罚)：

- z_float 的负梯度**只在小阈值附近有效**，远距离时梯度方向虽对但量级被 reach 淹没
- 加大 w_z_float 只是"罚得更重"，**不改变梯度方向的竞争关系**
- 真正的破局是**降低 `reach_tanh_scale`**，让 reach 的正梯度在远距离也显著

这是「负惩罚只告诉策略『别在哪』，方向引导必须靠正信号梯度」这一原理的
具体体现。

### 9.3 诊断流程（梯度反推法）

把 §8 的排查过程抽象为通用流程：

```
1. 跑 diagnose_so101_reward.py，拿到分量累计值
        ↓
2. 分类检查（先看最直接的信号）
   ├─ original_reward > 0？→ 任务有进展，跳到 4
   └─ original_reward ≈ 0？→ 进入"假成功"或"无进展"分支
        ↓
3. 假成功 vs 无进展的判别
   ├─ 某正分量虚高（如 lift/grasp 总分 >> 0）→ 假成功（reward hacking）
   │     → 定位基线/截断设计漏洞（见 reward_function_guide.md §9.6.1）
   └─ 所有正分量都接近 0 → 无进展
   │     → 检查负分量：是否有某项极负？
   │        ├─ ee_vel ≈ 0 且策略不动 → 静止局部最优（见 §9.4.2）
   │        └─ z_float 很负但策略悬浮 → 远距离梯度饱和
        ↓
4. 计算症状点梯度（见 reward_function_guide.md §9.6.2）
   - 用 reach_grad(w, k, dist) 算出 |dr/ddist|
   - |dr/ddist| < 0.5 → 降低 reach_tanh_scale
   - 正常梯度但仍不动 → 提高 w_reach 或降低 motion 惩罚
        ↓
5. 改一处，重跑 diagnose 对比
        ↓
6. 量级平衡校验（见 reward_function_guide.md §9.6.3）
   - Σ(正分量) : |Σ(负分量)| 应在 2:1 ~ 4:1
        ↓
7. 重训（RESUME=False），RewardBreakdownCallback 盯前 5 万步
```

### 9.4 SO101 系数表的梯度含义

下表把 [`wrappers.py`](../../robosuite/demos/so101_realistic/wrappers.py) 当前的系数
与梯度原理对应起来，方便后续调参时定位「调这个会影响什么梯度」：

| 系数 | 当前值 | 影响的梯度 | 调参直觉 |
|---|---|---|---|
| `w_reach_pbrs` | 20.0 | PBRS 差分幅度（α） | 不动 → 提高；震荡刷分 → 降低 |
| `reach_pbrs_scale` | 5.0 | tanh 陡峭度（k） | 远处不动 → 调小；近处不精细 → 调大 |
| `w_grasp` | 0.50 | 抓取过渡信号（二值） | 全程未抓 → 检查可达性，而非调此 |
| `w_lift` | 1.00 | 抬升峰值（正梯度幅度） | 仅在已抓取后有意义 |
| `lift_target_height` | 0.04 | lift 梯度饱和阈值 | 太小 → 轻抬满分；太大 → 信号稀疏 |
| `w_action_smooth` | 0.02 | 动作变化惩罚（负梯度） | 抖动 → 提高；不动 → 降低 |
| `w_joint_vel` | 0.002 | 关节速度惩罚（负梯度） | 同上 |
| `w_ee_vel` | 0.03 | 末端速度惩罚（负梯度） | 不动 → 大幅降低；抖动 → 提高 |
| `w_z_float` | 0.50 | 悬浮方向引导（负梯度） | **不要降低**，是下降方向的关键信号 |
| `z_float_threshold` | 0.04 | z_float 触发高度 | 误伤正常接近 → 调小；不抑制悬浮 → 调大 |

> 注：`r_reach` 已从"绝对 tanh 形式"改为 PBRS 差分形式，详见
> [reward_function_guide.md §9.5.3](./reward_function_guide.md#953-实战差分-reach-在-so101-上的破局效果)。
> 旧的 `w_reach` 与 `reach_tanh_scale` 已废弃。

### 9.5 与主指南的对应关系

本文档专注于"**工具与案例**"，背后的原理在主指南的
[§9 奖励函数优化的深层原理](./reward_function_guide.md#9-奖励函数优化的深层原理)
有完整论述。对照阅读建议：

| 本文档章节 | 主指南对应章节 |
|---|---|
| §1 奖励分量清单 | §9.1 奖励函数 = 策略梯度的来源 |
| §3.3 怎么读报告 | §9.6 诊断方法论 |
| §7 排查循环 | §9.7 调参决策树 |
| §8 真实案例 | §9.4 奖励地形与局部最优 |
| §8.4 修复（调系数） | §9.2 形状函数的梯度分析 |
| §8.5 验证 | §9.6.3 量级平衡校验流程 |
| §9.4 系数表（含 PBRS） | §9.5.3 PBRS 差分 reach 实战 |

### 9.6 一句话总结

**诊断的本质不是看"奖励多少分"，而是问"每个分项的梯度方向和量级是否指向任务目标"。**
工具提供数据，原理提供判断力——两者结合才能从"试参"升级为"调参"。

---

## 10. 文件清单

| 文件 | 作用 |
|---|---|
| `robosuite/demos/so101_realistic/wrappers.py` | 奖励分量定义 + `REWARD_COMPONENTS` 常量 |
| `robosuite/demos/so101_realistic/callbacks.py` | `RewardBreakdownCallback`（训练时监控） |
| `robosuite/demos/so101_realistic/rollout_utils.py` | `analyze_rollouts`（含分量分解） |
| `robosuite/demos/diagnose_so101_reward.py` | 独立诊断脚本（核心工具） |
| `robosuite/demos/visualize_rollout_so101.py` | 可视化 + 奖励分解堆叠图 |
| `robosuite/demos/train_rl_sb3_so101_realistic.py` | 训练脚本（已接入 callback + 测试段保存分量） |
| `robosuite/demos/eval_so101_visual.py` | 可视化评估 |
| `docs/tutorials/reward_function_guide.md` | 奖励函数设计完全指南（含 §9 深层原理） |
