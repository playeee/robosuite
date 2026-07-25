# SO101 机械臂集成到 robosuite 的实践记录

本文档记录把 [SO-ARM100/SO101](https://github.com/huggingface/lerobot/tree/main/examples/so100) 的真实机器人模型接入 [robosuite](https://robosuite.ai/) 仿真框架，并基于 SO101 训练 Lift 任务的全过程。内容包括模型准备、XML 调整、代码注册、常见问题修复，以及最终生成的训练脚本说明。

---

## 1. 目标与整体思路

### 1.1 目标

- 在 robosuite 中使用 SO101 机器人替代默认的 Panda/Sawyer 等机械臂。
- 运行 `Lift` 操作任务，验证 SO101 模型可正常加载、reset、step。
- 提供一份面向 sim2real 的训练脚本：加入初始化噪声、域随机化、稀疏奖励等更接近真实世界的设置。

### 1.2 集成流程概览

SO101 原始资产来自 `/home/playeee/projects/SO-ARM100`，核心是一个 MuJoCo-compatible 的 MJCF 文件。接入 robosuite 需要完成四步：

1. **资产复制**：把 STL mesh 与 `robot.xml` 放到 `robosuite/models/assets/robots/so101/`。
2. **机器人模型类**：在 `robosuite/models/robots/manipulators/` 中创建 `so101_robot.py`，继承 `ManipulatorModel`。
3. **夹爪模型类**：在 `robosuite/models/grippers/` 中创建 `so101_gripper.py`，继承 `GripperModel`。
4. **注册**：在 `__init__.py` 与 `ROBOT_CLASS_MAPPING` 中注册 SO101。
5. **控制器配置**：在 `robosuite/controllers/config/robots/default_so101.json` 中定义 JOINT_POSITION + GRIP 控制器。

---

## 2. 资产与 XML 调整

### 2.1 目录结构

```text
robosuite/models/assets/robots/so101/
├── robot.xml          # SO101 本体 MJCF
└── assets/            # STL mesh 文件
    ├── base_motor_holder_so101_v1.stl
    ├── base_so101_v2.stl
    ├── moving_jaw_so101_v1.stl
    ├── sts3215_03a_v1.stl
    ├── ...
```

### 2.2 mesh 必须显式命名

原始 `robot.xml` 中的 mesh 定义类似：

```xml
<mesh file="waveshare_mounting_plate_so101_v2.stl"/>
```

robosuite 在合并 XML 时会根据 `name` 属性添加 `robot0_` 前缀。如果没有 `name`，geom 引用 `mesh="xxx"` 会被前缀化为 `mesh="robot0_xxx"`，而 asset 中的 mesh 却仍是未命名的，导致 MuJoCo 报 `mesh 'robot0_xxx' not found`。

**修复**：给每个 mesh 添加 `name`，并把 `file` 改为相对于 xml 文件的路径：

```xml
<mesh name="waveshare_mounting_plate_so101_v2" file="assets/waveshare_mounting_plate_so101_v2.stl"/>
```

> 注意：robosuite 的 `resolve_asset_dependency` 会把 `file` 与 xml 所在目录直接拼接，**不会**读取 `meshdir` 属性。因此 `file` 必须写成 `assets/xxx.stl`，不能只用 `xxx.stl`。

### 2.3 视觉/碰撞 group 调整

SO101 原始 XML 中 visual geom 的 `group="2"`、collision geom 的 `group="3"`。robosuite 默认的 group 映射为：

- `group="1"`：视觉 mesh
- `group="0"`：碰撞 mesh

需要把所有 `group="2"` 改为 `group="1"`，`group="3"` 改为 `group="0"`，否则渲染或碰撞检测会异常。

### 2.4 夹爪与手臂分离

SO101 原始 XML 把 gripper joint 直接定义在 robot 本体的最后一个 link 中：

```xml
<body name="gripper">
  ...
  <joint name="gripper" .../>
  <body name="moving_jaw_so101_v1">...</body>
</body>
```

这导致 `robot_model.joints` 同时包含 5 个 arm joints 和 1 个 gripper joint。robosuite 内部在 reset 时用 `_ref_joint_pos_indexes`（长度为 6）去赋值 `init_qpos`（长度只有 5），触发 `shape mismatch` 错误。

**修复**：把 gripper 的几何体、joint、actuator、sensor 全部拆到独立的 `grippers/so101_gripper.xml` 中，robot.xml 只保留空的 `gripper` body 作为挂载点。robosuite 在加载机器人时会自动把 gripper.xml 中的 `right_gripper` body 合并到 robot.xml 的 `gripper` body 下。

### 2.5 夹爪 XML 的注意事项

- **命名前缀**：`SO101Gripper.naming_prefix` 必须覆盖为 `robot{idn}_`，与机器人本体一致，才能共享 robot.xml 中定义的 mesh/material。
- **不使用 class 属性**：gripper.xml 单独解析时找不到 robot.xml 中的 `<default class="...">`。需要把 `class="visual"`、`class="collision"`、`class="sts3215"` 内联成具体属性。
- **geom 必须命名**：gripper.xml 中的 geom 如果没有 `name`，MuJoCo 会自动命名为 `g0`、`g1`...，与 robot.xml 中的自动命名冲突，报 `repeated name`。
- **end-effector sites**：保留 `gripperframe`、`grip_site`、`ee_x/y/z`、`grip_site_cylinder`，满足 robosuite 的力传感器与抓取检测需求。

### 2.6 添加 `right_center` site

arm controller 在 `update_state()` 时需要 `{prefix}right_center` site 作为 base pose 参考。在 robot.xml 的 base body 中添加：

```xml
<site name="right_center" pos="0 0 0" size="0.01" rgba="1 0.3 0.3 1" group="2"/>
```

---

## 3. 代码注册

### 3.1 机器人模型类

文件：`robosuite/models/robots/manipulators/so101_robot.py`

关键属性：

```python
@property
def default_base(self):
    # SO101 自带底座，不需要额外的 RethinkMount
    return "NullBase"

@property
def default_gripper(self):
    return {"right": "SO101Gripper"}

@property
def _eef_name(self):
    # 告诉 robosuite 末端执行器 body 的名称
    return {"right": "gripper"}
```

阻尼设置只需 5 个值（对应 5 个 arm joints）：

```python
self.set_joint_attribute(attrib="damping", values=np.array([0.6, 0.6, 0.6, 0.6, 0.6]))
```

### 3.2 夹爪模型类

文件：`robosuite/models/grippers/so101_gripper.py`

核心是让 `naming_prefix` 与机器人一致：

```python
@property
def naming_prefix(self):
    return f"robot{str(self.idn).split('_')[0]}_"
```

### 3.3 注册入口

- `robosuite/models/robots/manipulators/__init__.py`：导入 `SO101`。
- `robosuite/models/grippers/__init__.py`：把 `SO101Gripper` 加入 `GRIPPER_MAPPING`。
- `robosuite/robots/__init__.py`：把 `SO101` 加入 `ROBOT_CLASS_MAPPING`。

### 3.4 控制器配置

文件：`robosuite/controllers/config/robots/default_so101.json`

```json
{
    "type": "BASIC",
    "body_parts": {
        "arms": {
            "right": {
                "type": "JOINT_POSITION",
                "input_max": 1,
                "input_min": -1,
                "output_max": [0.1, 0.1, 0.1, 0.1, 0.1],
                "output_min": [-0.1, -0.1, -0.1, -0.1, -0.1],
                "kp": 50,
                "damping_ratio": 1,
                "impedance_mode": "fixed",
                "gripper": {"type": "GRIP"}
            }
        }
    }
}
```

> 环境会通过 `load_composite_controller_config(robot="SO101")` 自动加载该文件，训练脚本中无需手动指定 controller_configs。

---

## 4. 关键错误与修复速查

| 错误信息 | 原因 | 修复 |
|---------|------|------|
| `KeyError: 'visual'` / `KeyError: 'sts3215'` | default class 嵌套或 class 属性在独立 gripper.xml 中找不到 | 内联 geom/joint/actuator 属性，不再依赖 class |
| `KeyError: '2'` in group_mapping | group 编号与 robosuite 默认映射冲突 | visual 改为 `group="1"`，collision 改为 `group="0"` |
| `mesh 'robot0_xxx' not found` | mesh 没有 `name` 属性，或 `file` 路径未相对于 xml 目录 | 给 mesh 加 name，file 写成 `assets/xxx.stl` |
| `shape mismatch: value array of shape (5,) could not be broadcast to indexing result of shape (6,)` | gripper joint 被算作 arm joint | 把 gripper joint/actuator 移到 gripper.xml |
| `repeated name 'robot0_g0_vis'` | gripper.xml 中的 geom 无 name，与 robot.xml 自动命名冲突 | 给 gripper.xml 中的 geom 显式命名 |
| `No "site" with name robot0_right_center` | 缺少 arm controller base pose site | 在 base body 中添加 `<site name="right_center">` |

---

## 5. 训练脚本详解与自定义指南

本节详细说明 `robosuite/demos/train_rl_sb3_so101_realistic.py` 的结构、三种训练模式，以及如何针对真实 SO101 的组装误差、实际任务需求、奖励设计和训练流程进行自定义。

---

### 5.1 脚本位置与结构

文件：`robosuite/demos/train_rl_sb3_so101_realistic.py`

脚本由三部分组成：

1. **配置区**（脚本顶部）：机器人名称、训练步数、模式开关 `TRAIN_MODE`、是否继续训练等。
2. **`make_env()`**：环境工厂函数，每次被 `SubprocVecEnv` 调用时创建一个独立的 SO101 + Lift 环境，并包装域随机化、GymWrapper、Monitor。
3. **`main()`**：创建向量化环境、加载/创建 SAC 模型、训练、保存、测试。

为什么需要工厂函数？

`SubprocVecEnv` 会为每个并行环境 spawn 一个子进程，子进程通过重新 `import` 主模块来获取环境构造逻辑。工厂函数确保每个子进程创建独立的环境实例，互不干扰。

---

### 5.2 三种训练模式

通过脚本顶部的 `TRAIN_MODE` 切换：

```python
TRAIN_MODE = "realistic_state"   # 可选："easy" / "realistic_state" / "vision"
```

#### 5.2.1 `"easy"` 模式：快速验证 SO101

- `use_object_obs=True`：观测包含 cube 的真实位置与朝向（oracle 信息）。
- `reward_shaping=True`：使用稠密奖励（接近 + 抓取 + 抬起）。
- 适合首次验证：SO101 模型是否能加载、控制器是否响应、策略能否学会 Lift。

#### 5.2.2 `"realistic_state"` 模式：面向 sim2real（默认）

- `use_object_obs=False`：策略只能看到机器人本体状态（关节角、关节速度、末端执行器力/力矩等），不能读取 cube 位姿。
- `reward_shaping=False`：只有成功抬起才给奖励，与真实任务评价一致。
- 开启动力学域随机化：摩擦、质量、阻尼、armature 等参数每回合随机扰动。
- **训练难度显著高于 easy 模式**，建议先用 easy 模式确认模型无误，再切换到此模式。

#### 5.2.3 `"vision"` 模式：纯视觉策略（预留扩展）

- 需要把 `use_camera_obs=True` 打开，并自定义图像 Wrapper 提取相机图像。
- SB3 使用 `CnnPolicy` 训练。
- 可同时开启 color/camera/lighting 域随机化。
- 脚本末尾有详细的扩展注释。

---

### 5.3 自定义机器人模型：应对真实 SO101 的组装误差

真实 SO101 组装后常出现：关节轴线不完全垂直、连杆有轻微偏移、底座不水平、夹爪对不齐等。仿真中可以通过以下方式模拟这些误差，让策略对真实机器人更鲁棒。

#### 5.3.1 修改关节轴线与位置

直接编辑 `robosuite/models/assets/robots/so101/robot.xml` 中的 joint/body 定义。例如 shoulder 关节如果有点歪：

```xml
<!-- 原始 -->
<joint axis="0 0 1" name="shoulder_pan" type="hinge" .../>

<!-- 改为带微小偏角的轴线（单位向量） -->
<joint axis="0.02 0.01 0.999" name="shoulder_pan" type="hinge" .../>
```

或者修改 body 的 `pos` 偏移量，模拟连杆安装误差：

```xml
<body name="shoulder" pos="0.040 0.001 0.062" quat="...">
```

> 建议每次只改一个参数，并在 MuJoCo 可视化中检查运动学链是否合理。

#### 5.3.2 添加 backlash（背隙）关节

真实舵机存在齿轮背隙。SO101 原始 XML 已经定义了一个 `backlash` default class，可以在两个连杆之间插入一个额外的 hinge joint：

```xml
<default class="backlash">
  <joint damping="0.01" frictionloss="0" armature="0.01"
         limited="true" range="-0.0087 0.0087"/>
</default>
```

使用方法：在 parent body 和 child body 之间插入 backlash body：

```xml
<body name="linkA" ...>
  <joint name="jointA" .../>
  <body name="linkA_backlash" pos="0 0 0">
    <joint name="jointA_backlash" class="backlash" axis="0 0 1" type="hinge"/>
    <body name="linkB" ...>
      <joint name="jointB" .../>
    </body>
  </body>
</body>
```

> 注意：每增加一个 backlash joint，自由度 +1，需要在 `so101_robot.py` 的 `set_joint_attribute` 中同步调整数组长度。

#### 5.3.3 调整初始位姿

真实 SO101 的初始姿态可能与仿真不同。修改 `so101_robot.py` 中的 `init_qpos`：

```python
@property
def init_qpos(self):
    # 5 个 arm joints 的初始角度（弧度）
    return np.array([0.0, -0.5, 1.0, -0.5, 0.0])
```

#### 5.3.4 调整阻尼、摩擦、电机惯量

在 `so101_robot.py` 的 `__init__` 中：

```python
self.set_joint_attribute(attrib="damping", values=np.array([0.5, 0.5, 0.4, 0.3, 0.3]))
self.set_joint_attribute(attrib="frictionloss", values=np.array([0.05, 0.05, 0.04, 0.03, 0.03]))
self.set_joint_attribute(attrib="armature", values=np.array([0.03, 0.03, 0.02, 0.02, 0.02]))
```

也可以直接修改 `robot.xml` 中 `sts3215` default class 的 joint 参数。

#### 5.3.5 简化碰撞体

如果仿真中碰撞不稳定或训练过慢，可以把 mesh 碰撞体替换为 primitive（box/capsule/sphere）：

```xml
<geom name="upper_arm_collision" type="capsule" group="0"
      pos="0 0 0.05" quat="1 0 0 0" size="0.025" height="0.12"/>
```

保留 visual mesh，但 collision 用 capsule 可以显著提升稳定性和速度。

---

### 5.4 自定义环境与任务

#### 5.4.1 修改 Lift 参数

在 `make_env()` 的 `suite.make()` 中可调：

```python
env = suite.make(
    "Lift",
    robots="SO101",
    table_full_size=(0.8, 0.8, 0.05),       # 桌子尺寸
    table_friction=(1.0, 5e-3, 1e-4),        # 滑动/扭转/滚动摩擦
    horizon=200,                             # 每回合最大步数
    control_freq=20,                         # 控制频率 Hz
    camera_names="agentview",                # 需要渲染的相机
    camera_heights=84,
    camera_widths=84,
)
```

#### 5.4.2 更换物体

把 cube 换成圆柱、球或其他物体，需要创建新的 Object 类或修改 `placement_initializer`。例如把 cube 换成小圆柱：

```python
from robosuite.models.objects import CylinderObject

cylinder = CylinderObject(
    name="cylinder",
    size=(0.02, 0.04),      # (半径, 高度)
    rgba=[0, 1, 0, 1],
)
```

> 注意：这需要在自定义环境类中替换 `self.cube`。

#### 5.4.3 创建全新任务

最干净的方式是继承 `Lift` 或其他 manipulation env：

```python
from robosuite.environments.manipulation.lift import Lift

class SO101Push(Lift):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reward(self, action=None):
        # 自定义奖励：把 cube 推到目标位置
        dist = np.linalg.norm(self.sim.data.body_xpos[self.cube_body_id] - self.target_pos)
        return -dist  # 示例：距离越近奖励越高

    def _check_success(self):
        return np.linalg.norm(self.sim.data.body_xpos[self.cube_body_id] - self.target_pos) < 0.05
```

注册后通过 `suite.make("SO101Push", robots="SO101", ...)` 使用。

---

### 5.5 自定义奖励函数

#### 5.5.1 继承 Lift 重写 reward

推荐方式：创建 `robosuite/environments/manipulation/so101_lift_custom.py`：

```python
import numpy as np
from robosuite.environments.manipulation.lift import Lift

class SO101LiftCustom(Lift):
    def reward(self, action=None):
        # 基础：是否成功
        if self._check_success():
            return 2.25

        reward = 0.0

        # 1. 接近奖励
        dist = self._gripper_to_target(
            gripper=self.robots[0].gripper,
            target=self.cube.root_body,
            target_type="body",
            return_distance=True,
        )
        reward += 1.0 - np.tanh(8.0 * dist)  # 比原版更平缓

        # 2. 对齐奖励：鼓励夹爪朝向与 cube 一致
        gripper_mat = self.sim.data.site_xmat[self.sim.model.site_name2id("robot0_grip_site")]
        # ... 计算朝向差异并加入奖励

        # 3. 动作惩罚：鼓励平滑、低幅值动作
        if action is not None:
            reward -= 0.001 * np.sum(action ** 2)

        return reward * self.reward_scale / 2.25 if self.reward_scale is not None else reward
```

#### 5.5.2 奖励设计技巧

- **分阶段奖励**：先接近、再对齐、再抓取、最后抬起。
- **动作正则化**：惩罚大动作、抖动，鼓励平滑运动。
- **成功奖励要足够大**：通常应至少是阶段奖励总和的 1~2 倍，避免策略 stuck 在局部最优。
- **稀疏奖励课程**：先用 `reward_shaping=True` 训练，收敛后切到 `reward_shaping=False` fine-tune。

#### 5.5.3 组合奖励示例

```python
reward = (
    0.5 * reaching_reward +      # 接近
    0.25 * alignment_reward +    # 对齐
    0.25 * grasp_reward +        # 抓取
    1.0 * lift_reward +          # 抬起（成功后给出）
    -0.001 * action_penalty      # 动作平滑
)
```

---

### 5.6 训练技巧

#### 5.6.1 课程学习（Curriculum Learning）

从简单设置开始，逐步增加难度：

1. **阶段 1**：`TRAIN_MODE="easy"`，物体初始位置固定或范围很小。
2. **阶段 2**：增大 `placement_initializer` 的随机范围。
3. **阶段 3**：切换到 `realistic_state`，关闭 object obs。
4. **阶段 4**：加入视觉观测。

可以通过环境包装器在训练过程中动态调整难度：

```python
class CurriculumWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # 根据当前成功率调整 placement 范围
        return obs, reward, terminated, truncated, info
```

#### 5.6.2 预训练 + 微调

1. 在 easy 模式下训练 50 万步，保存模型。
2. 切换到 realistic_state，加载模型继续训练：

```python
RESUME = True
MODEL_PATH = "lift_so101_sac_easy.zip"
VEC_NORMALIZE_PATH = "vec_normalize_so101_easy.pkl"
TOTAL_TIMESTEPS = 1_000_000
```

> 切换模式时需要重新创建 VecNormalize，因为观测维度可能变化（easy 模式 48 维，realistic_state 模式 38 维）。

#### 5.6.3 域随机化调参

不要一次性把所有随机化开到最大。建议顺序：

1. 只随机化 `friction`。
2. 加入 `mass`、`inertia`。
3. 加入 `damping`、`armature`。
4. 最后加入视觉随机化（如果用图像）。

扰动幅度从 5% 开始，观察策略是否还能成功，再逐步增大到 10%~20%。

#### 5.6.4 动作平滑

在策略输出后加低通滤波，减少高频抖动：

```python
class ActionSmoothWrapper(gym.ActionWrapper):
    def __init__(self, env, alpha=0.7):
        super().__init__(env)
        self.alpha = alpha
        self.last_action = np.zeros(env.action_space.shape)

    def action(self, action):
        smoothed = self.alpha * self.last_action + (1 - self.alpha) * action
        self.last_action = smoothed
        return smoothed
```

#### 5.6.5 早停与评估

不要只看训练奖励，要单独评估：

```python
from stable_baselines3.common.evaluation import evaluate_policy

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
```

如果连续多个 checkpoint 评估没有提升，可以：
- 降低学习率
- 增大 buffer size
- 调整奖励权重
- 回到更简单的课程阶段

#### 5.6.6 多随机种子实验

RL 训练方差大，建议至少跑 3 个不同 seed：

```bash
for seed in 0 1 2; do
  SEED=$seed conda run -n robosuite python train_rl_sb3_so101_realistic.py
done
```

在脚本中把 seed 传给 `suite.make(seed=SEED)` 和 SB3 的 `model.set_random_seed(SEED)`。

#### 5.6.7 记录训练视频

用 SB3 的 `VecVideoRecorder` 记录评估视频：

```python
from stable_baselines3.common.vec_env import VecVideoRecorder

env = VecVideoRecorder(
    env,
    video_folder="./videos/",
    record_video_trigger=lambda x: x % 10000 == 0,
    video_length=200,
)
```

---

### 5.7 运行命令与监控

#### 5.7.1 运行训练

```bash
conda run -n robosuite python robosuite/demos/train_rl_sb3_so101_realistic.py
```

#### 5.7.2 TensorBoard 监控

```bash
conda run -n robosuite tensorboard --logdir=./logs/sac_lift_so101_realistic/
```

核心指标：

- `rollout/ep_rew_mean`：每回合平均奖励，最核心。
- `rollout/ep_len_mean`：回合长度，变短通常表示策略更快成功。
- `train/actor_loss`、`train/critic_loss`：网络训练是否稳定。
- `train/ent_coef`：SAC 的熵系数，反映探索程度。

#### 5.7.3 可视化策略

创建独立测试脚本，开启渲染：

```python
env = suite.make(
    "Lift",
    robots="SO101",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=20,
    horizon=200,
)
```

> 测试时记得加载对应的 `VecNormalize` 参数，否则策略表现会异常。

---

## 6. 最小验证

### 6.1 单环境验证

```python
import robosuite as suite
from robosuite.wrappers import GymWrapper

env = suite.make(
    "Lift",
    robots="SO101",
    has_renderer=False,
    use_camera_obs=False,
    control_freq=20,
    horizon=200,
    reward_shaping=True,
)
env = GymWrapper(env)
obs, _ = env.reset()
print(obs.shape)  # easy 模式下 (48,)，action space (6,)
```

### 6.2 多进程环境验证

由于 `SubprocVecEnv` 会 spawn 子进程，相关代码必须放在 `if __name__ == "__main__":` 中。

---

## 7. 后续扩展建议

1. **视觉策略**：按脚本末尾注释实现 `ImageWrapper`，使用 `SAC("CnnPolicy", ...)` 训练。
2. **控制器调参**：根据真实 STS3215 舵机的 P/D 增益和力矩限制调整 `default_so101.json`。
3. **更丰富的域随机化**：在 `DomainRandomizationWrapper` 中启用 color/camera/lighting 随机化。
4. **真实数据对齐**：把 lerobot 中 SO100/SO101 的真实关节角范围、初始姿态导入 `init_qpos`。

---

## 8. 参考文件

- 机器人模型：`robosuite/models/robots/manipulators/so101_robot.py`
- 夹爪模型：`robosuite/models/grippers/so101_gripper.py`
- 夹爪 XML：`robosuite/models/assets/grippers/so101_gripper.xml`
- 机器人 XML：`robosuite/models/assets/robots/so101/robot.xml`
- 控制器配置：`robosuite/controllers/config/robots/default_so101.json`
- 训练脚本：`robosuite/demos/train_rl_sb3_so101_realistic.py`
