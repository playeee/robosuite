# 仿真到真实的迁移
本页面涵盖缩小机器人仿真现实差距的随机化技术。这些技术涉及[视觉观测](#视觉)、[系统动力学](#动力学)和[传感器](#传感器)，用于提高将仿真训练模型迁移到真实世界的有效性。


## 视觉

已经充分证明，在仿真中随机化视觉在 sim2real 应用中可以发挥重要作用。**robosuite** 提供了各种 `Modder` 类来控制视觉环境的不同方面。这包括：

- `CameraModder`：用于控制相机参数的 Modder，包括视场角（FOV）和位姿
- `TextureModder`：用于控制可视物体外观的 Modder，包括纹理和材质属性
- `LightingModder`：用于控制光照参数的 Modder，包括光源属性和位姿

用户可以使用这些 Modder 直接覆盖默认仿真设置，或在仿真过程中随机化其各自的属性。我们提供了 [demo_domain_randomization.py](../demos.html#domain-randomization) 来展示所有这些 modder 在每个仿真步中应用于随机化环境。


## 动力学

为了实现合理的运行时速度，许多物理仿真平台通常必须简化底层物理模型。Mujoco 也不例外，因此，摩擦、阻尼和接触约束等许多参数无法完全捕捉真实世界的动力学。

为了更好地补偿这一点，**robosuite** 提供了 `DynamicsModder` 类，可以控制环境中每个模型的各个动力学参数。这些参数按元素组分类，下面进行了简要描述（更多信息，请参见 [Mujoco XML 参考](http://www.mujoco.org/book/XMLreference.html)）：
 
#### Opt（全局）参数
- `density`：介质（如空气）的密度
- `viscosity`：介质（如空气）的粘度

#### Body 参数
- `position`：(x, y, z) 物体相对于父物体的位置
- `quaternion`：(qw, qx, qy, qz) 物体相对于父物体的四元数
- `inertia`：(ixx, iyy, izz) 与此物体相关的惯性矩阵的对角分量
- `mass`：物体的质量

#### Geom 参数
- `friction`：(滑动、扭转、滚动) 此 geom 的摩擦值
- `solref`：(timeconst, dampratio) 此 geom 的接触求解器值
- `solimp`：(dmin, dmax, width, midpoint, power) 此 geom 的接触求解器阻抗值

#### Joint 参数
- `stiffness`：此关节的刚度
- `frictionloss`：与此关节相关的摩擦损失
- `damping`：此关节的阻尼值
- `armature`：此关节的齿轮惯性

此 `DynamicsModder` 遵循与其他 `Modder` 类相同的基本 API，并允许启用逐参数和逐组的随机化。除了随机化之外，此 modder 还可以实例化以在运行时选择性地修改值。下面给出了一个简短的示例：

```python
import robosuite as suite
from robosuite.utils.mjmod import DynamicsModder
import numpy as np

# 创建环境和 modder
env = suite.make("Lift", robots="Panda")
modder = DynamicsModder(sim=env.sim, random_state=np.random.RandomState(5))

# 定义用于轻松打印的函数
cube_body_id = env.sim.model.body_name2id(env.cube.root_body)
cube_geom_ids = [env.sim.model.geom_name2id(geom) for geom in env.cube.contact_geoms]

def print_params():
    print(f"cube mass: {env.sim.model.body_mass[cube_body_id]}")
    print(f"cube frictions: {env.sim.model.geom_friction[cube_geom_ids]}")
    print()

# 打印初始参数值
print("INITIAL VALUES")
print_params()

# 修改方块属性
modder.mod(env.cube.root_body, "mass", 5.0)                                # 使方块非常重
for geom_name in env.cube.contact_geoms:
    modder.mod(geom_name, "friction", [2.0, 0.2, 0.04])           # 大幅增加摩擦
modder.update()                                                   # 确保更改在仿真中传播

# 打印修改后的参数值
print("MODIFIED VALUES")
print_params()

# 我们还可以随时恢复默认值（原始值）
modder.restore_defaults()

# 打印恢复后的初始参数值
print("RESTORED VALUES")
print_params()
```

运行 [demo_domain_randomization.py](../demos.html#domain-randomization) 是演示此功能的另一种方法（尽管是一个极端示例）。

请注意，modder 已经有一些合理性检查，以防止可能不良/无意义的行为，例如向自由关节添加阻尼/摩擦损失，或向通常本身不刚硬的关节设置非零刚度值。


## 传感器

默认情况下，Mujoco 传感器是确定性的且无延迟，这在真实世界中通常是不切实际的假设。为了更好地缩小这一领域差距，**robosuite** 通过 [Observable](../source/robosuite.utils.html#module-robosuite.utils.observables) 类 API 提供了一个现实、可定制的接口。Observables 建模真实传感器采样，其中真值数据被采样（`sensor`），通过损坏函数（`corrupter`），最后通过滤波函数（`filter`）。此外，每个 observable 都有自己的 `sampling_rate` 和 `delayer` 函数来模拟传感器延迟。虽然在环境创建期间使用默认值实例化每个 observable，但用户可以在运行时使用 `env.modify_observable(...)` 修改这些组件中的每一个。此外，每个 observable 都被分配一个模态，并在 `env.step()` 调用期间在返回的观测字典中组合在一起。例如，如果一个环境包含相机观测和单个机器人的本体感受观测，则观测字典结构可能如下所示：

```python
{
    "frontview_image": np.array(...),    # 模态为 "image"
    "frontview_depth": np.array(...),    # 模态为 "image"
    "robot0_joint_pos": np.array(...),   # 模态为 "robot0_proprio"
    "robot0_gripper_pos": np.array(...), # 模态为 "robot0_proprio"
    "image-state": np.array(...),           # 这是所有图像观测的拼接
    "robot0_proprio-state": np.array(...),  # 这是所有 robot0_proprio 观测的拼接
}
```

请注意，为了内存效率，`image-state` 默认不返回（可以在 `robosuite/macros.py` 中切换）。

我们通过 [demo_sensor_corruption.py](../demos.html#sensor-realism) 展示如何使用 `Observable` 功能来建模传感器损坏和延迟。我们还强调，每个 `sensor`、`corrupter` 和 `filter` 函数都可以任意指定以满足最终用户的使用需求。例如，这些 observables 的一个常见用例是跟踪以高于环境步（控制）频率运行的传感器的采样值。在这种情况下，可以利用 `filter` 函数在采样实时传感器值时进行跟踪。我们在下面提供了一个展示此能力的最简脚本：

```python
import robosuite as suite
import numpy as np
from robosuite.utils.buffers import RingBuffer

# 创建环境实例
control_freq = 10
env = suite.make("Lift", robots="Panda", has_offscreen_renderer=False, use_camera_obs=False, control_freq=control_freq)

# 定义一个环形缓冲区来存储关节位置值
buffer = RingBuffer(dim=env.robots[0].robot_model.dof, length=10)

# 创建一个函数，用作关节位置 Observable 的 "filter"
# 这是一个直通操作，但每次调用时都记录该值
# 根据 Observables API，这应接受任意数值并返回相同的类型/形状
def filter_fcn(corrupted_value):
    # 记录输入值
    buffer.push(corrupted_value)
    # 返回此值（执行无操作）
    return corrupted_value

# 现在，让我们使用此 filter 函数启用关节位置 Observable
env.modify_observable(
    observable_name="robot0_joint_pos",
    attribute="filter",
    modifier=filter_fcn,
)

# 让我们提高采样率以展示 Observable 在每个环境步中多次更新的能力
obs_sampling_freq = control_freq * 4
env.modify_observable(
    observable_name="robot0_joint_pos",
    attribute="sampling_rate",
    modifier=obs_sampling_freq,
)

# 以正关节速度动作执行单个环境步
action = np.ones(env.robots[0].robot_model.dof) * 1.0
env.step(action)

# 现在我们可以分析记录了哪些值
np.set_printoptions(precision=2)
print(f"\nPolicy Frequency: {control_freq}, Observable Sampling Frequency: {obs_sampling_freq}")
print(f"Number of recorded samples after 1 policy step: {buffer._size}\n")
for i in range(buffer._size):
    print(f"Recorded value {i}: {buffer.buf[i]}")
```
