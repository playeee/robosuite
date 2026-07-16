## 构建您自己的环境

**robosuite** 在创建您自己的环境方面提供了极大的灵活性。一个[任务](../modeling/task)通常涉及一个[机器人](../modeling/robot_model)的参与，其末端执行器为[夹爪](../modeling/robot_model.html#gripper-model)，一个[场景](../modeling/arena)（工作空间），以及机器人交互的[物体](../modeling/object_model)。有关我们设计架构的详细概述，请查看模块中的[概览](../modules/overview)页面。我们的建模 API 提供了将这些模块化元素组合成场景的方法，场景可以加载到 MuJoCo 中进行仿真。要构建您自己的环境，我们建议您查看[环境类](../simulation/environment)，这些类已使用这些 API 定义机器人环境和任务，以及我们标准化环境的[源代码](https://github.com/ARISE-Initiative/robosuite/tree/master/robosuite/environments)。下面我们将逐步演示如何使用我们的 API 构建一个新的桌面操作环境。

**步骤 1：创建世界。** 所有 mujoco 物体定义都存储在 xml 中。我们创建一个 [MujocoWorldBase](../source/robosuite.models) 类来完成此操作。
```python
from robosuite.models import MujocoWorldBase

world = MujocoWorldBase()
```

**步骤 2：创建机器人。** 存储机器人 xml 的类可以按如下方式创建。
```python
from robosuite.models.robots import Panda

mujoco_robot = Panda()
```
我们可以通过创建夹爪实例并在机器人上调用 add_gripper 方法来为机器人添加夹爪。
```python
from robosuite.models.grippers import gripper_factory

gripper = gripper_factory('PandaGripper')
mujoco_robot.add_gripper(gripper)
```
要将机器人添加到世界中，我们将机器人放置到所需位置并将其合并到世界中
```python
mujoco_robot.set_base_xpos([0, 0, 0])
world.merge(mujoco_robot)
```

**步骤 3：创建桌子。** 我们可以初始化 [TableArena](../source/robosuite.models.arenas) 实例，该实例创建桌子和地板平面
```python
from robosuite.models.arenas import TableArena

mujoco_arena = TableArena()
mujoco_arena.set_origin([0.8, 0, 0])
world.merge(mujoco_arena)
```

**步骤 4：添加物体。** 有关 `MujocoObject` 的详细信息，请参阅关于 [MujocoObject](../modeling/object_model) 的文档，我们可以创建一个球并将其添加到世界中。
```python
from robosuite.models.objects import BallObject
from robosuite.utils.mjcf_utils import new_joint

sphere = BallObject(
    name="sphere",
    size=[0.04],
    rgba=[0, 0.5, 0.5, 1]).get_obj()
sphere.set('pos', '1.0 0 1.0')
world.worldbody.append(sphere)
```

**步骤 5：运行仿真。** 一旦我们创建了物体，就可以通过运行以下命令获取 `mujoco.MjModel` 模型
```python
model = world.get_model(mode="mujoco")
```
这是一个 `MjModel` 实例，随后可用于仿真。例如，
```python
import mujoco

data = mujoco.MjData(model)
while data.time < 1:
    mujoco.mj_step(model, data)
```
