# 基本用法

## 运行标准化环境
**robosuite** 提供了一组用于基准测试的标准化操作任务。这些预定义的环境可以通过 `make` 函数轻松实例化。我们提供的与环境交互的 API 很简单，并且与 [OpenAI Gym](https://github.com/openai/gym/) 使用的 API 类似。下面是一个与环境交互的最简示例。

<div class="admonition warning">
<p class="admonition-title">Mac 用户请注意！</p>

希望使用默认 mjviewer 渲染器的 Mac 用户需要在 "python" 命令前加上 "mj" 前缀：`mjpython ...`
</div>

```python
import numpy as np
import robosuite as suite

# 创建环境实例
env = suite.make(
    env_name="Lift", # 可尝试其他任务，如 "Stack" 和 "Door"
    robots="Panda",  # 可尝试其他机器人，如 "Sawyer" 和 "Jaco"
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

# 重置环境
env.reset()

for i in range(1000):
    action = np.random.randn(*env.action_spec[0].shape) * 0.1
    obs, reward, done, info = env.step(action)  # 在环境中执行动作
    env.render()  # 在显示器上渲染
````

上面的脚本创建了一个带有屏幕渲染器的仿真环境，可用于可视化和定性评估。`step()` 函数接受一个 `action` 作为输入，并返回一个元组 `(obs, reward, done, info)`，其中 `obs` 是一个包含观测值 `[(name_string, np.array), ...]` 的 `OrderedDict`，`reward` 是每步获得的即时奖励，`done` 是一个布尔标志，指示 episode 是否已终止，`info` 是一个包含额外元数据的字典。

每个环境还可以配置许多其他参数。它们提供的功能包括无头渲染、获取像素观测、更改相机设置、使用奖励塑造以及添加额外的低级观测。详情请参阅 [Environment](modules/environments) 模块和 [Environment 类](simulation/environment) API。

展示 **robosuite** 各种功能的演示脚本可在[此处](demos)获取。每个脚本的用途和使用说明可在每个文件的开头找到。
