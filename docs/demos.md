# 演示展示

我们提供了一组[演示脚本](https://github.com/ARISE-Initiative/robosuite/tree/master/robosuite/demos)，用于展示 **robosuite** 的各项功能。

<div class="admonition warning">
<p class="admonition-title">Mac 用户请注意！</p>

对于这些脚本，希望使用默认 mjviewer 渲染器的 Mac 用户需要在 "python" 命令前加上 "mj" 前缀：`mjpython ...`
</div>

### 环境配置
`demo_random_action.py` 脚本是您应该首先尝试的入门演示脚本。它突出了我们仿真环境的模块化设计。它允许用户通过命令行选择一个[环境](modules/environments)、一个或多个[机器人](modules/robots)及其[控制器](modules/controllers)来创建新的仿真实例。该脚本创建一个环境实例，并使用从控制器特定动作空间中抽取的均匀随机动作来控制机器人。当前版本 **robosuite** 支持的所有环境、机器人、控制器和夹爪类型的列表分别由 `suite.ALL_ENVIRONMENTS`、`suite.ALL_ROBOTS`、`suite.ALL_PART_CONTROLLERS` 和 `suite.ALL_GRIPPERS` 定义。


### 控制器测试
`demo_control.py` 脚本演示了 **robosuite** 中每个控制器的各种功能。
对于给定的控制器，遍历每个维度并从其
中性（静止）值开始执行扰动 `test_value`，持续一定时间 "steps_per_action"，然后返回所有中性值
持续 `steps_per_rest` 时间，再继续下一个动作维度。
例如，假设 `OSC_POSE` 控制器（不含夹爪）的预期动作空间是 `(dx, dy, dz, droll, dpitch, dyaw)`，则随时间推移的测试动作序列将是：

```
***START OF DEMO***
( dx,  0,  0,  0,  0,  0, grip)     <-- x 方向平移                  持续 'steps_per_action' 步
(  0,  0,  0,  0,  0,  0, grip)     <-- 无运动（暂停）               持续 'steps_per_rest' 步
(  0, dy,  0,  0,  0,  0, grip)     <-- y 方向平移                  持续 'steps_per_action' 步
(  0,  0,  0,  0,  0,  0, grip)     <-- 无运动（暂停）               持续 'steps_per_rest' 步
(  0,  0, dz,  0,  0,  0, grip)     <-- z 方向平移                  持续 'steps_per_action' 步
(  0,  0,  0,  0,  0,  0, grip)     <-- 无运动（暂停）               持续 'steps_per_rest' 步
(  0,  0,  0, dr,  0,  0, grip)     <-- 绕 roll (x) 轴旋转           持续 'steps_per_action' 步
(  0,  0,  0,  0,  0,  0, grip)     <-- 无运动（暂停）               持续 'steps_per_rest' 步
(  0,  0,  0,  0, dp,  0, grip)     <-- 绕 pitch (y) 轴旋转          持续 'steps_per_action' 步
(  0,  0,  0,  0,  0,  0, grip)     <-- 无运动（暂停）               持续 'steps_per_rest' 步
(  0,  0,  0,  0,  0, dy, grip)     <-- 绕 yaw (z) 轴旋转            持续 'steps_per_action' 步
(  0,  0,  0,  0,  0,  0, grip)     <-- 无运动（暂停）               持续 'steps_per_rest' 步
***END OF DEMO***
```

因此，`OSC_POSE` 控制器应首先依次沿 x 方向线性移动，然后是 y 方向，再是 z 方向，然后开始依次绕其 x 轴、y 轴、z 轴旋转。有关每个控制器的概述，请参阅 [Controllers](modules/controllers) 文档。控制器预期按照其控制空间以受控的方式运行。在测试期间每个控制器的预期顺序定性描述如下：

* `OSC_POSE`：夹爪相对于全局坐标系依次沿 x、y、z 方向线性移动，然后依次绕 x 轴、y 轴、z 轴旋转
* `OSC_POSITION`：夹爪相对于全局坐标系依次沿 x、y、z 方向线性移动
* `IK_POSE`：夹爪相对于局部机器人末端执行器坐标系依次沿 x、y、z 方向线性移动，然后依次绕 x 轴、y 轴、z 轴旋转
* `JOINT_POSITION`：机器人关节以受控方式依次移动
* `JOINT_VELOCITY`：机器人关节以受控方式依次移动
* `JOINT_TORQUE`：与其他控制器不同，关节力矩控制器预期表现得相当迟缓，因为"控制器"实际上只是 mujoco 执行器直接力矩控制的封装。因此，0 力矩的"中性"值在机器人具有非零速度时不能保证机器人稳定！


### 域随机化

`demo_domain_randomization.py` 脚本展示了我们的域随机化功能。通过使用 [DomainRandomizationWrapper](source/robosuite.wrappers) 包装环境可以轻松启用域随机化。目前，随机化功能主要关注视觉变化，包括颜色、纹理和相机视点。

![collection of grippers](images/domain_randomization.png)


### 传感器真实性
`demo_sensor_corruption.py` 脚本展示了如何使用 [Observable](modules/sensors) 功能来建模传感器损坏和延迟。[Observable](modules/sensors) 类提供了易于使用的接口，用于模拟真实世界传感器噪声、可变传感器延迟和真实传感器的采样频率。这些技术有助于提高在 robosuite 上训练的策略的泛化能力和鲁棒性，促进向真实硬件的 sim2real 迁移。例如，请参阅 [Zhu et al. RSS'2018](https://arxiv.org/abs/1802.09564) 的附录 B 和 [Tan et al. RSS'2018](https://arxiv.org/abs/1804.10332) 的第 V 节。


### 夹爪选择
`demo_gripper_selection.py` 脚本展示了如何为环境选择夹爪。这由 `gripper_type` 关键字参数控制。所有夹爪的集合由全局变量 `robosuite.ALL_GRIPPERS` 定义。

![collection of grippers](images/gripper_collection.png)

### 夹爪交互与程序化生成
`demo_gripper_interaction.py` 脚本说明了将夹爪导入场景并使其
与带执行器的物体交互的过程。它还展示了如何使用 MJCF 实用工具函数的建模 API 程序化生成场景。


### 轨迹回放
`demo_collect_and_playback_data.py` 展示了如何使用 [DataCollectionWrapper](source/robosuite.wrappers) 包装器记录机器人 rollout 轨迹数据并回放。该包装器将轨迹中的环境状态记录并存储为 `.npz` 格式的临时文件（默认路径：`/tmp`）。回放时，它从磁盘加载存储的状态并将模拟器重置为这些状态。示例：
```
$ python demo_collect_and_playback_data.py --environment Door
```

### OpenAI Gym 风格 API
此 `demo_gym_functionality.py` 脚本展示了如何将环境适配为兼容 [OpenAI Gym](https://gym.openai.com/) 风格的 API。当使用需要支持这些 API 的学习流水线时，这非常有用。例如，这可与 [OpenAI Baselines](https://github.com/openai/baselines) 一起使用 RL 训练智能体。我们基于 OpenAI Gym 文档[Getting Started with Gym](https://gym.openai.com/docs/)部分中的一些代码片段编写了此脚本。以下片段用于演示 OpenAI Gym 的基本功能。

```python
import gym
env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
```

为了将 **robosuite** API 适配为兼容 OpenAI Gym 风格，此脚本演示了如何使用 [GymWrapper](source/robosuite.wrappers) 轻松实现这一点。


### 遥操作
`demo_device_control.py` 脚本展示了如何使用[控制设备](modules/devices)（如键盘或 SpaceMouse）遥操作机器人。用户输入设备可以通过 `--device` 参数设置，有以下两个选项：

* **键盘**
    我们使用键盘控制机器人的末端执行器。
    键盘通过各种按键提供 6 自由度控制命令。

    **注意：**
        在 macOS 上运行此脚本，必须使用 root 权限运行。

* **SpaceMouse**
    我们使用来自 [3Dconnexion](https://www.3dconnexion.com/spacemouse_wireless/en/) 的 SpaceMouse 3D 鼠标控制机器人的末端执行器。该鼠标提供 6 自由度控制命令。

    我们使用了 SpaceMouse Wireless。下面的论文使用了同一设备
    为模仿学习收集人类演示。

    ```text
    Reinforcement and Imitation Learning for Diverse Visuomotor Skills
    Yuke Zhu, Ziyu Wang, Josh Merel, Andrei Rusu, Tom Erez, Serkan Cabi, Saran Tunyasuvunakool,
    János Kramár, Raia Hadsell, Nando de Freitas, Nicolas Heess
    RSS 2018
    ```

    **注意：**
        当前实现仅支持 macOS（可以添加 Linux 支持）。
        运行脚本前请下载并安装[驱动](https://www.3dconnexion.com/service/drivers.html)。

* **DualSense**
    我们使用来自 [DualSense](https://www.playstation.com/en-us/accessories/dualsense-wireless-controller/) 的 DualSense 摇杆控制机器人的末端执行器。该摇杆提供 6 自由度控制命令。

    **注意：**
        请确保 `hidapi` 可以在您的计算机上检测到 DualSense。在 Linux 上，您可以在 `/etc/udev/rules.d` 中添加 udev 规则以无需 root 权限即可访问设备。有关规则内容，请参阅 [game-device-udev](https://codeberg.org/fabiscafe/game-devices-udev)。

* **Mujoco GUI**
        Mujoco GUI 提供了图形用户界面，用于查看和与 mujoco 仿真交互。我们使用 GUI 和鼠标拖放 mocap 体，其
        位姿由控制器跟踪。更具体地说，一旦从运行 `python demo_device_control.py` 加载了 mujoco GUI，您首先需要按 <Tab> 键进入交互式 mujoco 查看器状态。然后，您应该双击
        一个 mocap 体。最后，要拖动 mocap 体，您可以按 <Ctrl> 或 <Shift> 键来平移或旋转 mocap 体。对于 Mac 用户，您需要使用 `mjpython demo_device_control.py`。有关更多详细信息，请参阅 [mujoco](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer) 的说明。


此外，`--pos_sensitivity` 和 `--rot_sensitivity` 提供相对增益，用于增大 / 减小用户输入
设备的灵敏度。


此外，请使用以下参数选择环境细节：

* `--environment`：要执行的任务，例如 `Lift`、`TwoArmPegInHole`、`NutAssembly` 等。

* `--robots`：执行任务所用的机器人，例如 `Tiago`、`Panda`、`GR1`、`Sawyer` 等。请注意，环境包含完整性检查，因此 `TwoArm...` 环境将不接受具有单个单臂机器人的配置。

* `--config`：仅适用于且仅应在 `TwoArm...` 环境中指定。指定当输入两个机器人时
        任务所需的机器人配置。选项为 {`parallel` 和 `opposed`}

    * `parallel`：设置环境使两个机器人并排
                站立，面向同一方向。期望在 `--robots` 参数中
                指定一个由机器人名称组成的 2 元组。

    * `opposed`：设置环境使两个机器人相对
                而立，从相反方向面对面。期望在 `--robots` 参数中
                指定一个由机器人名称组成的 2 元组。


示例：
* 普通单臂环境：
```
$ python demo_device_control.py --environment PickPlaceCan --robots Sawyer
```
* 双臂双手协调环境：
```
$ python demo_device_control.py --environment TwoArmLift --robots Tiago
```
* 双臂多单臂机器人环境：
```
$ python demo_device_control.py --environment TwoArmLift --robots Sawyer Sawyer --config parallel
```

### 视频录制
`demo_video_recording.py` 脚本展示了如何使用 `imageio` 库录制机器人 rollout 视频。此脚本使用离屏渲染。这对于生成机器人策略行为的定性视频非常有用。生成的视频为 mp4 格式。示例：
```sh
$ python demo_video_recording.py --environment Lift --robots Panda
```

### 渲染选项
`demo_renderers.py` 脚本展示了如何在仿真环境中使用不同的渲染器。我们当前版本支持默认的 MuJoCo 渲染器。有关这些渲染器的更多信息，请参阅 [Renderer](modules/renderers) 模块。示例：
```sh
$ python demo_renderers.py --renderer default
```
`--renderer` 标志可以设置为 `mujoco` 或 `default`

### 导出到 USD
导出到 USD 允许用户在外部渲染器（如 NVIDIA Omniverse 和 Blender）中渲染 **robosuite** 轨迹。为了导出到 USD，您必须安装导出器所需的依赖项。
```sh
$ pip install usd-core pillow tqdm
```
安装依赖项后，可以通过 `from robosuite.utils.usd import exporter` 导入 USD 导出器。`exporter` 模块中的 `USDExporter` 类负责导出与 **robosuite** 轨迹相关的所有必要资产和 USD 文件。 

首先，实例化一个 **robosuite** 环境。每个环境都关联一个 MjModel 和 MjData 实例。可以使用以下方式检索这些属性：
```python
model = env.sim.model._model
data = env.sim.data._data
```
`model` 和 `data` 都被 USD 导出器使用。定义 robosuite 环境后，使用以下参数创建一个 `USDExporter` 对象。

* `model`（必需）：一个 MjModel 实例。
* `max_geom`：可选整数，指定可在同一场景中渲染的
geom 的最大数量。如果为 None，将基于模型中可渲染 geom 的
估计最大数量自动
选择。
* `output_directory_name`：用于存储 USD 渲染器生成的输出帧
和资产的根目录名称。
和资产的根目录名称。
* `light_intensity`：外部渲染器中灯光的默认强度。
* `shareable`：使用资产的相对路径而不是绝对路径，以允许
文件在不同用户之间共享。
* `online`：如果使用 USD 导出器进行在线渲染，则设置为 true。当使用 Isaac Sim 渲染时，此值
设置为 true。如果 online 设置为 true，shareable 必须为 false。
* `framerate`：渲染时导出场景的帧率
* `camera_names`：mujoco 模型中定义的要渲染的固定相机列表。
* `stage`：用于将场景中物体添加到的预定义舞台。
* `verbose`：决定是否打印更新。

`USDExporter` 改编自 [MuJoCo](https://github.com/google-deepmind/mujoco)。为了在输出的 USD 轨迹中添加新帧，请调用 `exporter` 模块中的 `update_scene`。 

```python
exp = exporter.USDExporter(model=model, output_directory_name="usd_demo")
exp.update_scene(data)
```

这将使用仿真中的当前位姿更新场景中的所有 geom。要保存 USD 轨迹，请使用 `save_scene` 方法。

```python
exp.save_scene(filetype="usd")
```

用户可以将场景保存为 .usd、.usda 或 .usdc 文件。有关 USD 渲染器的更全面示例，请参阅 [`demo_usd_export.py`]() 脚本。此演示允许用户使用设备（即键盘或 spacemouse）遥操作机器人，并将收集的轨迹保存为 USD 文件。 
