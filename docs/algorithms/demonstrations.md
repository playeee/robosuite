# 人类演示

## 收集人类演示

我们提供了遥操作工具，允许用户使用输入设备（如键盘、[SpaceMouse](https://www.3dconnexion.com/spacemouse_compact/en/)、[DualSense](https://www.playstation.com/en-us/accessories/dualsense-wireless-controller/) 和 mujoco-gui）控制机器人。此功能使我们能够收集用于学习的人类演示数据集。我们提供了一个示例脚本来演示如何收集演示。我们的 [collect_human_demonstrations](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/scripts/collect_human_demonstrations.py) 脚本接受以下参数：

- `directory:` 用于存储收集的演示 pickle 文件的文件夹路径
- `environment:` 您希望为其收集演示的环境名称
- `device:` "keyboard"、"spacemouse"、"dualsense" 或 "mjgui" 之一
- `renderer:` Mujoco 的内置交互式查看器（mjviewer）或 OpenCV 查看器（mujoco）
- `camera:` 传递多个相机名称以启用多个视图。请注意，使用多个视图时必须启用 "mujoco" 渲染器，不支持 "mjviewer"。

有关如何使用设备的详细信息，请参阅[设备页面](https://robosuite.ai/docs/modules/devices.html)。

## 回放人类演示

我们包含了一个示例脚本，演示如何加载和回放演示。我们的 [playback_demonstrations_from_hdf5](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/scripts/playback_demonstrations_from_hdf5.py) 脚本从演示 pickle 文件中随机选择演示 episode 并回放它们。


## 现有数据集

我们在 `models/assets/demonstrations` 中为每个任务提供了一些示例演示。


## 收集演示的结构

每组演示都收集为 `demo.hdf5` 文件。`demo.hdf5` 文件的结构如下。

- data（组）

  - date（属性）- 收集日期

  - time（属性）- 收集时间

  - repository_version（属性）- 收集时使用的仓库版本

  - env（属性）- 收集演示的环境名称

  - demo1（组）- 第一个演示的组（每个演示都有一个组）

    - model_file（属性）- 对应于 MJCF mujoco 模型的 xml 字符串

    - states（数据集）- 按时间排序的扁平化 mujoco 状态

    - actions（数据集）- 按时间排序的环境动作

  - demo2（组）- 第二个演示的组

    ... 

    （以此类推）

存储 mujoco 状态而非原始观测的原因是为了便于在后处理步骤中检索不同类型的观测。这也节省了磁盘空间（图像数据集要大得多）。


## 使用演示进行学习

[robomimic](https://arise-initiative.github.io/robomimic-web/) 框架使得使用您自己的[使用 robosuite 收集的数据集](https://arise-initiative.github.io/robomimic-web/docs/introduction/datasets.html#robosuite-hdf5-datasets)训练策略变得容易。该框架还包含许多有用的示例，说明如何将 hdf5 数据集集成到您自己的学习流水线中。

robosuite 仓库还有一些实用工具，可用于使用演示改变训练 RL 策略时 episode 的初始状态分布——这在[几项](https://arxiv.org/abs/1802.09564)[先前](https://arxiv.org/abs/1807.06919)[工作](https://arxiv.org/abs/1804.02717)中已被证明有效。例如，我们提供了一个通用工具，用于设置各种类型的学习课程，规定在执行环境重置时如何从演示 episode 中采样。更多信息请参见 `DemoSamplerWrapper` 类。

## 警告
我们已验证，确定性动作回放专门在*最初收集演示的同一台机器*上回放演示时有效。然而，这意味着确定性动作回放不能保证（实际上非常不可能）跨平台甚至在相同操作系统的不同机器上工作。

虽然动作回放轨迹与原始收集的状态轨迹即使不完全相同也相当相似，但它们确实会随时间漂移，不应依赖它们来准确复制演示。相反，我们建议直接设置状态以重现收集的轨迹，如 [playback_demonstrations_from_hdf5](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/scripts/playback_demonstrations_from_hdf5.py) 所示。
