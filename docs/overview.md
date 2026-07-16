# 概览

![gallery of_environments](images/gallery.png)


**robosuite** 是一个由 [MuJoCo](http://mujoco.org/) 物理引擎驱动的机器人学习仿真框架。它还提供了一套用于可复现研究的基准环境。当前版本（v1.5）支持多样化的机器人形态（包括人形机器人）、自定义机器人组合、复合控制器（包括全身控制器）、更多遥操作设备以及照片级真实感渲染。本项目是更广泛的[通过模拟环境推进机器人智能（ARISE）倡议](https://github.com/ARISE-Initiative)的一部分，旨在降低在 AI 与机器人技术交叉领域开展前沿研究的门槛。

数据驱动的算法，例如强化学习和模仿学习，为机器人学提供了强大且通用的工具。这些学习范式在深度学习新进展的推动下，在各类机器人控制问题上取得了一些令人振奋的成果。然而，可复现性的挑战和机器人硬件有限的可用性阻碍了研究进展。**robosuite** 的总体目标是为研究人员提供：

* 一套标准化的基准测试任务，用于严格的评估和算法开发；
* 一种模块化设计，为设计新的机器人仿真环境提供极大的灵活性；
* 高质量的机器人控制器实现和现成的学习算法，以降低入门门槛。

该框架最初于 2017 年底由[斯坦福视觉与学习实验室](http://svl.stanford.edu)（SVL）的研究人员作为机器人学习研究的内部工具开发。现在，它由 SVL、[UT 机器人感知与学习实验室](http://rpl.cs.utexas.edu)（RPL）以及 NVIDIA [通用具身智能体研究组](https://research.nvidia.com/labs/gear/)（GEAR）积极维护并用于机器人学研究项目。我们欢迎社区为本项目贡献代码。详情请查阅我们的[贡献指南](CONTRIBUTING.md)。

**Robosuite** 提供了模块化的 API 设计，通过程序化生成构建新的环境、机器人形态和机器人控制器。我们在下面重点介绍这些主要特性：

* **标准化任务**：一套具有高度多样性和不同复杂度、用于可复现研究的标准化操作任务，以及 RL 基准测试结果；
* **程序化生成**：模块化 API，可通过组合机器人模型、场景和参数化的 3D 物体以编程方式创建新环境和新任务。请查看我们的仓库 [robosuite_models](https://github.com/ARISE-Initiative/robosuite_models) 获取专为 robosuite 定制的额外机器人模型。
* **机器人控制器**：可选的控制器类型用于命令机器人，例如关节空间速度控制、逆运动学控制、操作空间控制和全身控制；
* **遥操作设备**：可选的遥操作设备，包括键盘、SpaceMouse、DualSense 和 MuJoCo 查看器拖放；
* **多模态传感器**：异构类型的感知信号，包括低级物理状态、RGB 相机、深度图和本体感觉；
* **人类演示**：用于收集人类演示、回放演示数据集以及利用演示数据进行学习的工具。请查看我们的姊妹项目 [robomimic](https://arise-initiative.github.io/robomimic-web/)；
* **照片级真实感渲染**：集成先进的图形工具，提供实时照片级真实感的仿真场景渲染，包括支持 NVIDIA Isaac Sim 渲染。

## 引用
如果您在出版物中使用本框架，请引用 [**robosuite**](https://robosuite.ai)：
```bibtex
@inproceedings{robosuite2020,
  title={robosuite: A Modular Simulation Framework and Benchmark for Robot Learning},
  author={Yuke Zhu and Josiah Wong and Ajay Mandlekar and Roberto Mart\'{i}n-Mart\'{i}n and Abhishek Joshi and Kevin Lin and Abhiram Maddukuri and Soroush Nasiriany and Yifeng Zhu},
  booktitle={arXiv preprint arXiv:2009.12293},
  year={2020}
}
```
