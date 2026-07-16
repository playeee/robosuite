# 基准测试

标准策略学习算法的基准测试结果。

## v1.0

我们提供了一组标准化的基准实验，作为未来实验的基线。具体来说，我们在所选任务组合（全部）上测试了 [Soft Actor-Critic](https://arxiv.org/abs/1812.05905)——一种最先进的无模型 RL 算法，使用了本体感受和物体特定观测的组合、机器人（`Panda`、`Sawyer`）以及控制器（`OSC_POSE`、`JOINT_VELOCITY`）。我们的实验是在 [rlkit](https://github.com/vitchyr/rlkit) 的扩展版本中实现和执行的，rlkit 是一个流行的基于 PyTorch 的 RL 框架和算法库。为了便于复现，我们已在[基准仓库](https://github.com/ARISE-Initiative/robosuite-benchmark)上发布了官方基准测试结果。

![benchmarking_results](../images/benchmarking/benchmarking_results.png)

所有智能体都训练了 500 个 epoch，每个 episode 500 步，并使用相同的标准化算法超参数（具体的参数值请参见上面的基准仓库）。智能体接收低维物理状态作为策略的输入。这些实验在 2 个 CPU 和 12G 显存且无 GPU 的环境下运行，每个实验大约需要两天完成。我们将每步奖励归一化为 1.0，使得每个 episode 的最大可能回报为 500。上面展示了所进行的每个任务的实验，每个实验的训练曲线显示了在五个随机种子上评估回报平均值的均值和标准差。

我们选择了两个最简单的环境，**方块抬升（Block Lifting）** 和 **开门（Door Opening）**，对操作空间控制器（`OSC_POSE`）和关节速度控制器（`JOINT_VELOCITY`）进行消融研究。我们观察到，仅控制器的选择对学习效率就有明显影响。两个机器人在使用操作空间控制器时都能更快地学会解决任务，我们推测这归功于在任务空间中加速了探索；这凸显了这种基于阻抗的控制器在改善以往受动作空间参数化限制的机器人任务性能方面的潜力。SAC 算法能够解决九个环境中的三个，包括 **方块抬升**、**开门** 和 **双臂插销入孔（Two Arm Peg-In-Hole）**，而在其他环境中进展缓慢，这些环境需要在更长的任务时间跨度内进行智能探索。对于未来的实验，我们建议使用九个环境与 Panda 机器人和操作空间控制器的组合，即上面基准测试图中 Panda (OSC) 的蓝色曲线，以进行标准化和公平的比较。

## v0.3

- 请参阅 [Surreal](http://svl.stanford.edu/assets/papers/fan2018corl.pdf) 论文获取基准测试结果。复现结果的代码可在[此处](https://github.com/SurrealAI/surreal)获取。
- 有关 [RoboTurk](https://roboturk.stanford.edu/) 数据集上的模仿学习结果，请参阅原始 [RoboTurk](https://arxiv.org/abs/1811.02790) 论文以及 [IRIS](https://arxiv.org/abs/1911.05321) 论文。
