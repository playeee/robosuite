from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat


class Lift(ManipulationEnv):
    """
    本类对应单机械臂的"抬起立方体"（Lift）任务。

    任务目标：控制机械臂抓起桌面上的立方体，并使其抬升至桌面以上一定高度。
    这是 robosuite 中最基础的操作任务，常作为 RL 入门与算法验证的基准环境。

    Args:
        robots (str 或 str 列表): 指定本环境中要实例化的具体机械臂。
            例如："Sawyer" 会生成一个机械臂；["Panda", "Panda", "Sawyer"] 会生成三个机械臂。
            注意：本任务必须是单臂机器人！

        env_configuration (str): 指定机器人在环境中的摆放方式（默认为 "default"）。
            对于大多数单臂环境，该参数对机器人设置没有影响。

        controller_configs (str 或 dict 列表): 若设置，则包含用于创建自定义控制器的相关参数；
            否则使用本任务的默认控制器。若所有机器人使用相同控制器，可传入单个 dict；
            否则应传入与 "robots" 参数等长的列表。

        gripper_types (str 或 str 列表): 夹爪类型，用于从夹爪工厂实例化夹爪模型。
            默认为 "default"，即与 'robots' 指定机器人配套的默认夹爪；
            None 表示移除夹爪；其它有效值会覆盖默认夹爪。
            若所有机器人使用相同夹爪类型，可传入单个 str；否则应传入与 "robots" 等长的列表。

        base_types (None 或 str 或 str 列表): 底座类型，用于从底座工厂实例化底座模型。
            默认为 "default"，即与 'robots' 指定机器人配套的默认底座；
            None 表示无底座；其它有效值会覆盖默认底座。
            若所有机器人使用相同底座类型，可传入单个 str；否则应传入与 "robots" 等长的列表。

        initialization_noise (dict 或 dict 列表): 包含初始化噪声参数的字典。
            预期的键及其对应值类型如下：

            :`'magnitude'`: 施加到机器人给定初始关节位置上的单变量随机噪声的缩放系数。
                设为 `None` 或 0.0 表示不施加噪声。
                若噪声类型为 "gaussian"，该系数缩放所施加的标准差；
                若噪声类型为 "uniform"，该系数设定采样范围的边界。
            :`'type'`: 噪声类型，可为 "gaussian" 或 "uniform"。

            若所有机器人使用相同噪声，可传入单个 dict；否则应传入与 "robots" 等长的列表。

            :Note: 指定 "default" 会自动使用默认噪声设置。
                指定 None 会自动创建所需 dict，并将 "magnitude" 设为 0.0。

        table_full_size (3-tuple): 桌子的 x、y、z 三维尺寸。

        table_friction (3-tuple): 桌子的三个 MuJoCo 摩擦参数。

        use_camera_obs (bool): 若为 True，每次观测都包含渲染的图像。

        use_object_obs (bool): 若为 True，观测中包含物体（立方体）的信息。

        reward_scale (None 或 float): 按指定数值缩放归一化后的奖励函数。
            若为 None，环境奖励保持未归一化状态。

        reward_shaping (bool): 若为 True，使用稠密奖励（dense reward）。

        placement_initializer (ObjectPositionSampler): 若提供，将在每次 reset 时用于放置物体；
            否则默认使用 UniformRandomSampler。

        has_renderer (bool): 若为 True，在可视化窗口中渲染仿真状态，而非无头（headless）模式。

        has_offscreen_renderer (bool): 若为 True，则使用离屏渲染。

        render_camera (str): 当 `has_renderer` 为 True 时要渲染的相机名称。
            设为 'None' 会使用默认视角，该视角可被用户用鼠标拖拽/平移，便于观察。

        render_collision_mesh (bool): 若为 True，则在相机中渲染碰撞网格；否则为 False。

        render_visual_mesh (bool): 若为 True，则在相机中渲染可视化网格；否则为 False。

        render_gpu_device_id (int): 用于离屏渲染的 GPU 设备 id。
            默认为 -1，此时设备将从环境变量（GPUS 或 CUDA_VISIBLE_DEVICES）推断。

        control_freq (float): 每秒接收多少次控制信号。该值决定了每次动作输入之间经过的仿真时间。

        lite_physics (bool): 是否优化 MuJoCo 的 forward/step 调用以降低总仿真开销。
            设为 False 可保持与 robosuite <= 1.4.1 采集数据集的向后兼容性。

        horizon (int): 每个 episode 恰好持续 @horizon 个时间步。

        ignore_done (bool): 若为 True，则永不终止环境（忽略 @horizon）。

        hard_reset (bool): 若为 True，在 reset 时重新加载模型、仿真器和渲染对象；
            否则只调用 sim.reset 并重置 robosuite 内部变量。

        camera_names (str 或 str 列表): 要渲染的相机名称。
            若所有相机渲染使用相同名称，可传入单个 str；否则应传入相机列表。

            :Note: 若 @use_camera_obs 为 True，则必须至少指定一个相机。

            :Note: 要渲染所有机器人的某类相机（如 "robotview" 或 "eye_in_hand"），
                可使用 "all-{name}" 约定（如 "all-robotview"）自动渲染每个机器人相机列表中的所有图像。

        camera_heights (int 或 int 列表): 相机画面高度。
            若所有相机画面使用相同高度，可传入单个 int；否则应传入与 "camera names" 等长的列表。

        camera_widths (int 或 int 列表): 相机画面宽度。
            若所有相机画面使用相同宽度，可传入单个 int；否则应传入与 "camera names" 等长的列表。

        camera_depths (bool 或 bool 列表): 若为 True 渲染 RGB-D，否则渲染 RGB。
            若所有相机使用相同深度设置，可传入单个 bool；否则应传入与 "camera names" 等长的列表。

        camera_segmentations (None 或 str 或 str 列表 或 list of list of str): 每个相机使用的分割类型。
            有效选项为：

                `None`: 不使用分割传感器
                `'instance'`: 类-实例级别的分割
                `'class'`: 类级别的分割
                `'element'`: 单个 geom 级别的分割

            若不为 None，可指定多种分割类型。[str 列表 / str 或 None] 表示为所有相机指定
            [多种 / 一种] 分割。list of list of str 则为每个相机单独指定分割设置。

    Raises:
        AssertionError: [指定了无效的机器人数量]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # 可选值: {None, instance, class, element}
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        # =========================================================================
        # 桌面（ tabletop ）相关设置
        # =========================================================================
        # table_offset = (0, 0, 0.8) 表示桌面中心在世界坐标系 z=0.8m 处
        # 这是 Lift 任务的固定桌面高度，机械臂需将物体抬升至该高度以上 4cm 才算成功
        # =========================================================================
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # 奖励函数配置
        # reward_scale: 奖励缩放系数，None 表示不归一化
        # reward_shaping: True 用稠密奖励（dense），False 用稀疏奖励（sparse）
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # 是否在观测中包含物体的真实状态（ground-truth object states）
        # True: 观测含 cube 的位姿（oracle 信息，便于学习，但不真实）
        # False: 仅靠相机图像感知物体（更接近 sim2real）
        self.use_object_obs = use_object_obs

        # 物体放置初始化器（placement initializer）
        # 决定每次 reset 时 cube 在桌面上的随机摆放方式
        # 若为 None，则 _load_model 中会默认创建一个 UniformRandomSampler
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    def reward(self, action=None):
        """
        任务的奖励函数。

        稀疏未归一化奖励:
            - 若立方体被抬起，给一个离散奖励 2.25

        使用 reward shaping 时的未归一化分量之和:
            - Reaching（接近）: in [0, 1]，鼓励机械臂接近立方体
            - Grasping（抓取）: in {0, 0.25}，机械臂抓住立方体时非零
            - Lifting（抬起）: in {0, 1}，机械臂抬起立方体时非零

        稀疏奖励仅包含 lifting 分量。

        注意：最终奖励会被归一化并乘以 reward_scale / 2.25，
        使得最高得分等于 reward_scale。

        参数:
            action (np array): [未使用]

        返回:
            float: 奖励值
        """
        # =========================================================================
        # 【RL 核心】奖励函数设计（reward shaping）——RL 中最需要工程经验的部分
        # =========================================================================
        #
        # 奖励函数是 RL 的"指挥棒"——它定义了智能体要优化什么。
        # 设计不好会导致：
        #   - 探索困难（稀疏奖励下智能体看不到信号）
        #   - Reward hacking（策略找到漏洞获取奖励但不解决任务）
        #
        # 奖励设计两大流派：
        #
        # ┌─────────────────────────────────────────────────────────────────────┐
        # │ 1. 稀疏奖励 (sparse reward)                                          │
        # │    只有任务完成时给奖励（如本函数 _check_success 时给 2.25）           │
        # │    - 优点：定义简单、无 reward hacking、学到的策略最贴近真实目标        │
        # │    - 缺点：探索困难，RL 几乎学不到东西（特别是长 horizon 任务）        │
        # │    - 适用：短 horizon 任务、有演示辅助、或用好奇心中探索                │
        # └─────────────────────────────────────────────────────────────────────┘
        #
        # ┌─────────────────────────────────────────────────────────────────────┐
        # │ 2. 稠密奖励 (dense reward / reward shaping)                          │
        # │    每步都给与任务进度相关的奖励                                        │
        # │    - 优点：提供稠密梯度信号，大幅提升样本效率                           │
        # │    - 缺点：可能引导策略走捷径（reward hacking），且设计需要领域知识     │
        # │    - 适用：长 horizon 任务、需要快速训练                                │
        # └─────────────────────────────────────────────────────────────────────┘
        #
        # 本函数通过 self.reward_shaping 开关切换两种模式，是 robosuite 的标准实践。
        # 建议入门先用 reward_shaping=True，熟练后挑战 sparse reward。
        #
        # ───────────────────────────────────────────────────────────────────────
        # Lift 任务的奖励分解（分层奖励设计）:
        #
        #   阶段1: Reaching（接近）    → 鼓励机械臂靠近立方体
        #   阶段2: Grasping（抓取）   → 鼓励夹爪接触并抓住立方体
        #   阶段3: Lifting（抬起）    → 鼓励抬起立方体（最终目标）
        #
        # 这种"分阶段奖励"是机器人操作任务中最经典的设计模式：
        #   - 把长 horizon 任务分解为多个子目标
        #   - 每个子目标提供局部奖励信号
        #   - 子目标之间有逻辑先后顺序（必须先接近才能抓取，先抓取才能抬起）
        # ───────────────────────────────────────────────────────────────────────
        # =========================================================================
        reward = 0.0

        # =========================================================================
        # 阶段3（最终目标）：稀疏完成奖励
        # =========================================================================
        # 任务成功（cube 被抬起超过桌面 4cm）时给固定奖励 2.25
        #
        # 为什么是 2.25 这个魔法数？
        #   它是稠密奖励各分量最大值之和：1.0 (reaching) + 0.25 (grasping) + 1.0 (lifting)
        #   这样设计保证：
        #     - 成功的 reward = 2.25（最大值）
        #     - 稠密模式下即使不成功也能获得部分奖励（如 0.8 接近分）
        #     - 成功和不成功的奖励有显著差异，便于策略区分
        #
        # 注意 elif 而非 if：成功时不再计算稠密奖励，避免双重计数
        # 稀疏完成奖励（sparse completion reward）
        # =========================================================================
        if self._check_success():
            reward = 2.25

        # =========================================================================
        # 阶段1 + 阶段2：稠密奖励（reward shaping）
        # =========================================================================
        # 仅在未成功时计算稠密奖励，引导策略向成功方向探索
        # 使用稠密塑形奖励（shaping reward）
        elif self.reward_shaping:

            # ----------------------------------------------------------------------
            # 阶段1: Reaching reward（接近奖励）
            # ----------------------------------------------------------------------
            # 计算 gripper（夹爪）到 cube（立方体）的距离
            # 然后用 tanh 函数映射到 [0, 1] 区间
            #
            # 为什么用 tanh？
            #   tanh 是 RL 中常用的奖励塑形函数，有以下优点：
            #     1. 平滑可微：梯度处处存在，利于策略梯度优化
            #     2. 有界：输出在 [0, 1]，不会出现奖励爆炸
            #     3. 距离越近奖励越大，符合直觉
            #
            # 公式: reaching_reward = 1 - tanh(10 * dist)
            #   - dist=0（贴近）:   reward = 1 - 0 = 1（最大）
            #   - dist=0.1（10cm）: reward = 1 - tanh(1) ≈ 1 - 0.76 = 0.24
            #   - dist=1.0（1m）:   reward = 1 - tanh(10) ≈ 0（几乎为0）
            #
            # 系数 10.0 控制衰减速度：
            #   - 系数大：只有很近时才有显著奖励（更"挑剔"）
            #   - 系数小：远距离也能获得部分奖励（更"宽容"）
            #   调参经验：从 5-20 范围内试，观察训练曲线
            #
            # RL 重要概念——potential-based shaping:
            #   理论上，基于势函数的奖励塑形 F(s,s') = γΦ(s') - Φ(s) 不改变最优策略
            #   本函数的 reaching_reward 是简化版，可能改变最优策略，但实践中效果好
            # ----------------------------------------------------------------------
            dist = self._gripper_to_target(
                gripper=self.robots[0].gripper, target=self.cube.root_body, target_type="body", return_distance=True
            )
            reaching_reward = 1 - np.tanh(10.0 * dist)
            reward += reaching_reward

            # ----------------------------------------------------------------------
            # 阶段2: Grasping reward（抓取奖励）
            # ----------------------------------------------------------------------
            # 二值奖励：检测夹爪是否接触 cube，若接触给 0.25
            #
            # 这是"子目标完成奖励"（subgoal reward），鼓励策略学会先抓住物体
            #
            # 为什么是二值而非连续？
            #   - 抓取是离散事件（抓住或没抓住），难以用连续指标衡量
            #   - 二值奖励提供明确的"阶段性成就感"，帮助策略识别关键里程碑
            #
            # _check_grasp 通过检测夹爪 geom 与 cube geom 的接触力来判断是否抓住
            #
            # 为什么 grasping 奖励（0.25）小于 reaching 奖励（1.0）？
            #   - reaching 是过程奖励，需要持续引导
            #   - grasping 是里程碑奖励，只需触发一次即可
            #   - 比例过大可能导致策略"只想抓住而不想抬起"
            # ----------------------------------------------------------------------
            if self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube):
                reward += 0.25

        # =========================================================================
        # 奖励归一化与缩放
        # =========================================================================
        # 让最大回报 = reward_scale（默认 1.0），便于跨任务比较和调参
        #
        # 为什么需要归一化？
        #   - 不同任务奖励量级差异大（Lift 最大 2.25，NutAssembly 可能 4.0）
        #   - RL 算法（如 SAC）的熵正则项、学习率等对奖励量级敏感
        #   - 归一化后所有任务最大 reward 都是 reward_scale，调参更统一
        #
        # RL 经验：还可以进一步做 return 归一化（running std of returns）
        #   库：stable_baselines3.common.vec_env.VecNormalize
        # 若指定了 reward_scale，则按比例缩放奖励
        # =========================================================================
        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25

        return reward

    def _load_model(self):
        """
        加载 MJCF（XML）模型，并将其存入 self.model。
        """
        super()._load_model()

        # 根据桌面尺寸调整机器人底座位姿，使机械臂相对桌面处于合适的工作位置
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # 加载桌面工作空间模型（TableArena 包含桌面及其支撑结构）
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena 的原点始终设为世界坐标系原点 [0, 0, 0]
        mujoco_arena.set_origin([0, 0, 0])

        # 初始化任务关注的物体（cube 立方体）
        # 下面先定义其纹理与材质属性，再创建 BoxObject
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.cube = BoxObject(
            name="cube",
            size_min=[0.020, 0.020, 0.020],  # 立方体最小边长 2cm（注释为更早的默认值 1.5cm）
            size_max=[0.022, 0.022, 0.022],  # 立方体最大边长 2.2cm（注释为更早的默认值 1.8cm）
            rgba=[1, 0, 0, 1],
            material=redwood,
            rng=self.rng,
        )

        # 创建物体放置初始化器（placement initializer）
        # 若外部已传入，则复用之；否则创建一个在桌面中心附近小范围随机摆放的采样器
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cube)
        else:
            # UniformRandomSampler: 在 x/y ∈ [-3cm, 3cm] 范围内均匀随机摆放 cube
            #   reference_pos: 以桌面中心为参考点
            #   z_offset=0.01: cube 初始略高于桌面 1cm，避免初始穿透
            #   ensure_valid_placement=True: 确保放置位置合法（不与其它物体碰撞）
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube,
                x_range=[-0.03, 0.03],
                y_range=[-0.03, 0.03],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
                rng=self.rng,
            )

        # 将 arena（场景）、robots（机器人）、objects（物体）组合成完整的任务模型
        # ManipulationTask 会合并各部分 MJCF 并处理坐标系对齐
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cube,
        )

    def _setup_references(self):
        """
        设置对重要组件的引用（reference）。

        这里的"引用"通常是一个索引或索引列表，指向 MuJoCo 扁平化数组中对应的元素
        ——MuJoCo 正是以这种扁平化数组的形式存储物理仿真数据的。
        通过 body_name2id 等方法把"名字"映射为"索引"，后续访问位姿、速度等数据时更高效。
        """
        super()._setup_references()

        # 本环境额外需要的物体引用：记录 cube 刚体在 MuJoCo 中的 body id
        # 后续 reward / obs / success 判定都通过该 id 访问 cube 的位姿
        self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)

    def _setup_observables(self):
        """
        设置本环境使用的可观测项（observables）。若启用 use_object_obs，则创建基于物体的可观测项。

        Returns:
            OrderedDict: 将可观测项名称映射到对应 Observable 对象的字典
        """
        observables = super()._setup_observables()

        # 低层物体信息（ground-truth object state）：仅当 use_object_obs=True 时提供
        # 这些是"作弊"观测（oracle），真实 sim2real 场景下不可用，仅用于简化学习
        if self.use_object_obs:
            # 定义本组观测的模态（modality），用于观测分组与命名
            modality = "object"

            # cube 相关的可观测项：位置与朝向
            # @sensor 装饰器把普通函数注册为"传感器"，robosuite 会在每个控制步调用它
            @sensor(modality=modality)
            def cube_pos(obs_cache):
                # 返回 cube 中心在世界坐标系下的位置 [x, y, z]
                return np.array(self.sim.data.body_xpos[self.cube_body_id])

            @sensor(modality=modality)
            def cube_quat(obs_cache):
                # 返回 cube 的朝向四元数，并转换为 xyzw 约定（MuJoCo 默认是 wxyz）
                return convert_quat(np.array(self.sim.data.body_xquat[self.cube_body_id]), to="xyzw")

            sensors = [cube_pos, cube_quat]

            # 获取每个机械臂的前缀（短前缀 / 完整前缀），用于命名区分多臂场景下的观测
            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])

            # 夹爪到 cube 的相对位置传感器；每个机械臂各一个
            # 这类"相对观测"对策略学习更友好（不变于世界坐标系平移）
            sensors += [
                self._get_obj_eef_sensor(full_pf, "cube_pos", f"{arm_pf}gripper_to_cube_pos", modality)
                for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
            ]
            names = [s.__name__ for s in sensors]

            # 创建 Observable 对象并注册到字典中
            # sampling_rate 设为控制频率，表示每个控制步采样一次
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        重置仿真器的内部状态配置。
        """
        super()._reset_internal()

        # 若不是直接从 XML 加载（即非确定性 reset），则使用放置采样器重置所有物体位置
        # deterministic_reset=True 时物体位置由 MJCF 静态指定，无需运行时采样
        if not self.deterministic_reset:

            # 从放置初始化器中采样所有物体的位置与朝向
            object_placements = self.placement_initializer.sample()

            # 遍历所有物体，把采样得到的位置/朝向写回仿真器
            # set_joint_qpos 直接设置自由关节的 (pos, quat)，实现物体瞬移到目标位置
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def visualize(self, vis_settings):
        """
        除调用父类方法外，还会根据夹爪到 cube 的距离对夹爪可视化站点（site）进行着色。

        Args:
            vis_settings (dict): 将可视化关键字映射到 True/False 的字典，决定是否可视化对应组件。
                应包含 "grippers" 关键字以及其它相关选项。
        """
        # 先调用父类方法，完成基础可视化
        super().visualize(vis_settings=vis_settings)

        # 根据夹爪到 cube 的距离，对夹爪可视化站点着色（距离越近颜色越显著）
        # 这是一种直观的调试手段，便于人眼判断策略是否在接近目标
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cube)

    def _check_success(self):
        """
        检查立方体是否已被抬起。

        返回:
            bool: 若立方体被抬起则返回 True
        """
        # =========================================================================
        # 【RL 核心】终止条件（termination condition）设计
        # =========================================================================
        # 终止条件决定 episode 何时结束，直接影响策略学习：
        #
        # ┌──────────────────────────────────────────────────────────────────────┐
        # │ 设计权衡：                                                            │
        # │   太严：策略很难看到成功信号，学不到东西（探索失败）                     │
        # │   太松：策略可能学到"作弊"行为（如把物体扔到桌外也算成功）              │
        # │   刚好：策略通过合理行为才能触发成功，奖励信号清晰                       │
        # └──────────────────────────────────────────────────────────────────────┘
        #
        # 终止条件在 RL 中的作用：
        #   1. 决定 done 信号，影响 episode 的长度和 bootstrap 方式
        #   2. 配合稀疏奖励使用——成功时给大奖，触发 reset 开始新 episode
        #   3. 影响 return（累计回报）的计算：done=True 时不再累计未来奖励
        #
        # 本任务的判定逻辑非常直观：cube 中心高度 > 桌面高度 + 4cm 余量
        #
        # 为什么用 4cm 余量？
        #   - 防止"刚好贴在桌面"被误判为成功（cube 有厚度，初始位置接近桌面）
        #   - 4cm 是经验值，足够明确地表示"抬起"动作
        #   - 太小：噪声导致误判
        #   - 太大：任务变难，策略难以成功
        #
        # 为什么用 body_xpos[2]（z 坐标）而非视觉检测？
        #   - 仿真器直接提供精确位姿，无需视觉感知（视觉会引入误差和复杂性）
        #   - 计算量小，每个 step 都能检查
        #   - 在 sim2real 时才需要用视觉方法估计位姿
        #
        # ───────────────────────────────────────────────────────────────────────
        # Robosuite 各任务的 _check_success 体现了不同任务的难度差异：
        #
        #   Lift:        高度判定（最简单，只需 z 坐标）
        #   Door:        门把手角度判定（需要关节角度）
        #   PickPlace:   物体在目标位置附近（需同时判定 x,y,z 三维距离）
        #   NutAssembly: 螺母对齐到柱子（需判定位姿匹配，最复杂）
        #   ToolHang:    工具挂在钩子上（需要复杂的空间关系判定）
        #
        # 难度递增主要体现在：
        #   1. 成功条件的维度（1D 高度 → 3D 位置 → 6D 位姿）
        #   2. 容差范围（4cm → 2cm → 1cm）
        #   3. 任务 horizon（短任务 → 长任务）
        # ───────────────────────────────────────────────────────────────────────
        # =========================================================================

        # 获取 cube 中心的 z 坐标（世界坐标系下的高度）
        # self.sim.data.body_xpos 是 MuJoCo 仿真器中所有刚体的位置数组
        # [2] 表示 z 轴分量（[0]=x, [1]=y, [2]=z）
        cube_height = self.sim.data.body_xpos[self.cube_body_id][2]

        # 获取桌面的 z 坐标（桌面表面高度）
        # table_offset 是桌面在 MJCF 模型中的偏移量
        table_height = self.model.mujoco_arena.table_offset[2]

        # 判定：cube 中心高度 > 桌面高度 + 4cm 余量
        # 即 cube 必须被抬升至桌面以上一定高度才算成功（margin = 0.04m）
        return cube_height > table_height + 0.04
