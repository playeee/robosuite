from collections import OrderedDict
from copy import deepcopy

import numpy as np

import robosuite.macros as macros
from robosuite.environments.base import MujocoEnv
from robosuite.robots import ROBOT_CLASS_MAPPING
from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
from robosuite.utils.observables import Observable, sensor


class RobotEnv(MujocoEnv):
    """
    在 Mujoco 中初始化一个机器人环境。

    本类继承自 `MujocoEnv`，在通用环境基础上增加了机器人相关逻辑：机器人实例化、
    控制器配置、相机观测(包括 RGB、深度、分割图)、初始化噪声等，是后续具体任务
    环境(如 manipulation)的基类。

    Args:
        robots: 要在本环境内实例化的具体机器人规格(可传入单个或多个)。

        env_configuration (str): 指定机器人在环境中的放置方式，默认 "default"，
            由子类自行解释其含义。

        controller_configs (str or list of dict): 若设置，则包含创建自定义控制器的相关参数；
            否则使用该任务的默认控制器。若所有机器人使用相同控制器，可传入单个 dict；
            否则应传入长度与 "robots" 一致的列表。

        mount_types (str or list of str): mount 类型，用于从 mount factory 实例化 mount 模型。
            默认 "default"，即与机器人规格关联的默认 mount；"NullMount" 表示不使用 mount；
            其它合法值会覆盖默认 mount。可传入单个 str，或长度与 "robots" 一致的列表。

        initialization_noise (dict or list of dict): 包含初始化噪声参数的字典/字典列表。
            预期键与对应值类型如下：

            :`'magnitude'`: 对机器人各初始关节位置施加的单变量随机噪声缩放因子。
                设为 `None` 或 0.0 表示不加噪声。若为 "gaussian"，该值缩放标准差；
                若为 "uniform"，该值设置采样范围边界。
            :`'type'`: 噪声类型，可选 "gaussian" 或 "uniform"。

            可传入单个 dict 用于所有机器人，或长度与 "robots" 一致的列表。

            :Note: 传入 "default" 会自动使用默认噪声设置；传入 None 会创建
                magnitude=0.0 的字典。

        use_camera_obs (bool): 若为 True，每次观测都会包含渲染的图像。

        has_renderer (bool): 若为 True，在可视化器中渲染仿真状态，而非无界面模式。

        has_offscreen_renderer (bool): 是否使用离屏渲染。

        render_camera (str or list of str): 当 `has_renderer` 为 True 时要渲染的相机名。
            设为 'None' 会使用默认视角，可由用户用鼠标拖拽/平移。传入字符串列表会从多个视角渲染。

        render_collision_mesh (bool): True 表示在相机中渲染碰撞网格。

        render_visual_mesh (bool): True 表示在相机中渲染可视化网格。

        render_gpu_device_id (int): 用于离屏渲染的 GPU 设备 id。默认 -1，从环境变量
            (GPUS 或 CUDA_VISIBLE_DEVICES) 推断。

        control_freq (float): 每秒接收多少次控制信号。这决定了每次动作输入之间经过的仿真时间。

        lite_physics (bool): 是否优化 mujoco 的 forward/step 调用以降低仿真开销。
            设为 False 以保持与 robosuite <= 1.4.1 数据集的向后兼容。

        horizon (int): 每个回合固定持续 @horizon 个时间步。

        ignore_done (bool): 若为 True，永不终止环境(忽略 @horizon)。

        hard_reset (bool): 若为 True，在 reset 时重新加载模型/sim/渲染对象；否则只调用
            sim.reset() 并重置 robosuite 内部变量。

        load_model_on_init (bool): 若为 True，在 __init__ 中加载并初始化模型和渲染器；
            否则在第一次 reset() 时初始化。

        camera_names (str or list of str): 要渲染的相机名。可传入单个 str 用于所有相机，
            或传入相机列表。

            :Note: 若 @use_camera_obs 为 True，至少要指定一个相机。

            :Note: 要渲染所有机器人某类型相机(如 "robotview" 或 "eye_in_hand")，
                可使用 "all-{name}" 约定(如 "all-robotview")自动渲染每个机器人相机列表中的对应图像。

        camera_heights (int or list of int): 相机帧高度。可传入单个 int 或长度与 "camera_names" 一致的列表。

        camera_widths (int or list of int): 相机帧宽度。可传入单个 int 或长度与 "camera_names" 一致的列表。

        camera_depths (bool or list of bool): True 表示渲染 RGB-D，否则仅 RGB。可传入单个 bool 或列表。

        camera_segmentations (None or str or list of str or list of list of str): 每个相机使用的分割类型。
            可选值：

                `None`: 不使用分割传感器
                `'instance'`: 类实例级分割
                `'class'`: 类级分割
                `'element'`: 逐 geom 级分割

            非 None 时可指定多种分割类型。`[str 列表 / 单个 str 或 None]` 表示为所有相机指定
            `[多种 / 一种]` 分割；`list of list of str` 表示按相机单独指定分割设置。

        robot_configs (list of dict): 由子类初始化器设置的每机器人配置。

        seed (int): 环境随机种子。默认 None，即不设定。

    Raises:
        ValueError: [相机观测需要离屏渲染器]
        ValueError: [使用相机观测时必须指定相机名]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        base_types="default",
        controller_configs=None,
        initialization_noise=None,
        use_camera_obs=True,
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
        load_model_on_init=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        robot_configs=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        # 首先，校验输入的机器人数量是否正确
        self.env_configuration = env_configuration
        self._check_robot_configuration(robots)

        # Robot: 统一成 list 形式存储
        robots = list(robots) if type(robots) is list or type(robots) is tuple else [robots]
        self.num_robots = len(robots)
        self.robot_names = robots
        self.robots = self._input2list(None, self.num_robots)
        self._action_dim = None

        # Robot base: 把 base_types 也统一成长度为 num_robots 的列表
        base_types = self._input2list(base_types, self.num_robots)

        # Composite Controller: 把控制器配置统一成长度为 num_robots 的列表
        controller_configs = self._input2list(controller_configs, self.num_robots)

        # Initialization Noise: 把初始化噪声参数也统一成长度为 num_robots 的列表
        initialization_noise = self._input2list(initialization_noise, self.num_robots)

        # 观测相关：object_obs 表示真值观测，camera_obs 表示图像观测
        self.use_camera_obs = use_camera_obs

        # 相机/渲染相关设置
        self.has_offscreen_renderer = has_offscreen_renderer
        self.camera_names = (
            list(camera_names) if type(camera_names) is list or type(camera_names) is tuple else [camera_names]
        )
        self.num_cameras = len(self.camera_names)

        # 把各相机维度参数也统一成列表形式
        self.camera_heights = self._input2list(camera_heights, self.num_cameras)
        self.camera_widths = self._input2list(camera_widths, self.num_cameras)
        self.camera_depths = self._input2list(camera_depths, self.num_cameras)
        self.camera_segmentations = self._input2list(camera_segmentations, self.num_cameras)
        # 需要对 camera_segmentations 做更细致的解析，因为它可能是嵌套列表
        seg_is_nested = False
        for i, camera_s in enumerate(self.camera_segmentations):
            if isinstance(camera_s, list) or isinstance(camera_s, tuple):
                seg_is_nested = True
                break
        camera_segs = deepcopy(self.camera_segmentations)
        for i, camera_s in enumerate(self.camera_segmentations):
            if camera_s is not None:
                self.camera_segmentations[i] = self._input2list(camera_s, 1) if seg_is_nested else deepcopy(camera_segs)

        # 相机渲染的合理性检查
        if self.use_camera_obs and not self.has_offscreen_renderer:
            raise ValueError("Error: Camera observations require an offscreen renderer!")
        if self.use_camera_obs and self.camera_names is None:
            raise ValueError("Must specify at least one camera name when using camera obs")

        # 机器人配置——合并子类传入的 robot_config
        if robot_configs is None:
            robot_configs = [{} for _ in range(self.num_robots)]
        self.robot_configs = [
            dict(
                **{
                    "composite_controller_config": controller_configs[idx],
                    "base_type": base_types[idx],
                    "initialization_noise": initialization_noise[idx],
                    "control_freq": control_freq,
                    "lite_physics": lite_physics,
                },
                **robot_config,
            )
            for idx, robot_config in enumerate(robot_configs)
        ]

        # 调用父类初始化
        super().__init__(
            has_renderer=has_renderer,
            has_offscreen_renderer=self.has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            load_model_on_init=load_model_on_init,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    def visualize(self, vis_settings):
        """
        在调用父类方法的基础上，额外可视化机器人。

        Args:
            vis_settings (dict): 可视化关键字到 True/False 的映射，决定对应组件是否可视化。
                应包含 "robots" 关键字以及其它相关选项。
        """
        # 先调用父类方法
        super().visualize(vis_settings=vis_settings)
        # 逐个机器人独立可视化
        for robot in self.robots:
            robot.visualize(vis_settings=vis_settings)

    @property
    def _visualizations(self):
        """
        本环境可用的可视化关键字。

        Returns:
            set: 本环境中可单独可视化的所有组件
        """
        vis_set = super()._visualizations
        vis_set.add("robots")
        return vis_set

    @property
    def action_spec(self):
        """
        本环境的动作空间 (low, high)。

        Returns:
            2-tuple:
                - (np.array) 最小动作值
                - (np.array) 最大动作值
        """
        low, high = [], []
        for robot in self.robots:
            lo, hi = robot.action_limits
            low, high = np.concatenate([low, lo]), np.concatenate([high, hi])
        return low, high

    @property
    def action_dim(self):
        """
        动作空间维度。

        Returns:
            int: 动作空间维度
        """
        return self._action_dim

    @staticmethod
    def _input2list(inp, length):
        """
        辅助函数：把单个值或列表统一成长度为 @length 的列表。

        Args:
            inp (None or str or list): 要转换为列表的输入
            length (int): 将输入广播到的列表长度

        Returns:
            list: 输入 @inp 转换为长度为 @length 的列表
        """
        # 如有必要则转换为 list
        return list(inp) if type(inp) is list or type(inp) is tuple else [inp for _ in range(length)]

    def _load_model(self):
        """
        加载 xml 模型并放到 self.model 中。
        在父类基础上额外加载机器人。
        """
        super()._load_model()

        # 加载机器人
        self._load_robots()

    def _setup_references(self):
        """
        设置对重要组件的引用(reference)。

        reference 通常是一个索引或索引列表，指向 Mujoco 扁平存储的物理仿真数据数组
        中的对应元素。
        """
        super()._setup_references()

        # 设置机器人专属引用(注意：需要先对机器人 reset_sim 再 setup_references)
        for robot in self.robots:
            robot.reset_sim(self.sim)
            robot.setup_references()

    def _setup_observables(self):
        """
        设置本环境所用的 observables。

        会遍历所有机器人，收集它们的 observables 加入到自动生成的 observables 字典中；
        若启用了相机观测，还会为每个相机创建对应的图像/深度/分割 observable。

        Returns:
            OrderedDict: 从观测名映射到对应 Observable 对象的字典
        """
        observables = super()._setup_observables()
        # 遍历所有机器人，收集它们的 observables，归类为 proprioception 模态
        for robot in self.robots:
            robot_obs = robot.setup_observables()
            observables.update(robot_obs)

        # 若使用相机观测，则遍历相机并更新观测
        if self.use_camera_obs:
            # 构造传感器信息
            sensors = []
            names = []
            for (cam_name, cam_w, cam_h, cam_d, cam_segs) in zip(
                self.camera_names,
                self.camera_widths,
                self.camera_heights,
                self.camera_depths,
                self.camera_segmentations,
            ):

                # 为每个相机构造对应的传感器
                cam_sensors, cam_sensor_names = self._create_camera_sensors(
                    cam_name, cam_w=cam_w, cam_h=cam_h, cam_d=cam_d, cam_segs=cam_segs, modality="image"
                )
                sensors += cam_sensors
                names += cam_sensor_names

            # 若任一相机的 segmentation 不为 None，则把所有 site 缩小以避免它们出现在分割图中
            # (这是一个 hacky 的实现)
            if not all(seg is None for seg in self.camera_segmentations):
                self.sim.model.site_size[:, :] = 1.0e-8

            # 为这些相机制作 observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _create_camera_sensors(self, cam_name, cam_w, cam_h, cam_d, cam_segs, modality="image"):
        """
        为给定相机制造传感器的辅助函数。将其抽成独立函数调用，是为了避免在 _setup_observables()
        调用过程中出现局部函数命名冲突。

        Args:
            cam_name (str): 要为之创建传感器的相机名
            cam_w (int): 相机宽度
            cam_h (int): 相机高度
            cam_d (bool): 是否同时创建深度传感器
            cam_segs (None or list): 要使用的分割类型，每项可为：
                `None`: 不使用分割传感器
                `'instance'`: 类实例级分割
                `'class'`: 类级分割
                `'element'`: 逐 geom 级分割

            modality (str): 赋给所有传感器的模态
        Returns:
            2-tuple:
                sensors (list): 该相机的传感器数组
                names (list): 对应的观测名数组
        """
        # 确保使用正确的图像坐标约定(opencv / opengl)
        convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]

        # 构造传感器信息
        sensors = []
        names = []

        # 在 dict 中添加相机 observables
        rgb_sensor_name = f"{cam_name}_image"
        depth_sensor_name = f"{cam_name}_depth"
        segmentation_sensor_name = f"{cam_name}_segmentation"

        @sensor(modality=modality)
        def camera_rgb(obs_cache):
            img = self.sim.render(
                camera_name=cam_name,
                width=cam_w,
                height=cam_h,
                depth=cam_d,
            )
            if cam_d:
                # 若开启了深度，render 会返回 (rgb, depth)
                rgb, depth = img
                # 把深度图缓存起来供深度传感器使用
                obs_cache[depth_sensor_name] = np.expand_dims(depth[::convention], axis=-1)
                return rgb[::convention]
            else:
                return img[::convention]

        sensors.append(camera_rgb)
        names.append(rgb_sensor_name)

        if cam_d:
            # 深度传感器：从 obs_cache 读取深度，若不存在则返回零图
            @sensor(modality=modality)
            def camera_depth(obs_cache):
                return obs_cache[depth_sensor_name] if depth_sensor_name in obs_cache else np.zeros((cam_h, cam_w, 1))

            sensors.append(camera_depth)
            names.append(depth_sensor_name)

        if cam_segs is not None:
            # 定义分割使用的映射
            for cam_s in cam_segs:
                seg_sensor, seg_sensor_name = self._create_segementation_sensor(
                    cam_name=cam_name,
                    cam_w=cam_w,
                    cam_h=cam_h,
                    cam_s=cam_s,
                    seg_name_root=segmentation_sensor_name,
                    modality=modality,
                )

                sensors.append(seg_sensor)
                names.append(seg_sensor_name)

        return sensors, names

    def _create_segementation_sensor(self, cam_name, cam_w, cam_h, cam_s, seg_name_root, modality="image"):
        """
        为给定相机制造分割传感器的辅助函数。抽成独立函数调用以避免 _setup_observables()
        过程中局部函数命名冲突。

        Args:
            cam_name (str): 要为之创建传感器的相机名
            cam_w (int): 相机宽度
            cam_h (int): 相机高度
            cam_s (None or list): 分割类型，应为：
                `'instance'`: 类实例级分割
                `'class'`: 类级分割
                `'element'`: 逐 geom 级分割
            seg_name_root (str): 该传感器的命名前缀

            modality (str): 赋给所有传感器的模态

        Returns:
            2-tuple:
                camera_segmentation (function): 为该分割传感器生成的传感器函数
                name (str): 对应的传感器名
        """
        # 确保使用正确的图像坐标约定
        convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]

        if cam_s == "instance":
            # 实例级分割：从 instances_to_ids 构造名称到 id 的映射，再由 geom id 映射到实例 id
            name2id = {inst: i for i, inst in enumerate(list(self.model.instances_to_ids.keys()))}
            mapping = {idn: name2id[inst] for idn, inst in self.model.geom_ids_to_instances.items()}
        elif cam_s == "class":
            # 类级分割：从 classes_to_ids 构造名称到 id 的映射，再由 geom id 映射到类 id
            name2id = {cls: i for i, cls in enumerate(list(self.model.classes_to_ids.keys()))}
            mapping = {idn: name2id[cls] for idn, cls in self.model.geom_ids_to_classes.items()}
        else:  # element 级分割
            # 不需要额外映射
            mapping = None

        @sensor(modality=modality)
        def camera_segmentation(obs_cache):
            seg = self.sim.render(
                camera_name=cam_name,
                width=cam_w,
                height=cam_h,
                depth=False,
                segmentation=True,
            )
            # 取出分割图并按坐标约定翻转
            seg = np.expand_dims(seg[::convention, :, 1], axis=-1)
            # 若使用实例或类级分割，则把原始 id 映射到分组的 id
            if mapping is not None:
                seg = (
                    np.fromiter(map(lambda x: mapping.get(x, -1), seg.flatten()), dtype=np.int32).reshape(
                        cam_h, cam_w, 1
                    )
                    + 1
                )
            return seg

        name = f"{seg_name_root}_{cam_s}"

        return camera_segmentation, name

    def _reset_internal(self):
        """
        重置仿真内部配置。
        在父类基础上重置动作维度、调用各机器人 reset() 并更新动作空间维度，
        以及处理 "all-xxx" 这种相机名约定以展开为各机器人对应相机列表。
        """
        # 调用父类的 reset 逻辑
        super()._reset_internal()

        # 重置动作维度
        self._action_dim = 0

        # 重置每个机器人，并累加其动作维度
        for robot in self.robots:
            robot.reset(deterministic=self.deterministic_reset, rng=self.rng)
            self._action_dim += robot.action_dim

        # 视情况更新相机列表
        if self.use_camera_obs:
            temp_names = []
            for cam_name in self.camera_names:
                if "all-" in cam_name:
                    # 需要把 "all-" 后关键字匹配的所有机器人相机名添加进来
                    start_idx = len(temp_names) - 1
                    key = cam_name.replace("all-", "")
                    for robot in self.robots:
                        for robot_cam_name in robot.robot_model.cameras:
                            if key in robot_cam_name:
                                temp_names.append(robot_cam_name)
                    # 同时需要把对应相机的 width/height/depth 也广播展开
                    end_idx = len(temp_names) - 1
                    self.camera_widths = (
                        self.camera_widths[:start_idx]
                        + [self.camera_widths[start_idx]] * (end_idx - start_idx)
                        + self.camera_widths[(start_idx + 1) :]
                    )
                    self.camera_heights = (
                        self.camera_heights[:start_idx]
                        + [self.camera_heights[start_idx]] * (end_idx - start_idx)
                        + self.camera_heights[(start_idx + 1) :]
                    )
                    self.camera_depths = (
                        self.camera_depths[:start_idx]
                        + [self.camera_depths[start_idx]] * (end_idx - start_idx)
                        + self.camera_depths[(start_idx + 1) :]
                    )
                else:
                    # 直接把该相机名加入临时列表
                    temp_names.append(cam_name)
            # 最后用更新后的相机名替换原列表
            self.camera_names = temp_names

    def _pre_action(self, action, policy_step=False):
        """
        覆盖父类方法：使用各自的控制器和夹爪控制对机器人施加动作。

        Args:
            action (np.array): 要施加到机器人上的控制。注意这是一个扁平的一维数组，
                当存在多个机器人时包含所有机器人的动作。对每个机器人对应的动作子段而言，
                前 @self.robots[i].controller.control_dim 维是其控制器期望动作；
                若该机器人有夹爪，紧接的 @self.robots[i].gripper.dof 维是夹爪控制量。
            policy_step (bool): 是否为新的策略步(新动作)

        Raises:
            AssertionError: [动作维度不正确]
        """
        # 校验动作维度
        assert len(action) == self.action_dim, "environment got invalid action dimension -- expected {}, got {}".format(
            self.action_dim, len(action)
        )

        # 按控制器动作更新各机器人关节
        cutoff = 0
        for idx, robot in enumerate(self.robots):
            robot_action = action[cutoff : cutoff + robot.action_dim]
            robot.control(robot_action, policy_step=policy_step)
            cutoff += robot.action_dim

    def _load_robots(self):
        """
        实例化机器人并把它们保存到 self.robots 属性中。
        """
        # 遍历机器人列表，为每一个创建 Robot 对象
        for idx, (name, config) in enumerate(zip(self.robot_names, self.robot_configs)):
            # 创建机器人实例
            self.robots[idx] = ROBOT_CLASS_MAPPING[name](robot_type=name, idn=idx, **config)
            # 然后加载机器人模型
            self.robots[idx].load_model()

    def reward(self, action):
        """
        默认调用父类方法。
        """
        return super().reward(action)

    def _check_success(self):
        """
        默认调用父类方法。
        """
        return super()._check_success()

    def _check_robot_configuration(self, robots):
        """
        校验输入的机器人及对应的任务/配置组合是否合法。
        应在每个具体任务模块中实现。

        Args:
            robots (str or list of str): 任务级环境输入的机器人请求
        """
        raise NotImplementedError
