"""
操作(Manipulation)环境模块。

本模块定义了 :class:`ManipulationEnv`，它是 robosuite 中所有“操作类”任务环境(例如抓取、放置、堆叠等)
的基类。该类继承自 :class:`RobotEnv`，在通用的机器人环境之上增加了与夹爪(gripper)、物体(object)、
末端执行器(end-effector)相关的感知(sensor)与可视化能力。

具体任务子类(如 Lift、Stack、NutAssembly 等)应继承本类并实现任务特有的模型加载、奖励、终止条件
等抽象方法。
"""

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.robot_env import RobotEnv
from robosuite.models.base import MujocoModel
from robosuite.models.grippers import GripperModel
from robosuite.robots import ROBOT_CLASS_MAPPING, FixedBaseRobot, MobileRobot
from robosuite.robots.legged_robot import LeggedRobot
from robosuite.utils.observables import Observable, sensor


class ManipulationEnv(RobotEnv):
    """
    在 Mujoco 中初始化一个面向“操作任务”的机器人环境。

    本类是所有操作类任务(抓取、放置、组装等)环境的基类，它在 :class:`RobotEnv` 通用能力之上
    扩展了与夹爪、物体、末端执行器相关的感知(observable sensor)以及可视化(vis)能力。
    子类需要实现任务特有的模型加载、奖励函数、终止条件等抽象方法。

    Args:
        robots: 指定该环境中要实例化的具体机械臂。
            例如传入 "Sawyer" 会生成一个机械臂；
            传入 ["Panda", "Panda", "Sawyer"] 会生成三个机械臂。

        env_configuration (str): 指定机器人在环境中的布局方式。默认为 "default"，
            其具体含义由各子类自行解释。

        controller_configs (str or list of dict): 若指定，则包含用于创建自定义控制器的相关参数；
            否则使用该任务默认的控制器。可以传入单个 dict(所有机器人共用同一控制器)，
            也可以传入与 "robots" 等长的 list(为每个机器人分别指定控制器)。

        base_types (str or list of str): 基座(base)类型，用于从基座工厂实例化基座模型。
            默认为 "default"，即使用 'robots' 所指定机器人对应的默认基座。
            "NullMount" 表示不挂载任何基座；其它(有效的)模型名则会覆盖默认基座。
            可以传入单个 str(所有机器人共用)，或与 "robots" 等长的 list。

        gripper_types (None or str or list of str): 夹爪类型，用于从夹爪工厂实例化夹爪模型。
            默认为 "default"，即使用 'robots' 指定机器人对应的默认夹爪；
            None 表示移除夹爪；其它(有效的)模型名则会覆盖默认夹爪。
            可以传入单个 str(所有机器人共用)，或与 "robots" 等长的 list。

        initialization_noise (dict or list of dict): 包含初始化噪声参数的字典。
            其期望的键和对应值类型如下：

            :`'magnitude'`: 对机器人各初始关节位置施加单变量随机噪声的缩放因子。
                设为 `None` 或 0.0 表示不施加噪声。
                若使用 "gaussian" 噪声，该值缩放所施加的标准差；
                若使用 "uniform" 噪声，该值设定采样范围的上下界。
            :`'type'`: 噪声类型，可选 "gaussian" 或 "uniform"。

            可以传入单个 dict(所有机器人共用)，或与 "robots" 等长的 list。

            :Note: 指定 "default" 将自动使用默认噪声设置；
                指定 None 将自动创建所需 dict，并把 "magnitude" 设为 0.0。

        use_camera_obs (bool): 若为 True，每次观测都包含渲染出的图像。

        has_renderer (bool): 若为 True，在可视化器中渲染仿真状态，而非以无界面(headless)模式运行。

        has_offscreen_renderer (bool): 是否使用离屏(off-screen)渲染。

        render_camera (str or list of str): 当 `has_renderer` 为 True 时，要渲染的相机名称。
            设为 'None' 会使用默认视角，该视角可由用户用鼠标拖拽/平移。
            传入字符串列表时会从多个相机视角同时渲染。

        render_collision_mesh (bool): 是否在相机中渲染碰撞网格(collision mesh)。

        render_visual_mesh (bool): 是否在相机中渲染可视化网格(visual mesh)。

        render_gpu_device_id (int): 用于离屏渲染的 GPU 设备 id。
            默认为 -1，此时设备将从环境变量(GPUS 或 CUDA_VISIBLE_DEVICES)推断得到。

        control_freq (float): 每秒接收多少次控制信号。它决定了每次动作输入之间所经过的仿真时间基数。

        lite_physics (bool): 是否优化 mujoco 的 forward / step 调用以降低总体仿真开销。
            设为 False 可保持与 robosuite <= 1.4.1 采集数据集的向后兼容。

        horizon (int): 每个回合(episode)固定持续 @horizon 个时间步。

        ignore_done (bool): 若为 True，环境永不终止(忽略 @horizon)。

        hard_reset (bool): 若为 True，在 reset 调用时重新加载模型、仿真器和渲染对象；
            否则只调用 sim.reset 并重置所有 robosuite 内部变量。

        load_model_on_init (bool): 若为 True，在 __init__ 构造函数中加载并初始化模型与渲染器；
            否则在首次调用 reset() 时再初始化这些组件。

        camera_names (str or list of str): 要渲染的相机名称。
            可以传入单个 str(所有相机共用)，或相机列表。

            :Note: 当 @use_camera_obs 为 True 时，至少需要指定一个相机。

            :Note: 要渲染所有机器人某一类型的相机(如 "robotview" 或 "eye_in_hand")，
                可使用 "all-{name}" 约定(例如 "all-robotview")，自动渲染每个机器人相机列表中的所有图像。

        camera_heights (int or list of int): 相机画面高度。
            可以传入单个 int(所有相机共用)，或与 "camera_names" 等长的 list。

        camera_widths (int or list of int): 相机画面宽度。
            可以传入单个 int(所有相机共用)，或与 "camera_names" 等长的 list。

        camera_depths (bool or list of bool): True 表示渲染 RGB-D，否则只渲染 RGB。
            可以传入单个 bool(所有相机共用)，或与 "camera_names" 等长的 list。

        camera_segmentations (None or str or list of str or list of list of str): 每个相机使用的分割(segmentation)类型。
            有效选项：

                `None`: 不使用分割传感器
                `'instance'`: 类实例级别的分割
                `'class'`: 类级别的分割
                `'element'`: 逐 geom 级别的分割

            若不为 None，可指定多种分割类型。[str 或 None 的 list / 单个 str 或 None]
            分别为所有相机指定[多种 / 一种]分割；str 的 list 的 list 则为每个相机单独指定分割设置。

        seed (int): 环境随机种子。默认为 None，即环境不带种子(完全随机)。

    Raises:
        ValueError: [相机观测需要离屏渲染器]
        ValueError: [使用相机观测时必须指定相机名]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        base_types="default",
        gripper_types="default",
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
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        # 机器人信息：统一把 robots 归一化为 list，便于后续按数量遍历
        robots = list(robots) if type(robots) is list or type(robots) is tuple else [robots]
        num_robots = len(robots)

        # 夹爪类型：通过 _input2list 把单个值扩展成与机器人数量等长的 list
        gripper_types = self._input2list(gripper_types, num_robots)

        # 组装传给父类构造函数的机器人配置；这里把每个机器人对应的夹爪类型塞进配置字典
        robot_configs = [
            {
                "gripper_type": gripper_types[idx],
            }
            for idx in range(num_robots)
        ]

        # 调用父类(RobotEnv)的初始化，完成模型/仿真器/观测/渲染等通用初始化
        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
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
            load_model_on_init=load_model_on_init,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            robot_configs=robot_configs,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    @property
    def _visualizations(self):
        """
        本环境支持的可视化关键字。

        该属性返回所有可被单独(选择性)可视化的组件名称集合，供可视化开关
        (例如 env.set可视化)按需开关某一类可视化。

        Returns:
            set: 本环境中所有可被单独可视化的组件名称集合
        """
        # 先获取父类支持的可视化项，再叠加本类特有的“夹爪(grippers)”可视化
        vis_set = super()._visualizations
        vis_set.add("grippers")
        return vis_set

    def _get_obj_eef_sensor(self, prefix, obj_key, fn_name, modality):
        """
        创建一个传感器函数，返回由 @obj_key 指定的物体与由 @prefix 指定的末端执行器之间的相对位置。

        具体来说，相对位置 = 物体位置 - 末端执行器位置，即物体在末端执行器坐标系外的世界系相对位移。
        该传感器常用于策略观测，让机器人“知道”物体相对自己末端的位置。

        Args:
            prefix (str): 末端执行器所属机械臂的前缀
            obj_key (str): 在观测缓存(observation cache)中访问物体位置的键
            fn_name (str): 赋给该传感器函数的名称
            modality (str): 该传感器的模态(modality)，例如 "object"

        Returns:
            function: 传感器函数，返回物体与末端执行器之间的相对位置(世界系下)
        """

        @sensor(modality)
        def sensor_fn(obs_cache):
            # 仅当缓存中同时存在物体位置和末端位置时才计算差值，否则返回 0 向量
            return (
                obs_cache[obj_key] - obs_cache[f"{prefix}eef_pos"]
                if obj_key in obs_cache and f"{prefix}eef_pos" in obs_cache
                else np.zeros(3)
            )

        # 设置函数名，便于在观测缓存中按名字检索其输出
        sensor_fn.__name__ = fn_name
        return sensor_fn

    def _get_world_pose_in_gripper_sensor(self, prefix, fn_name, modality):
        """
        创建一个传感器函数，返回夹爪的逆位姿(inverse pose)。

        即返回“世界坐标系在夹爪坐标系下”的位姿矩阵(4x4齐次变换矩阵)，
        它是夹爪到世界变换矩阵的逆。该位姿常用于将世界系下的物体位姿换算到夹爪系。

        Args:
            prefix (str): 夹爪所属机械臂的前缀
            fn_name (str): 赋给该传感器函数的名称
            modality (str): 该传感器的模态

        Returns:
            function: 传感器函数，返回世界系与夹爪系之间的相对位姿(4x4 矩阵)
        """

        @sensor(modality=modality)
        def fn(obs_cache):
            # 末端位姿(位置+四元数) -> 齐次矩阵 -> 取逆，得到 world_pose_in_gripper
            return (
                T.pose_inv(T.pose2mat((obs_cache[f"{prefix}eef_pos"], obs_cache[f"{prefix}eef_quat"])))
                if f"{prefix}eef_pos" in obs_cache and f"{prefix}eef_quat" in obs_cache
                else np.eye(4)
            )

        fn.__name__ = fn_name
        return fn

    def _get_rel_obj_eef_sensor(self, prefix, obj_key, fn_name, new_key_prefix, modality):
        """
        创建一个传感器函数，返回由 @obj_key 指定的物体相对末端执行器的相对位置，
        并把相对四元数写入观测缓存。该传感器函数会用到机器人夹爪的逆位姿
        (world_pose_in_gripper)，该逆位姿应已存在于观测缓存中。

        与 _get_obj_eef_sensor 不同，这里返回的是“物体在夹爪坐标系下”的相对位姿，
        而非世界系下的差值；同时会把相对四元数存入缓存，供后续传感器读取。

        Args:
            prefix (str): 在观测缓存中访问机器人臂逆位姿所用的前缀
            obj_key (str): 在观测缓存中访问物体位置/四元数所用的键
            fn_name (str): 赋给该传感器函数的名称
            new_key_prefix (str): 写入缓存时使用的新键前缀
            modality (str): 该传感器的模态

        Returns:
            function: 传感器函数，返回物体与末端执行器之间的相对位置(夹爪系下)
        """

        @sensor(modality=modality)
        def fn(obs_cache):
            # 若缓存中缺少物体位姿或夹爪逆位姿，则直接返回默认 0 向量
            if any(
                [
                    name not in obs_cache
                    for name in [f"{obj_key}_pos", f"{obj_key}_quat", f"world_pose_in_{prefix}gripper"]
                ]
            ):
                return np.zeros(3)
            # 物体世界位姿 -> 齐次矩阵
            obj_pose = T.pose2mat((obs_cache[f"{obj_key}_pos"], obs_cache[f"{obj_key}_quat"]))
            # 将物体位姿从世界系转换到夹爪系(利用 world_pose_in_gripper)
            rel_pose = T.pose_in_A_to_pose_in_B(obj_pose, obs_cache[f"world_pose_in_{prefix}gripper"])
            # 拆成相对位置和相对四元数
            rel_pos, rel_quat = T.mat2pose(rel_pose)
            # 把相对四元数写入缓存，供 _get_obj_eef_rel_quat_sensor 读取
            obs_cache[f"{obj_key}_to_{new_key_prefix}eef_quat"] = rel_quat
            return rel_pos

        fn.__name__ = fn_name
        return fn

    def _get_obj_eef_rel_quat_sensor(self, prefix, obj_key, fn_name, modality):
        """
        创建一个传感器函数，返回由 @obj_key 指定的物体与由 @prefix 指定的末端执行器之间的相对四元数。

        该传感器依赖 _get_rel_obj_eef_sensor 写入缓存的相对四元数
        (键名形如 "{obj_key}_to_{prefix}eef_quat")，因此通常与前者配套使用。

        Args:
            prefix (str): 末端执行器所属机械臂的前缀
            obj_key (str): 在观测缓存中访问物体相对四元数所用的键
            fn_name (str): 赋给该传感器函数的名称
            modality (str): 该传感器的模态

        Returns:
            function: 传感器函数，返回物体与末端执行器之间的相对四元数
        """

        @sensor(modality)
        def sensor_fn(obs_cache):
            # 从缓存读取相对四元数；若尚未生成则返回 0 四元数(长度为4)
            return (
                obs_cache[f"{obj_key}_to_{prefix}eef_quat"]
                if f"{obj_key}_to_{prefix}eef_quat" in obs_cache
                else np.zeros(4)
            )

        sensor_fn.__name__ = fn_name
        return sensor_fn

    def _check_grasp(self, gripper, object_geoms):
        """
        检查由 @gripper 指定的夹爪是否正在抓取环境中由 @object_geoms 指定的物体。
        若指定了多个夹爪，只要其中任意一个夹爪抓到物体即返回 True。

        默认判定逻辑：当 "left_fingerpad" 和 "right_fingerpad" 两个 geom 组中，
        各自至少有一个 geom 与 @object_geoms 中的任一 geom 发生接触时，视为成功抓取。
        也可以通过 @gripper 自定义夹爪 geom 组。

        Args:
            gripper (GripperModel or str or list of str or list of list of str or dict):
                若为 MujocoModel(GripperModel)，则检查该夹爪是否抓取物体(按 "left_fingerpad"
                和 "right_fingerpad" geom 组判定)。
                否则该参数用于自定义“共同定义一次抓取”的夹爪 geom 组：可以是
                str(单个 geom 构成一个组)、list of str(多个组，每组单个 geom)、
                list of list of str(多个组，每组多个 geom)，
                或在机器人有多个臂/夹爪时使用 dict。判定条件是：每个组中至少有一个 geom
                与 @object_geoms 中的任一 geom 接触，才会返回 True。
            object_geoms (str or list of str or MujocoModel):
                若传入 MujocoModel，则检查与其 contact_geoms 的任何碰撞；
                否则应传入构成物体、用于接触判定的具体 geom 名称(单个或列表)。

        Returns:
            bool: 夹爪是否正在抓取指定物体
        """
        # 把物体 geom 和夹爪 geom 统一成标准化形式(list of str)
        if isinstance(object_geoms, MujocoModel):
            o_geoms = object_geoms.contact_geoms
        else:
            o_geoms = [object_geoms] if type(object_geoms) is str else object_geoms

        if isinstance(gripper, GripperModel):
            # 夹爪模型：取左右指腹两个 geom 组作为判定组
            g_geoms = [gripper.important_geoms["left_fingerpad"], gripper.important_geoms["right_fingerpad"]]
        elif type(gripper) is str:
            # 单个 geom：包成一个组
            g_geoms = [[gripper]]
        elif isinstance(gripper, dict):
            # 多臂情况：dict 形式 {arm: GripperModel}，任一臂抓到即返回 True
            assert all([isinstance(gripper[arm], GripperModel) for arm in gripper]), "Invalid gripper dict format!"
            return any([self._check_grasp(gripper[arm], object_geoms) for arm in gripper])
        else:
            # list 形式：把每个元素归一化为 list of str(即一个组)
            g_geoms = [[g_group] if type(g_group) is str else g_group for g_group in gripper]

        # 逐组检查：每个组都必须至少有一个 geom 与物体 geoms 接触，否则判为未抓取
        for g_group in g_geoms:
            if not self.check_contact(g_group, o_geoms):
                return False
        return True

    def _gripper_to_target(self, gripper, target, target_type="body", return_distance=False):
        """
        计算由 @gripper 指定的夹爪到 @target 指定目标之间的 (x,y,z) 笛卡尔距离
        (target_pos - gripper_pos)。若设置 @return_distance，则返回欧氏标量距离。
        若 @gripper 是 dict，则返回所有夹爪中到目标的最小距离。

        Args:
            gripper (MujocoModel or dict): 夹爪模型(用于取 grip site 位置)；多臂时可传 dict
            target (MujocoModel or str): 可以是 site / geom / body 名称，或作为目标的模型。
                若传入模型，则用其根 body 作为目标。
            target_type (str): 取值之一 {"body", "geom", "site"}，表示 @target 所指元素的类型。
            return_distance (bool): 若设置，返回欧氏距离(标量)而非笛卡尔距离向量

        Returns:
            np.array or float: 夹爪到目标的(笛卡尔或欧氏)距离
        """
        if isinstance(gripper, dict):
            assert all([isinstance(gripper[arm], GripperModel) for arm in gripper]), "Invalid gripper dict format!"
            # 多臂情况下：取所有臂中到目标的最小距离
            if return_distance:
                return min(
                    [self._gripper_to_target(gripper[arm], target, target_type, return_distance) for arm in gripper]
                )
            else:
                # 笛卡尔距离时，按向量模长最小的那个返回
                return min(
                    [self._gripper_to_target(gripper[arm], target, target_type, return_distance) for arm in gripper],
                    key=lambda x: np.linalg.norm(x),
                )

        # 获取夹爪位置(取 grip_site 的世界坐标)
        gripper_pos = self.sim.data.get_site_xpos(gripper.important_sites["grip_site"])
        # 根据目标类型获取目标位置：模型取根 body，否则按 target_type 取 body/site/geom
        if isinstance(target, MujocoModel):
            target_pos = self.sim.data.get_body_xpos(target.root_body)
        elif target_type == "body":
            target_pos = self.sim.data.get_body_xpos(target)
        elif target_type == "site":
            target_pos = self.sim.data.get_site_xpos(target)
        else:
            target_pos = self.sim.data.get_geom_xpos(target)
        # 计算位置差
        diff = target_pos - gripper_pos
        # 按需返回标量距离或向量差
        return np.linalg.norm(diff) if return_distance else diff

    def _visualize_gripper_to_target(self, gripper, target, target_type="body"):
        """
        根据夹爪到 @target 的欧氏距离，对夹爪的可视化 site 进行着色：
        距离越近颜色越偏绿，越远越偏红(红 -> 绿)。若传入 dict 形式的多个夹爪，
        则对每个夹爪分别进行可视化。

        Args:
            gripper (MujocoModel or dict): 夹爪模型(用于更新其 grip site 的 rgb)
            target (MujocoModel or str): 可以是 site / geom / body 名称，或作为目标的模型。
                若传入模型，则用其根 body 作为目标。
            target_type (str): 取值之一 {"body", "geom", "site"}，表示 @target 所指元素的类型。
        """
        if isinstance(gripper, dict):
            assert all([isinstance(gripper[arm], GripperModel) for arm in gripper]), "Invalid gripper dict format!"
            # 多臂：逐个臂分别可视化
            for arm in gripper:
                self._visualize_gripper_to_target(gripper[arm], target, target_type)
            return
        # 获取夹爪位置(取 grip_site 的世界坐标)
        gripper_pos = self.sim.data.get_site_xpos(gripper.important_sites["grip_site"])
        # 根据目标类型获取目标位置：模型取根 body，否则按 target_type 取 body/site/geom
        if isinstance(target, MujocoModel):
            target_pos = self.sim.data.get_body_xpos(target.root_body)
        elif target_type == "body":
            target_pos = self.sim.data.get_body_xpos(target)
        elif target_type == "site":
            target_pos = self.sim.data.get_site_xpos(target)
        else:
            target_pos = self.sim.data.get_geom_xpos(target)
        # 根据到目标的(平方)距离对夹爪 site 着色：远红近绿
        dist = np.sum(np.square((target_pos - gripper_pos)))
        max_dist = 0.1
        # 距离越近 scaled 越接近1(越绿)，指数 15 使颜色过渡集中在近距离处
        scaled = (1.0 - min(dist / max_dist, 1.0)) ** 15
        rgba = np.zeros(3)
        rgba[0] = 1 - scaled  # 红色分量：远->1，近->0
        rgba[1] = scaled      # 绿色分量：远->0，近->1
        # 把算好的 rgb 写回 grip site 的 rgba(仅前3通道，alpha 保持不变)
        self.sim.model.site_rgba[self.sim.model.site_name2id(gripper.important_sites["grip_site"])][:3] = rgba

    def _get_arm_prefixes(self, robot, include_robot_name=True):
        """
        返回环境中各机械臂的命名前缀，用于访问本体的感知(proprioceptive)信息。
        约定：若只有一个臂，则前缀不含臂名；否则返回每个臂的前缀列表，
        其中包含机器人的命名前缀(当 include_robot_name 为真)和臂名。

        例如单臂时返回 ["robot0_"]，双臂时返回 ["robot0_arm0_", "robot0_arm1_"]。

        Args:
            robot (RobotModel): 要从中提取前缀的机器人模型

        Returns:
            list: 各机械臂的命名前缀列表
        """

        # 若需要包含机器人名，则取其 naming_prefix，否则为空串
        name_pf = robot.robot_model.naming_prefix if include_robot_name else ""
        # 单臂：直接返回机器人前缀(不含臂名)
        if len(robot.arms) == 1:
            return [name_pf]

        # 多臂：为每个臂拼接 "{机器人前缀}{臂名}_" 作为前缀
        prefixes = [f"{name_pf}{arm}_" for arm in robot.arms]
        return prefixes

    def _check_robot_configuration(self, robots):
        """
        合法性检查：确保输入的机器人及其对应的任务/配置组合是合法的。
        应在每个具体任务子模块中实现/覆盖以做更细致的校验。

        Args:
            robots (str or list of str): 任务级环境传入的机器人请求
        """
        # 确保所有输入的机器人都是“操作型”机器人(固定基座/移动/足式)
        if type(robots) is str:
            robots = [robots]
        for robot in robots:
            assert issubclass(ROBOT_CLASS_MAPPING[robot], FixedBaseRobot) or issubclass(
                ROBOT_CLASS_MAPPING[robot], MobileRobot or issubclass(ROBOT_CLASS_MAPPING[robot], LeggedRobot)
            ), f"Only manipulator robots supported for manipulation environment! Got {ROBOT_CLASS_MAPPING[robot]}"
