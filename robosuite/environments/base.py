"""
robosuite 环境基类模块。

本模块定义了 robosuite 中所有 Mujoco 环境的根基类 :class:`MujocoEnv`，以及用于环境注册的
元类 :class:`EnvMeta` 和工厂函数 :func:`make`。所有具体环境(如 RobotEnv、ManipulationEnv
及其子类)最终都继承自 :class:`MujocoEnv`，并自动通过元类注册到 :data:`REGISTERED_ENVS`，
从而可以通过 :func:`make` 按名称实例化。

该基类封装了与 Mujoco 仿真器交互的通用逻辑：模型加载、仿真器初始化、渲染器初始化、
观测(observables)管理、reset/step 主循环、可视化开关、接触检测等。
"""

import os
import xml.etree.ElementTree as ET
from collections import OrderedDict
from copy import deepcopy

import numpy as np

import robosuite
import robosuite.macros as macros
import robosuite.utils.sim_utils as SU
from robosuite.renderers.base import load_renderer_config
from robosuite.renderers.viewer import OpenCVViewer
from robosuite.utils import SimulationError, XMLError
from robosuite.utils.binding_utils import MjRenderContextOffscreen, MjSim

# 全局的环境注册表：环境类名 -> 环境类对象
REGISTERED_ENVS = {}


def register_env(target_class):
    """
    将目标环境类注册到全局注册表 REGISTERED_ENVS 中。

    通常由 EnvMeta 元类在类定义时自动调用，无需手动调用。

    Args:
        target_class (type): 要注册的环境类
    """
    REGISTERED_ENVS[target_class.__name__] = target_class


def make(env_name, *args, **kwargs):
    """
    实例化一个 robosuite 环境。
    本方法尝试以稍显粗糙的方式对齐 gym.make 的等价功能。
    Args:
        env_name (str): 要初始化的 robosuite 环境名称
        *args: 传给具体环境类初始化函数的额外位置参数
        **kwargs: 传给具体环境类初始化函数的额外关键字参数
    Returns:
        MujocoEnv: 所请求的 robosuite 环境实例
    Raises:
        Exception: [无效的环境名称]
    """
    if env_name not in REGISTERED_ENVS:
        raise Exception(
            "Environment {} not found. Make sure it is a registered environment among: {}".format(
                env_name, ", ".join(REGISTERED_ENVS)
            )
        )
    return REGISTERED_ENVS[env_name](*args, **kwargs)


class EnvMeta(type):
    """用于自动注册环境类的元类"""

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)

        # 这里列出所有不应被注册的环境基类(仅作为父类使用)
        _unregistered_envs = ["MujocoEnv", "RobotEnv", "ManipulationEnv", "TwoArmEnv"]

        # 只有具体任务子类才会被注册到 REGISTERED_ENVS
        if cls.__name__ not in _unregistered_envs:
            register_env(cls)
        return cls


class MujocoEnv(metaclass=EnvMeta):
    """
    初始化一个 Mujoco 环境。

    本类是 robosuite 中所有环境的根基类，封装了与 Mujoco 仿真器交互的通用逻辑，
    包括模型/仿真器/渲染器初始化、观测(observables)管理、reset/step 主循环、
    可视化开关、接触检测等。具体任务环境应继承本类(或其子类)并实现
    `_load_model`、`_setup_observables`、`reward`、`_check_success` 等抽象方法。

    Args:
        has_renderer (bool): 若为 True，在可视化器中渲染仿真状态，而非以无界面(headless)模式运行。
        has_offscreen_renderer (bool): 是否使用离屏(off-screen)渲染。
        render_camera (str or list of str): 当 `has_renderer` 为 True 时要渲染的相机名称。
            设为 'None' 会使用默认视角，该视角可由用户用鼠标拖拽/平移。
            传入字符串列表时会从多个相机视角同时渲染。
        render_collision_mesh (bool): 是否在相机中渲染碰撞网格(collision mesh)。
        render_visual_mesh (bool): 是否在相机中渲染可视化网格(visual mesh)。
        render_gpu_device_id (int): 用于离屏渲染的 GPU 设备 id。
            默认为 -1，此时设备将从环境变量(GPUS 或 CUDA_VISIBLE_DEVICES)推断得到。
        control_freq (float): 每秒接收多少次控制信号。它决定了每次动作输入之间所经过的仿真时间。
        lite_physics (bool): 是否优化 mujoco 的 forward / step 调用以降低总体仿真开销。
            设为 False 可保持与 robosuite <= 1.4.1 采集数据集的向后兼容。
        horizon (int): 每个回合(episode)固定持续 @horizon 个时间步。
        ignore_done (bool): 若为 True，环境永不终止(忽略 @horizon)。
        hard_reset (bool): 若为 True，在 reset 调用时重新加载模型、仿真器和渲染对象；
            否则只调用 sim.reset 并重置所有 robosuite 内部变量。
        load_model_on_init (bool): 若为 True，在 __init__ 构造函数中加载并初始化模型与渲染器；
            否则在首次调用 reset() 时再初始化这些组件。
        renderer (str): 要使用的渲染器名称字符串
        renderer_config (dict): 渲染器配置字典
        seed (int): 环境随机种子。默认为 None，即环境不带种子(完全随机)。
    Raises:
        ValueError: [无效的渲染器选择]
    """

    def __init__(
        self,
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
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        # 渲染相关属性
        self.has_renderer = has_renderer
        # 在屏幕渲染(非 mjviewer)时也需要离屏渲染器；或显式开启离屏渲染
        self.has_offscreen_renderer = (has_renderer and renderer != "mjviewer") or has_offscreen_renderer
        # 把单个相机名归一化为列表
        if render_camera is not None and isinstance(render_camera, str):
            render_camera = [render_camera]
        self.render_camera = render_camera
        self.render_collision_mesh = render_collision_mesh
        self.render_visual_mesh = render_visual_mesh
        self.render_gpu_device_id = render_gpu_device_id
        self.viewer = None

        # 仿真相关属性
        self._observables = {}  # 观测名 -> Observable 对象 的映射
        self._obs_cache = {}    # 观测名 -> 预计算/部分计算的观测值 的缓存
        self.control_freq = control_freq
        self.lite_physics = lite_physics
        self.horizon = horizon
        self.ignore_done = ignore_done
        self.hard_reset = hard_reset
        # 在 _initialize_sim() 中对模型 xml 进行处理的函数列表；
        # 默认包含 edit_model_xml 处理路径等
        self._xml_processors = [self.edit_model_xml]
        self.model = None
        self.cur_time = None
        self.model_timestep = None
        self.control_timestep = None
        self.deterministic_reset = False  # 是否对物体/机器人关节进行随机化重置

        self.renderer = renderer
        self.renderer_config = renderer_config

        self.seed = seed
        self.rng = np.random.default_rng(seed)  # 由种子初始化的随机数生成器

        self._ep_meta = {}  # 回合级元数据

        self.load_model_on_init = load_model_on_init

        # 标记环境是否已完成完整初始化
        self._env_is_initialized = False

        if self.load_model_on_init:
            # 加载模型(子类应实现 _load_model)
            self._load_model()

            # 初始化仿真器
            self._initialize_sim()

            # 初始化渲染器
            self.initialize_renderer()

            # 这些变量将在后续流程中赋值；
            # 这里先置为 None，防止在被赋值前被引用
            self.viewer = None
            self.viewer_get_obs = None

            # 执行其余内部(重新)初始化
            self._reset_internal()

            # 加载 observables：优先用 viewer 提供的设置方法，否则用本类的
            if hasattr(self.viewer, "_setup_observables"):
                self._observables = self.viewer._setup_observables()
            else:
                self._observables = self._setup_observables()

            # 检查 viewer 是否提供 _get_observations 方法，并设标记供后续使用
            self.viewer_get_obs = hasattr(self.viewer, "_get_observations")
            self._env_is_initialized = True
        else:
            # 这些变量将在后续流程中赋值；
            # 这里先置为 None，防止在被赋值前被引用
            self.sim = None
            self.viewer = None
            self.viewer_get_obs = None

    def initialize_renderer(self):
        """
        初始化并实例化渲染器(viewer)。

        根据 self.renderer 选择对应的渲染器后端：
          - "mujoco": 不创建专门 viewer(由子类自行处理)
          - "mjviewer": 使用 mujoco 原生 mjviewer 渲染器，且只允许指定一个相机
          - 其它: 抛出 ValueError
        若未提供 renderer_config 则会加载对应渲染器的默认配置。
        """
        self.renderer = self.renderer.lower()

        # 若未显式指定渲染器配置(且不是 mujoco 后端)，则加载默认配置
        if self.renderer_config is None and self.renderer != "mujoco":
            self.renderer_config = load_renderer_config(self.renderer)

        if self.renderer == "mujoco":
            # mujoco 后端不在此处创建 viewer
            pass
        elif self.renderer == "mjviewer":
            from robosuite.renderers.viewer import MjviewerRenderer

            # mjviewer 只允许指定一个相机
            if self.render_camera is not None:
                assert len(self.render_camera) == 1, "Only one camera can be specified for mjviewer"
                camera_id = self.sim.model.camera_name2id(self.render_camera[0])
            else:
                camera_id = None
            self.viewer = MjviewerRenderer(env=self, camera_id=camera_id, **self.renderer_config)
        else:
            raise ValueError(
                f"{self.renderer} is not a valid renderer name. Valid options include mjviewer (native mujoco renderer), mujoco"
            )

    def initialize_time(self, control_freq):
        """
        初始化仿真所用的时间常量。

        Args:
            control_freq (float): 仿真中控制循环运行的频率(Hz)
        """
        self.cur_time = 0
        # 单步仿真时间步长(由全局宏定义)
        self.model_timestep = macros.SIMULATION_TIMESTEP
        if self.model_timestep <= 0:
            raise ValueError("Invalid simulation timestep defined!")
        self.control_freq = control_freq
        if control_freq <= 0:
            raise SimulationError("Control frequency {} is invalid".format(control_freq))
        # 每次控制(动作)对应的时间步长 = 1 / 控制频率
        self.control_timestep = 1.0 / control_freq

    def set_xml_processor(self, processor):
        """
        设置在 _initialize_sim() 调用中处理 xml 字符串的处理器函数。

        Args:
            processor (None or function): 若指定，该方法应接收一个 xml 字符串
                (并返回处理后的 xml 字符串)。
        """
        self._xml_processors.append(processor)

    def _load_model(self):
        """加载 xml 模型，结果存入 self.model。子类应覆盖此方法。"""
        pass

    def _setup_references(self):
        """
        设置对重要组件的引用(reference)。一个引用通常是一个索引或索引列表，
        指向 MuJoCo 扁平化数组中对应的元素(MuJoCo 以扁平数组存储物理仿真数据)。
        """
        # 建立模型名到 ID 的映射
        self.model.generate_id_mappings(sim=self.sim)

    def _setup_observables(self):
        """
        为本环境设置要使用的 observables。

        子类应覆盖此方法以返回任务相关的观测；本基类返回空 dict。

        Returns:
            OrderedDict: 观测名 -> 对应 Observable 对象 的映射
        """
        return OrderedDict()

    def _initialize_sim(self, xml_string=None):
        """
        创建 MjSim 对象并存入 self.sim。若指定 @xml_string，则从该 xml 字符串创建 MjSim；
        否则从 self.model 获取 xml 来实例化仿真器。

        在创建仿真器前，会依次执行 self._xml_processors 中的每个处理器对 xml 进行处理。

        Args:
            xml_string (str): 若指定，则用该 xml 字符串创建 MjSim
        """
        xml = xml_string if xml_string else self.model.get_xml()

        # 在初始化 sim 之前，先用所有 xml 处理器处理 xml 字符串
        for processor in self._xml_processors:
            xml = processor(xml)

        # 创建仿真器实例
        self.sim = MjSim.from_xml_string(xml)

        # 跑一步 forward，确保改动已传播到 sim 状态
        self.sim.forward()

        # 根据控制频率设置仿真时间常量
        self.initialize_time(self.control_freq)

    def reset(self):
        """
        重置仿真。

        若 hard_reset 为真(且非确定性重置)，会销毁并重建模型/仿真器/渲染器；
        否则只调用 sim.reset() 进行软重置。
        重置后会重建 observables、关闭所有可视化 site、更新状态，并返回新的观测。

        Returns:
            OrderedDict: 重置后的环境观测空间
        """
        # TODO(yukez): investigate black screen of death
        # 若使用 mjviewer，reset 前先销毁旧 viewer(避免黑屏等问题)

        # 始终先终止 mjviewer
        if self.renderer == "mjviewer":
            self._destroy_viewer()

        if (self.sim is None) or (self.hard_reset and not self.deterministic_reset):
            # 硬重置：销毁并重建模型与仿真器
            if self.renderer == "mujoco":
                self._destroy_viewer()
                self._destroy_sim()
            self._load_model()
            self._initialize_sim()
        # Else, we only reset the sim internally
        else:
            # 软重置：仅调用 sim.reset()
            self.sim.reset()

        if self._env_is_initialized is True:
            # 已初始化过：重置必要的 robosuite 内部变量
            self._reset_internal()
            self.sim.forward()
        else:
            # 首次初始化：初始化渲染器
            self.initialize_renderer()

            # 这些变量将在后续流程中赋值；
            # 这里先置为 None，防止在被赋值前被引用
            self.viewer = None
            self.viewer_get_obs = None

            # 执行其余内部(重新)初始化
            self._reset_internal()
            self.sim.forward()

            # 加载 observables：优先用 viewer 提供的设置方法，否则用本类的
            if hasattr(self.viewer, "_setup_observables"):
                self._observables = self.viewer._setup_observables()
            else:
                self._observables = self._setup_observables()

            # 检查 viewer 是否提供 _get_observations 方法，并设标记供后续使用
            self.viewer_get_obs = hasattr(self.viewer, "_get_observations")
            self._env_is_initialized = True

        # 清空观测缓存，并按需重置 observables
        self._obs_cache = {}
        self._reset_observables()

        # 默认关闭所有 site 的可视化
        self.visualize(vis_settings={vis: False for vis in self._visualizations})

        # 按需更新 site 状态
        self.update_state()

        if self.viewer is not None and self.renderer != "mujoco":
            self.viewer.reset()

        # 获取并返回重置后的观测(强制更新)
        observations = (
            self.viewer._get_observations(force_update=True)
            if self.viewer_get_obs
            else self._get_observations(force_update=True)
        )

        # Return new observations
        return observations

    def _reset_observables(self):
        """
        在硬重置情况下重新更新 observables 的 sensor 对象引用。

        因为硬重置会重建仿真器，sensor 闭包中持有的引用需要被刷新。
        """
        if self.hard_reset:
            # 硬重置：必须重新更新 sensor 对象引用
            if hasattr(self.viewer, "_setup_observables"):
                _observables = self.viewer._setup_observables()
            else:
                _observables = self._setup_observables()
            for obs_name, obs in _observables.items():
                self.modify_observable(observable_name=obs_name, attribute="sensor", modifier=obs._sensor)

    def _reset_internal(self):
        """重置仿真器的内部配置。"""

        # 创建可视化窗口或渲染器
        if self.has_renderer and self.viewer is None:
            if self.renderer == "mujoco":
                self.viewer = OpenCVViewer(self.sim)

                # 设置观看的相机角度
                if self.render_camera is not None:
                    camera_ids = []
                    for cam in self.render_camera:
                        camera_id = self.sim.model.camera_name2id(cam)
                        camera_ids.append(camera_id)
                    self.viewer.set_camera(camera_ids)

            elif self.renderer == "mjviewer":
                self.initialize_renderer()

        if self.has_offscreen_renderer:
            # 若尚未创建离屏渲染上下文则创建
            if self.sim._render_context_offscreen is None:
                render_context = MjRenderContextOffscreen(self.sim, device_id=self.render_gpu_device_id)
            # geomgroup[0] 控制碰撞网格显示；geomgroup[1] 控制可视化网格显示
            self.sim._render_context_offscreen.vopt.geomgroup[0] = 1 if self.render_collision_mesh else 0
            self.sim._render_context_offscreen.vopt.geomgroup[1] = 1 if self.render_visual_mesh else 0

        # additional housekeeping
        self.sim_state_initial = self.sim.get_state()
        self._setup_references()
        self.cur_time = 0
        self.timestep = 0
        self.done = False

        # Empty observation cache and reset all observables
        self._obs_cache = {}
        for observable in self._observables.values():
            observable.reset()

    def get_ep_meta(self):
        """
        返回包含回合元数据(episode metadata)的字典。

        Returns:
            dict: 回合元数据
        """
        return deepcopy(self._ep_meta)

    def set_ep_meta(self, meta):
        """
        设置回合元数据。

        Args:
            meta (dict): 包含回合元数据的字典
        """
        self._ep_meta = meta

    def unset_ep_meta(self):
        """
        清空(取消设置)回合元数据。
        """
        self._ep_meta = {}

    def _update_observables(self, force=False):
        """
        更新本环境中所有 observables。

        Args:
            force (bool): 若为 True，强制所有 observables 把其内部值更新为最新值。
                当你直接设置仿真状态而未真正 step 仿真时，想立即获取观测会很有用。
        """
        for observable in self._observables.values():
            observable.update(timestep=self.model_timestep, obs_cache=self._obs_cache, force=force)

    def _get_observations(self, force_update=False):
        """
        从环境中获取观测。

        会遍历所有 observables，收集已启用且处于 active 状态的观测值，
        并按模态(modality)将同名模态的观测拼接成一个数组(便于策略消费)。

        Args:
            force_update (bool): 若为 True，强制所有 observables 把其内部值更新为最新值。
                当你直接设置仿真状态而未真正 step 仿真时，想立即获取观测会很有用。
        Returns:
            OrderedDict: 包含观测的 OrderedDict [(观测名, np.array), ...]
        """
        observations = OrderedDict()
        obs_by_modality = OrderedDict()

        # 若请求强制更新，则先强制更新所有 observables
        if force_update:
            self._update_observables(force=True)

        # 遍历所有 observables，收集当前观测
        for obs_name, observable in self._observables.items():
            if observable.is_enabled() and observable.is_active():
                obs = observable.obs
                observations[obs_name] = obs
                modality = observable.modality + "-state"
                if modality not in obs_by_modality:
                    obs_by_modality[modality] = []
                # 统一成 numpy 数组以便后续拼接
                array_obs = [obs] if type(obs) in {int, float} or not obs.shape else obs
                obs_by_modality[modality].append(np.array(array_obs))

        # 把按模态分组的观测拼接到一起
        for modality, obs in obs_by_modality.items():
            # 为节省内存，仅在显式开启 CONCATENATE_IMAGES 时才拼接图像观测
            if modality == "image-state" and not macros.CONCATENATE_IMAGES:
                continue
            observations[modality] = np.concatenate(obs, axis=-1)

        return observations

    def step(self, action):
        """
        以控制命令 @action 在仿真中前进一步。

        由于 env.step 的频率低于 mjsim 的仿真时间步频率，控制器会在两次高层动作之间
        输出多次力矩命令；因此用 'policy_step' 标记当前这一步是策略步(新动作)还是
        仿真内部更新步。

        Args:
            action (np.array): 要在环境中执行的动作
        Returns:
            4-tuple:
                - (OrderedDict) 环境观测
                - (float) 环境奖励
                - (bool) 当前回合是否已结束
                - (dict) 杂项信息
        Raises:
            ValueError: [在回合结束后继续 step]
        """
        if self.done:
            raise ValueError("executing action in terminated episode")

        self.timestep += 1

        # 由于 env.step 频率低于 mjsim 时间步频率，内部控制器会在两次高层动作之间
        # 输出多次力矩命令。用 'policy_step' 标记当前是策略步(新动作)还是内部更新步
        policy_step = True

        # 以模型时间步速率循环仿真，直到完成一个控制时间步(由控制频率决定)
        for i in range(int(self.control_timestep / self.model_timestep)):
            if self.lite_physics:
                self.sim.step1()
            else:
                self.sim.forward()
            self._pre_action(action, policy_step)
            if self.lite_physics:
                self.sim.step2()
            else:
                self.sim.step()
            self._update_observables()
            policy_step = False

        # 注意：一次性累加控制时间步，避免浮点误差累积
        self.cur_time += self.control_timestep

        reward, done, info = self._post_action(action)

        if self.viewer is not None and self.renderer != "mujoco":
            self.viewer.update()
        elif self.has_renderer and self.renderer == "mjviewer" and self.viewer is None:
            # 之前被销毁的 viewer 需要重新启动
            self.initialize_renderer()
            # 让 mujoco viewer 进行渲染
            self.viewer.update()

        observations = self.viewer._get_observations() if self.viewer_get_obs else self._get_observations()
        return observations, reward, done, info

    def _pre_action(self, action, policy_step=False):
        """
        在执行动作前进行预处理。

        默认实现是直接把动作写入仿真器的控制数组(ctrl)。
        子类(如 RobotEnv)会覆盖此方法以通过控制器驱动机器人。

        Args:
            action (np.array): 要在环境中执行的动作
            policy_step (bool): 当前这一步是真正的策略步，还是仿真内部更新步
        """
        self.sim.data.ctrl[:] = action

    def _post_action(self, action):
        """
        在执行动作后进行善后处理。

        计算奖励、判断是否结束(done)。

        Args:
            action (np.array): 要在环境中执行的动作
        Returns:
            3-tuple:
                - (float) 环境奖励
                - (bool) 当前回合是否已结束
                - (dict) 空字典，留给子类填充信息
        """
        reward = self.reward(action)

        # 当经过的时间步数达到 horizon 即视为结束(ignore_done 为真时永不结束)
        self.done = (self.timestep >= self.horizon) and not self.ignore_done

        return reward, self.done, {}

    def reward(self, action):
        """
        奖励函数，应为状态和动作的函数。子类必须实现此方法。

        Args:
            action (np.array): 要在环境中执行的动作
        Returns:
            float: 环境奖励
        """
        raise NotImplementedError

    def render(self):
        """
        渲染到屏幕窗口。
        """
        self.viewer.render()

    def get_pixel_obs(self):
        """
        从指定渲染器获取环境的像素观测。
        """
        self.viewer.get_pixel_obs()

    def close_renderer(self):
        """
        关闭渲染器。
        """
        self.viewer.close()

    def observation_spec(self):
        """
        返回一个观测作为观测规范(observation specification)。

        一种替代设计是返回 OrderedDict，其键为观测名、值为观测形状。
        这里保留当前实现(直接返回观测)，因为实践中更易用。

        Returns:
            OrderedDict: 环境的观测
        """
        observation = self.viewer._get_observations() if self.viewer_get_obs else self._get_observations()
        return observation

    def clear_objects(self, object_names):
        """
        将名为 @object_names 的物体从任务空间中移除。这在支持
        @self.single_object_mode 这种“单类型物体”任务模式时很有用，
        无需修改模型定义。

        实现方式是把对应物体的关节位置设为远离工作空间的值(10,10,10)。

        Args:
            object_names (str or list of str): 要从任务工作空间移除的物体名
        """
        object_names = {object_names} if type(object_names) is str else set(object_names)
        for obj in self.model.mujoco_objects:
            if obj.name in object_names:
                # 把该物体关节的自由关节 qpos 设到远处(位置 10,10,10 + 单位四元数)
                self.sim.data.set_joint_qpos(obj.joints[0], np.array((10, 10, 10, 1, 0, 0, 0)))

    def visualize(self, vis_settings):
        """
        执行所需的可视化操作。

        Args:
            vis_settings (dict): 可视化关键字到 True/False 的映射，决定对应组件是否可视化。
                应包含 "env" 关键字以及其它相关选项。
        """
        # 设置环境物体 site 的可见性
        for obj in self.model.mujoco_objects:
            obj.set_sites_visibility(sim=self.sim, visible=vis_settings["env"])

    def edit_model_xml(self, xml_str):
        """
        对模型 xml 进行自定义修改，包括解析相对路径、对既有 demonstration 文件
        追加修改、以及其它自定义脚本。环境子类应覆盖此方法以增加环境特有的
        xml 编辑功能。

        本基类默认实现：把 mesh / texture 的文件路径中以 "robosuite" 开头的部分
        替换为当前 robosuite 安装路径，确保资源能被找到。

        Args:
            xml_str (str): 作为字符串的 Mujoco 仿真 XML 文件
        Returns:
            str: 修改后的 xml 字符串
        """

        # 取得当前 robosuite 包的安装目录
        path = os.path.split(robosuite.__file__)[0]
        path_split = path.split("/")

        # 替换 mesh 和 texture 的文件路径
        tree = ET.fromstring(xml_str)
        root = tree
        asset = root.find("asset")
        meshes = asset.findall("mesh")
        textures = asset.findall("texture")
        all_elements = meshes + textures

        for elem in all_elements:
            old_path = elem.get("file")
            if old_path is None:
                continue

            old_path_split = old_path.split("/")
            # 尝试把所有指向 robosuite 资源的路径替换为绝对路径
            check_lst = [loc for loc, val in enumerate(old_path_split) if val == "robosuite"]
            if len(check_lst) > 0:
                ind = max(check_lst)  # 取最后一次出现的位置
                new_path_split = path_split + old_path_split[ind + 1 :]
                new_path = "/".join(new_path_split)
                elem.set("file", new_path)

        return ET.tostring(root, encoding="utf8").decode("utf8")

    def reset_from_xml_string(self, xml_string):
        """
        从环境的 XML 描述字符串重新加载环境。

        Args:
            xml_string (str): 将直接加载到仿真器中的 xml 文件路径
        """

        self.close()

        # 因为从 xml_string 重新加载，所以是确定性重置
        self.deterministic_reset = True

        # 从 xml 初始化仿真器
        self._initialize_sim(xml_string=xml_string)

        # 然后按正常流程 reset
        self.reset()

        # 关闭确定性重置标记
        self.deterministic_reset = False

    def update_state(self):
        """
        更新环境的内部状态(可在 reset 仿真之后或 step 之后调用)。
        子类可覆盖此方法以更新 site 等可视化状态。
        """
        pass

    def check_contact(self, geoms_1, geoms_2=None):
        """
        查找两个 geom 组之间的接触。

        Args:
            geoms_1 (str or list of str or MujocoModel): 单个 geom 名、geom 名列表或模型。
                若为 MujocoModel，则检查其 contact_geoms。
            geoms_2 (str or list of str or MujocoModel or None): 另一个 geom 名或列表。
                若为 MujocoModel，则检查其 contact_geoms。若为 None，则检查 @geoms_1
                与环境中其它任意 geom 的碰撞。
        Returns:
            bool: 若 @geoms_1 中任一 geom 与 @geoms_2 中任一 geom 接触则为 True。
        """
        return SU.check_contact(sim=self.sim, geoms_1=geoms_1, geoms_2=geoms_2)

    def get_contacts(self, model):
        """
        检查与 @model(由其 contact_geoms 定义)的任何接触，并返回当前与该模型接触的
        geom 名称集合(排除属于该模型自身的 geom)。

        Args:
            model (MujocoModel): 要检查接触的模型
        Returns:
            set: 当前与该模型接触的(去重)geom 名称集合
        Raises:
            AssertionError: [输入类型无效]
        """
        return SU.get_contacts(sim=self.sim, model=model)

    def add_observable(self, observable):
        """
        向本环境添加一个 observable。

        Args:
            observable (Observable): Observable 实例
        """
        assert observable.name not in self._observables, (
            "Observable name {} is already associated with an existing observable! Use modify_observable(...) "
            "to modify a pre-existing observable.".format(observable.name)
        )
        self._observables[observable.name] = observable

    def modify_observable(self, observable_name, attribute, modifier):
        """
        修改名为 @observable_name 的 observable，用 @modifier 替换指定的 @attribute。

        Args:
             observable_name (str): 要修改的 observable 名
             attribute (str): 要修改的 observable 属性。
                可选值 {`'sensor'`, `'corrupter'`,`'filter'`,  `'delayer'`, `'sampling_rate'`,
                `'enabled'`, `'active'`}
             modifier (any): 用于替换的新函数/值。若为函数，签名应与被替换函数匹配。
        """
        # 找到对应 observable
        assert observable_name in self._observables, "No valid observable with name {} found. Options are: {}".format(
            observable_name, self.observation_names
        )
        obs = self._observables[observable_name]
        # 按属性名替换对应内容
        if attribute == "sensor":
            obs.set_sensor(modifier)
        elif attribute == "corrupter":
            obs.set_corrupter(modifier)
        elif attribute == "filter":
            obs.set_filter(modifier)
        elif attribute == "delayer":
            obs.set_delayer(modifier)
        elif attribute == "sampling_rate":
            obs.set_sampling_rate(modifier)
        elif attribute == "enabled":
            obs.set_enabled(modifier)
        elif attribute == "active":
            obs.set_active(modifier)
        else:
            # 指定了无效属性
            raise ValueError(
                "Invalid observable attribute specified. Requested: {}, valid options are {}".format(
                    attribute, {"sensor", "corrupter", "filter", "delayer", "sampling_rate", "enabled", "active"}
                )
            )

    def _check_success(self):
        """
        检查任务是否完成。应由子类实现。

        Returns:
            bool: 任务是否完成
        """
        raise NotImplementedError

    def _destroy_viewer(self):
        """
        若当前的 mujoco 渲染器实例存在，则销毁它。
        """
        # 若存在活动的 viewer 窗口，则销毁
        if self.viewer is not None:
            self.viewer.close()  # change this to viewer.finish()?
            self.viewer = None

    def _destroy_sim(self):
        """
        若当前的 MjSim 实例存在，则销毁它。
        """
        if self.sim is not None:
            self.sim.free()
            self.sim = None

    def close(self):
        """在此处执行必要的清理工作。"""
        self._destroy_viewer()
        self._destroy_sim()

    @property
    def observation_modalities(self):
        """
        本环境观测的模态集合。

        Returns:
            set: 所有观测模态
        """
        return set([observable.modality for observable in self._observables.values()])

    @property
    def observation_names(self):
        """
        获取本环境所有 observables 的名称。

        Returns:
            set: 所有观测名
        """
        return set(self._observables.keys())

    @property
    def enabled_observables(self):
        """
        获取本环境所有已启用(enabled) observables 的名称。
        一个 observable 若在每个仿真时间步都持续计算/更新其值，则视为已启用。

        Returns:
            set: 所有已启用的观测名
        """
        return set([name for name, observable in self._observables.items() if observable.is_enabled()])

    @property
    def active_observables(self):
        """
        获取本环境所有活动(active) observables 的名称。
        一个 observable 若其值会在 _get_observations() 或 step() 返回的观测字典中出现
        (假设该 observable 已启用)，则视为活动。

        Returns:
            set: 所有活动观测名
        """
        return set([name for name, observable in self._observables.items() if observable.is_active()])

    @property
    def _visualizations(self):
        """
        本环境可用的可视化关键字。

        Returns:
            set: 本环境中可单独可视化的所有组件
        """
        return {"env"}

    @property
    def action_spec(self):
        """
        动作规范，应由子类实现。

        动作空间用 (low, high) 元组表示，二者均为 numpy 向量，分别给出每维动作
        的最小/最大限制。
        """
        raise NotImplementedError

    @property
    def action_dim(self):
        """
        动作空间维度。

        Returns:
            int: 动作空间维度
        """
        raise NotImplementedError
