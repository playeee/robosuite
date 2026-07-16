"""
This file implements a wrapper for facilitating domain randomization over
robosuite environments.
"""
# =============================================================================
# Sim2Real 核心：域随机化（Domain Randomization, DR）
# =============================================================================
# 为什么需要域随机化？
#   仿真器（MuJoCo）是真实世界的简化模型，存在"现实差距"（reality gap）：
#     - 物理参数不准：摩擦系数、质量、阻尼与真实机器人不同
#     - 视觉差异：渲染图像与真实相机拍摄的纹理、光照、颜色不同
#     - 传感器不完美：真实传感器有噪声、延迟、采样率限制
#
# 域随机化的核心思想：训练时主动随机化这些参数，让策略在"分布外"也能工作。
#   数学上：策略 π 学习最大化 E_{ξ~p(ξ)} [J(π, ξ)]，其中 ξ 是随机化的环境参数。
#   直觉上：如果策略能应对各种"奇怪的"仿真环境，真实世界只是其中一个特例。
#
# 本 Wrapper 提供了三类随机化，对应三种现实差距：
#   1. 视觉随机化（color/camera/lighting）：缩小"看起来不一样"的差距
#   2. 动力学随机化（dynamics）：缩小"物理行为不一样"的差距
#   3. 传感器随机化：见 Observable API（demo_sensor_corruption.py）
#
# 经典论文：OpenAI Dactyl (2018) 用 DR 训练仿真手，成功迁移到真实机器人转魔方。
# =============================================================================

import mujoco
import numpy as np

from robosuite.utils.log_utils import rs_assert
from robosuite.utils.mjmod import CameraModder, DynamicsModder, LightingModder, TextureModder
from robosuite.wrappers import Wrapper

DEFAULT_COLOR_ARGS = {
    # =========================================================================
    # 视觉随机化参数 1：颜色与纹理（解决"看起来不一样"的差距）
    # =========================================================================
    # 真实世界中物体颜色、纹理、反光特性千差万别：
    #   - 同样是"红色立方体"，真实拍摄的颜色受光照、相机白平衡影响
    #   - 物体表面可能是哑光、亮光、金属质感
    #
    # 通过颜色随机化，让策略学会忽略颜色细节，关注形状和位姿
    # =========================================================================
    "geom_names": None,  # 所有 geom 都被随机化（None 表示全部）
    "randomize_local": True,  # 在原颜色附近采样（小范围扰动，而非完全随机）
    "randomize_material": True,  # 随机化材质反光率/光泽度/镜面反射
    "local_rgb_interpolation": 0.2,  # RGB 扰动幅度（0=不变，1=完全随机）
    "local_material_interpolation": 0.3,  # 材质扰动幅度
    "texture_variations": ["rgb", "checker", "noise", "gradient"],  # 所有纹理变化类型
    "randomize_skybox": True,  # 默认也随机化天空盒（背景）
}

DEFAULT_CAMERA_ARGS = {
    # =========================================================================
    # 视觉随机化参数 2：相机外参与内参（解决"相机安装位置和参数不一样"的差距）
    # =========================================================================
    # 真实机器人上的相机安装存在误差：
    #   - 外参误差：相机位置和角度与 CAD 模型不完全一致（装配误差）
    #   - 内参误差：镜头焦距、视场角与标称值有偏差
    #
    # 通过相机随机化，让策略对相机参数的微小变化不敏感
    # 这对视觉 RL 的 sim2real 迁移至关重要
    # =========================================================================
    "camera_names": None,  # 所有相机都被随机化
    "randomize_position": True,  # 随机化相机位置（外参）
    "randomize_rotation": True,  # 随机化相机朝向（外参）
    "randomize_fovy": True,  # 随机化视场角（内参）
    "position_perturbation_size": 0.01,  # 位置扰动幅度 1cm
    "rotation_perturbation_size": 0.087,  # 旋转扰动幅度约 5度（弧度）
    "fovy_perturbation_size": 5.0,  # 视场角扰动幅度 5度
}

DEFAULT_LIGHTING_ARGS = {
    # =========================================================================
    # 视觉随机化参数 3：光照（解决"光照条件不一样"的差距）
    # =========================================================================
    # 真实世界的光照变化极大：
    #   - 室内/室外、白天/夜晚、阴天/晴天
    #   - 灯具类型、数量、位置不同
    #   - 阴影、反光、高光特性不同
    #
    # 光照随机化是 sim2real 中最有效的视觉随机化手段之一
    # OpenAI Dactyl 用了极度激进的纹理+光照随机化
    # =========================================================================
    "light_names": None,  # 所有光源都被随机化
    "randomize_position": True,  # 随机化光源位置
    "randomize_direction": True,  # 随机化光源方向
    "randomize_specular": True,  # 随机化镜面反射强度（高光）
    "randomize_ambient": True,  # 随机化环境光强度
    "randomize_diffuse": True,  # 随机化漫反射强度
    "randomize_active": True,  # 随机化光源是否开启
    "position_perturbation_size": 0.1,  # 光源位置扰动幅度
    "direction_perturbation_size": 0.35,  # 方向扰动幅度
    "specular_perturbation_size": 0.1,  # 镜面反射扰动幅度
    "ambient_perturbation_size": 0.1,  # 环境光扰动幅度
    "diffuse_perturbation_size": 0.1,  # 漫反射扰动幅度
}

DEFAULT_DYNAMICS_ARGS = {
    # =========================================================================
    # 动力学随机化参数——Sim2Real 中最重要的部分
    # 每个参数都对应真实世界中的一种不确定性来源
    # =========================================================================
    # Opt parameters（全局介质参数）
    # 介质密度/粘度影响空气阻力，真实世界中难以精确测量
    "randomize_density": True,
    "randomize_viscosity": True,
    "density_perturbation_ratio": 0.1,
    "viscosity_perturbation_ratio": 0.1,
    # Body parameters（刚体参数）
    # 质量/惯性不确定性来自：CAD 模型与实物差异、负载未知、装配误差
    "body_names": None,  # all bodies randomized
    "randomize_position": True,
    "randomize_quaternion": True,
    "randomize_inertia": True,
    "randomize_mass": True,
    "position_perturbation_size": 0.0015,
    "quaternion_perturbation_size": 0.003,
    "inertia_perturbation_ratio": 0.02,
    "mass_perturbation_ratio": 0.02,
    # Geom parameters（几何接触参数）
    # 摩擦系数是最重要的随机化对象——真实表面摩擦差异极大且难以精确测量
    # solref/solimp 控制软接触模型，影响抓取和碰撞行为
    "geom_names": None,  # all geoms randomized
    "randomize_friction": True,
    "randomize_solref": True,
    "randomize_solimp": True,
    "friction_perturbation_ratio": 0.1,
    "solref_perturbation_ratio": 0.1,
    "solimp_perturbation_ratio": 0.1,
    # Joint parameters（关节参数）
    # 阻尼/摩擦损失/armature 模拟电机和传动系统的不完美
    #   - damping:       关节阻尼（粘性摩擦）
    #   - frictionloss:  库仑摩擦（静摩擦）
    #   - armature:      电机转子惯量（影响加速度响应）
    "joint_names": None,  # all joints randomized
    "randomize_stiffness": True,
    "randomize_frictionloss": True,
    "randomize_damping": True,
    "randomize_armature": True,
    "stiffness_perturbation_ratio": 0.1,
    "frictionloss_perturbation_size": 0.05,
    "damping_perturbation_size": 0.01,
    "armature_perturbation_size": 0.01,
}


class DomainRandomizationWrapper(Wrapper):
    """
    允许在仿真过程中进行域随机化的包装器。

    参数:
        env (MujocoEnv): 要包装的环境。

        seed (int): 用于种子化本包装器所有随机化的整数。
            它被用来创建 np.random.RandomState 实例，确保此处的采样
            与代码其他地方的采样隔离。若未提供，将使用全局随机状态。

        randomize_color (bool): 若为 True，随机化 geom 颜色和纹理颜色

        randomize_camera (bool): 若为 True，随机化相机位置和参数

        randomize_lighting (bool): 若为 True，随机化光源位置和属性

        randomize_dyanmics (bool): 若为 True，随机化动力学参数

        color_randomization_args (dict): 颜色特定的随机化参数

        camera_randomization_args (dict): 相机特定的随机化参数

        lighting_randomization_args (dict): 光照特定的随机化参数

        dynamics_randomization_args (dict): 动力学特定的随机化参数

        randomize_on_reset (bool): 若为 True，每次调用 @reset 时随机化。
            这与设置 @randomize_every_n_steps 为 0 结合使用，
            可在每个 episode 生成一个新域。

        randomize_every_n_steps (int): 决定随机化发生的频率。
            设为 0 表示手动随机化（通过调用 @randomize_domain）

    """

    # =========================================================================
    # 【使用示例】如何在 RL 训练中使用域随机化
    # =========================================================================
    #
    #   # 1. 创建基础环境
    #   env = suite.make("Lift", robots="Panda", use_camera_obs=True, ...)
    #
    #   # 2. 用 DomainRandomizationWrapper 包装
    #   env = DomainRandomizationWrapper(
    #       env,
    #       randomize_color=True,      # 视觉随机化
    #       randomize_camera=True,     # 相机随机化
    #       randomize_lighting=True,   # 光照随机化
    #       randomize_dynamics=True,   # 动力学随机化（最重要）
    #       randomize_on_reset=True,   # 每个 episode 随机化一次
    #   )
    #
    #   # 3. 用 GymWrapper 包装以适配 RL 库
    #   env = GymWrapper(env)
    #
    #   # 4. 训练（策略会自动适应各种随机化的环境）
    #   model = SAC("MlpPolicy", env)
    #   model.learn(total_timesteps=1000000)
    #
    # 训练完成后，策略对环境参数变化具有鲁棒性，更容易迁移到真实机器人
    # =========================================================================

    def __init__(
        self,
        env,
        seed=None,
        randomize_color=True,
        randomize_camera=True,
        randomize_lighting=True,
        randomize_dynamics=True,
        color_randomization_args=DEFAULT_COLOR_ARGS,
        camera_randomization_args=DEFAULT_CAMERA_ARGS,
        lighting_randomization_args=DEFAULT_LIGHTING_ARGS,
        dynamics_randomization_args=DEFAULT_DYNAMICS_ARGS,
        randomize_on_reset=True,
        randomize_every_n_steps=1,
    ):
        super().__init__(env)

        self.seed = seed
        if seed is not None:
            self.random_state = np.random.RandomState(seed)
        else:
            self.random_state = None
        self.randomize_color = randomize_color
        self.randomize_camera = randomize_camera
        self.randomize_lighting = randomize_lighting
        self.randomize_dynamics = randomize_dynamics
        self.color_randomization_args = color_randomization_args
        self.camera_randomization_args = camera_randomization_args
        self.lighting_randomization_args = lighting_randomization_args
        self.dynamics_randomization_args = dynamics_randomization_args
        self.randomize_on_reset = randomize_on_reset
        self.randomize_every_n_steps = randomize_every_n_steps

        self.step_counter = 0

        self.modders = []

        if self.randomize_color:
            rs_assert(
                mujoco.__version__ == "3.1.1",
                (
                    "TextureModder requires mujoco version 3.1.1 to run. "
                    "Pending support for later versions. Alternatively, you can set randomize_color=False."
                ),
            )
            self.tex_modder = TextureModder(
                sim=self.env.sim, random_state=self.random_state, **self.color_randomization_args
            )
            self.modders.append(self.tex_modder)

        if self.randomize_camera:
            self.camera_modder = CameraModder(
                sim=self.env.sim,
                random_state=self.random_state,
                **self.camera_randomization_args,
            )
            self.modders.append(self.camera_modder)

        if self.randomize_lighting:
            self.light_modder = LightingModder(
                sim=self.env.sim,
                random_state=self.random_state,
                **self.lighting_randomization_args,
            )
            self.modders.append(self.light_modder)

        if self.randomize_dynamics:
            self.dynamics_modder = DynamicsModder(
                sim=self.env.sim,
                random_state=self.random_state,
                **self.dynamics_randomization_args,
            )
            self.modders.append(self.dynamics_modder)

        self.save_default_domain()

    def reset(self):
        """
        Extends superclass method to reset the domain randomizer.

        Returns:
            OrderedDict: Environment observation space after reset occurs
        """
        # =========================================================================
        # RL 训练中的随机化时机设计：
        #   - randomize_on_reset=True (默认)：每个 episode 开始时随机化一次
        #     优点：训练稳定，episode 内动力学一致，策略能学到连贯行为
        #     这是 RL 中最常用的设置——相当于每个 episode 对应一个"真实世界实例"
        #
        #   - randomize_every_n_steps=N：每 N 步随机化一次（甚至每步都随机化）
        #     优点：更强的鲁棒性，策略必须适应"实时变化"的环境
        #     缺点：训练更难收敛，episode 内动力学不连续，策略难以规划
        #     适用于：极端 sim2real 场景或 system identification 研究
        # =========================================================================
        # undo all randomizations
        self.restore_default_domain()

        # normal env reset
        ret = super().reset()

        # save the original env parameters
        self.save_default_domain()

        # reset counter for doing domain randomization at a particular frequency
        self.step_counter = 0

        # update sims
        for modder in self.modders:
            modder.update_sim(self.env.sim)

        if self.randomize_on_reset:
            # 每个 episode 开始时随机化环境参数
            # 关键：随机化后必须重新获取观测，因为视觉/相机参数已变
            self.randomize_domain()
            ret = self.env._get_observations()

        return ret

    def step(self, action):
        """
        Extends vanilla step() function call to accommodate domain randomization

        Returns:
            4-tuple:

                - (OrderedDict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        # 在执行动作前先更新随机化状态
        # 如果 randomize_every_n_steps > 0，会在 episode 中途动态改变环境
        # Step the internal randomization state
        self.step_randomization()

        return super().step(action)

    def step_randomization(self):
        """
        推进内部随机化状态。
        """
        # 按特定频率进行随机化的功能
        # =========================================================================
        # episode 内动态随机化的逻辑：
        #   - randomize_every_n_steps=0: 不在 step 中随机化（只在 reset 时随机化）
        #   - randomize_every_n_steps=1: 每步都随机化（最激进，但训练最不稳定）
        #   - randomize_every_n_steps=N: 每 N 步随机化一次（折中方案）
        #
        # 为什么要在 episode 中途随机化？
        #   - 更强的鲁棒性：策略必须适应"实时变化"的环境
        #   - 模拟真实场景中可能发生的参数漂移（如电机发热导致阻尼变化）
        #
        # 但代价是：
        #   - episode 内动力学不连续，策略难以规划长期行为
        #   - 训练更难收敛，可能需要更多样本
        # =========================================================================
        if self.randomize_every_n_steps > 0:
            if self.step_counter % self.randomize_every_n_steps == 0:
                self.randomize_domain()
        self.step_counter += 1

    def randomize_domain(self):
        """
        对环境运行域随机化。
        """
        # 遍历所有 modder，分别执行随机化
        # 每个 modder 负责一类参数：
        #   - TextureModder:  颜色和纹理
        #   - CameraModder:   相机外参和内参
        #   - LightingModder: 光照参数
        #   - DynamicsModder: 动力学参数（质量、摩擦、阻尼等）
        #
        # 这种"职责分离"设计便于灵活组合：
        #   - 只关心动力学：randomize_color=False, randomize_dynamics=True
        #   - 只关心视觉：  randomize_color=True,  randomize_dynamics=False
        for modder in self.modders:
            modder.randomize()

    def save_default_domain(self):
        """
        保存当前仿真模型参数，以便后续恢复。
        """
        # =========================================================================
        # 保存"原始"参数，用于在每次 reset 前恢复
        #
        # 为什么需要保存/恢复？
        #   - 随机化会修改 MuJoCo 模型参数（如摩擦系数、质量）
        #   - 这些修改是持久的，不会自动恢复
        #   - 若不恢复，下次随机化会基于"已随机化"的参数再次随机化
        #   - 导致参数分布漂移，偏离预期范围
        #
        # 正确流程（见 reset 方法）:
        #   1. restore_default_domain()  -> 恢复到原始参数
        #   2. env.reset()               -> 用原始参数 reset 环境
        #   3. save_default_domain()     -> 再次保存（以防 reset 改变了参数）
        #   4. randomize_domain()        -> 基于原始参数随机化
        # =========================================================================
        for modder in self.modders:
            modder.save_defaults()

    def restore_default_domain(self):
        """
        恢复上次调用 @save_default_domain 时保存的仿真模型参数。
        """
        # 恢复到保存的原始参数，避免随机化参数累积漂移
        for modder in self.modders:
            modder.restore_defaults()
