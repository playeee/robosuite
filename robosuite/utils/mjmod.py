"""
用于域随机化（Domain Randomization, DR）的 Modder 类集合。

本模块大量参考了 mujoco-py 的实现：
    https://github.com/openai/mujoco-py/blob/1fe312b09ae7365f0dd9d4d0e453f8da59fae0bf/mujoco_py/modder.py

=============================================================================
【核心概念】Modder 是什么？
=============================================================================

在 sim2real 中，我们希望智能体对仿真参数的变化具有鲁棒性。
Modder（修改器）的作用就是在仿真运行过程中，动态修改 MuJoCo 模型的各类属性：
    - LightingModder:   随机化光源的位置、方向、颜色、开关状态
    - CameraModder:     随机化相机的位置、朝向、视场角（fovy）
    - TextureModder:    随机化物体/天空盒的颜色与纹理（rgb / checker / noise / gradient）
    - DynamicsModder:   随机化物理动力学参数（摩擦、质量、惯量、阻尼、刚度等）

每个 Modder 都继承自 BaseModder，负责：
    1. save_defaults():   保存原始参数，便于恢复
    2. randomize():       按配置对参数进行随机扰动
    3. restore_defaults(): 恢复到原始参数

使用独立的 random_state 可以保证 Modder 内的随机采样不会影响外部其它随机过程。
"""

import copy
import os
from collections import defaultdict

import numpy as np
from PIL import Image

import robosuite
import robosuite.utils.transform_utils as trans
from robosuite.utils.binding_utils import MjRenderContextOffscreen


class BaseModder:
    """
    Modder 的基类，用于在仿真运行期间修改仿真属性。

    使用独立的 @random_state 可以确保 Modder 内部的采样不会受到 Modder 外部
    随机采样过程的影响，便于复现实验与控制随机性来源。

    Args:
        sim (MjSim): MuJoCo 仿真对象

        random_state (RandomState): np.random.RandomState 实例，用于对修改项进行随机化。
            若传入特定 RandomState，则 Modder 的随机性被隔离；若为 None，则使用全局随机状态。
    """

    def __init__(self, sim, random_state=None):
        self.sim = sim
        if random_state is None:
            # 若未指定随机状态，则退化为使用全局 RandomState 实例
            self.random_state = np.random.mtrand._rand
        else:
            self.random_state = random_state

    def update_sim(self, sim):
        """
        更新内部保存的仿真对象引用。

        Args:
            sim (MjSim): 新的 MjSim 对象
        """
        self.sim = sim

    @property
    def model(self):
        """
        Returns:
            MjModel: 当前 MjSim 的 MuJoCo 模型
        """
        # 提供快速便捷访问：直接返回 sim.model
        return self.sim.model


class LightingModder(BaseModder):
    """
    用于修改 MuJoCo 仿真中光照属性的 Modder。

    可对光源的位置、方向、镜面反射（specular）、环境光（ambient）、漫反射（diffuse）
    以及是否激活（active）等属性进行随机化，从而增强视觉策略对光照变化的鲁棒性。

    Args:
        sim (MjSim): MjSim 对象

        random_state (RandomState): np.random.RandomState 实例

        light_names (None 或 str 列表): 要参与随机化的光源名称列表。若为 None，则模型中所有光源都会被随机化。

        randomize_position (bool): 若为 True，随机化光源位置

        randomize_direction (bool): 若为 True，随机化光源方向

        randomize_specular (bool): 若为 True，随机化光源的镜面反射属性

        randomize_ambient (bool): 若为 True，随机化光源的环境光属性

        randomize_diffuse (bool): 若为 True，随机化光源的漫反射属性

        randomize_active (bool): 若为 True，随机化光源的开关状态

        position_perturbation_size (float): 位置随机化的扰动幅度

        direction_perturbation_size (float): 方向随机化的扰动幅度（默认 0.35 弧度约等于 20°）

        specular_perturbation_size (float): 镜面反射属性随机化的扰动幅度

        ambient_perturbation_size (float): 环境光属性随机化的扰动幅度

        diffuse_perturbation_size (float): 漫反射属性随机化的扰动幅度
    """

    def __init__(
        self,
        sim,
        random_state=None,
        light_names=None,
        randomize_position=True,
        randomize_direction=True,
        randomize_specular=True,
        randomize_ambient=True,
        randomize_diffuse=True,
        randomize_active=True,
        position_perturbation_size=0.1,
        direction_perturbation_size=0.35,  # 20 degrees
        specular_perturbation_size=0.1,
        ambient_perturbation_size=0.1,
        diffuse_perturbation_size=0.1,
    ):
        super().__init__(sim, random_state=random_state)

        if light_names is None:
            light_names = self.sim.model.light_names
        self.light_names = light_names

        self.randomize_position = randomize_position
        self.randomize_direction = randomize_direction
        self.randomize_specular = randomize_specular
        self.randomize_ambient = randomize_ambient
        self.randomize_diffuse = randomize_diffuse
        self.randomize_active = randomize_active

        self.position_perturbation_size = position_perturbation_size
        self.direction_perturbation_size = direction_perturbation_size
        self.specular_perturbation_size = specular_perturbation_size
        self.ambient_perturbation_size = ambient_perturbation_size
        self.diffuse_perturbation_size = diffuse_perturbation_size

        self.save_defaults()

    def save_defaults(self):
        """
        根据当前 MjSim 的状态与模型保存各光源的默认参数值。

        这些默认值后续用于：
            1. randomize(): 在默认值基础上加扰动
            2. restore_defaults(): 恢复原始光照配置
        """
        self._defaults = {k: {} for k in self.light_names}
        for name in self.light_names:
            self._defaults[name]["pos"] = np.array(self.get_pos(name))
            self._defaults[name]["dir"] = np.array(self.get_dir(name))
            self._defaults[name]["specular"] = np.array(self.get_specular(name))
            self._defaults[name]["ambient"] = np.array(self.get_ambient(name))
            self._defaults[name]["diffuse"] = np.array(self.get_diffuse(name))
            self._defaults[name]["active"] = self.get_active(name)

    def restore_defaults(self):
        """
        重新加载保存的默认参数值，恢复光照到随机化之前的状态。
        """
        for name in self.light_names:
            self.set_pos(name, self._defaults[name]["pos"])
            self.set_dir(name, self._defaults[name]["dir"])
            self.set_specular(name, self._defaults[name]["specular"])
            self.set_ambient(name, self._defaults[name]["ambient"])
            self.set_diffuse(name, self._defaults[name]["diffuse"])
            self.set_active(name, self._defaults[name]["active"])

    def randomize(self):
        """
        对仿真中所有请求随机化的光照属性进行随机化。
        """
        for name in self.light_names:
            if self.randomize_position:
                self._randomize_position(name)

            if self.randomize_direction:
                self._randomize_direction(name)

            if self.randomize_specular:
                self._randomize_specular(name)

            if self.randomize_ambient:
                self._randomize_ambient(name)

            if self.randomize_diffuse:
                self._randomize_diffuse(name)

            if self.randomize_active:
                self._randomize_active(name)

    def _randomize_position(self, name):
        """
        对指定光源的位置进行随机化。

        Args:
            name (str): 要随机化的光源名称
        """
        # 在默认值基础上叠加一个均匀分布的随机位置扰动
        delta_pos = self.random_state.uniform(
            low=-self.position_perturbation_size,
            high=self.position_perturbation_size,
            size=3,
        )
        self.set_pos(
            name,
            self._defaults[name]["pos"] + delta_pos,
        )

    def _randomize_direction(self, name):
        """
        对指定光源的方向进行随机化。

        Args:
            name (str): 要随机化的光源名称
        """
        # 采样一个小的随机轴角增量旋转，用于扰动光源方向
        random_axis, random_angle = trans.random_axis_angle(
            angle_limit=self.direction_perturbation_size, random_state=self.random_state
        )
        random_delta_rot = trans.quat2mat(trans.axisangle2quat(random_axis * random_angle))

        # 用该增量旋转对默认方向进行旋转，得到新的方向
        new_dir = random_delta_rot.dot(self._defaults[name]["dir"])
        self.set_dir(
            name,
            new_dir,
        )

    def _randomize_specular(self, name):
        """
        对指定光源的镜面反射（specular）属性进行随机化。

        Args:
            name (str): 要随机化的光源名称
        """
        delta = self.random_state.uniform(
            low=-self.specular_perturbation_size,
            high=self.specular_perturbation_size,
            size=3,
        )
        self.set_specular(
            name,
            self._defaults[name]["specular"] + delta,
        )

    def _randomize_ambient(self, name):
        """
        对指定光源的环境光（ambient）属性进行随机化。

        Args:
            name (str): 要随机化的光源名称
        """
        delta = self.random_state.uniform(
            low=-self.ambient_perturbation_size,
            high=self.ambient_perturbation_size,
            size=3,
        )
        self.set_ambient(
            name,
            self._defaults[name]["ambient"] + delta,
        )

    def _randomize_diffuse(self, name):
        """
        对指定光源的漫反射（diffuse）属性进行随机化。

        Args:
            name (str): 要随机化的光源名称
        """
        delta = self.random_state.uniform(
            low=-self.diffuse_perturbation_size,
            high=self.diffuse_perturbation_size,
            size=3,
        )
        self.set_diffuse(
            name,
            self._defaults[name]["diffuse"] + delta,
        )

    def _randomize_active(self, name):
        """
        随机决定指定光源是否激活（开/关）。

        Args:
            name (str): 要随机化的光源名称
        """
        # 以 0.5 概率决定光源开启或关闭
        active = int(self.random_state.uniform() > 0.5)
        self.set_active(name, active)

    def get_pos(self, name):
        """
        获取指定光源的位置。

        Args:
            name (str): 光源名称

        Returns:
            np.array: 光源的 (x, y, z) 位置

        Raises:
            AssertionError: 光源名称无效
        """
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name

        return self.model.light_pos[lightid]

    def set_pos(self, name, value):
        """
        设置指定光源的位置。

        Args:
            name (str): 光源名称
            value (np.array): 要设置的光源 (x, y, z) 位置

        Raises:
            AssertionError: 光源名称无效
            AssertionError: value 维度无效
        """
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name

        value = list(value)
        assert len(value) == 3, "Expected 3-dim value, got %s" % value

        self.model.light_pos[lightid] = value

    def get_dir(self, name):
        """
        获取指定光源的方向。

        Args:
            name (str): 光源名称

        Returns:
            np.array: 光源的 (x, y, z) 方向向量

        Raises:
            AssertionError: 光源名称无效
        """
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name

        return self.model.light_dir[lightid]

    def set_dir(self, name, value):
        """
        Sets direction of a specific light source

        Args:
            name (str): 光源名称
            value (np.array): 要设置的光源方向向量 (ax, ay, az)

        Raises:
            AssertionError: 光源名称无效
            AssertionError: value 维度无效
        """
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name

        value = list(value)
        assert len(value) == 3, "Expected 3-dim value, got %s" % value

        self.model.light_dir[lightid] = value

    def get_active(self, name):
        """
        获取指定光源是否处于激活状态。

        Args:
            name (str): 光源名称

        Returns:
            int: 光源是否激活，1 表示开启，0 表示关闭

        Raises:
            AssertionError: 光源名称无效
        """
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name

        return self.model.light_active[lightid]

    def set_active(self, name, value):
        """
        设置指定光源的激活状态。

        Args:
            name (str): 光源名称
            value (int): 1 表示开启，0 表示关闭

        Raises:
            AssertionError: 光源名称无效
        """
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name

        self.model.light_active[lightid] = value

    def get_specular(self, name):
        """
        获取指定光源的镜面反射（specular）颜色。

        Args:
            name (str): 光源名称

        Returns:
            np.array: 光源的 (r, g, b) 镜面反射颜色

        Raises:
            AssertionError: 光源名称无效
        """
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name

        return self.model.light_specular[lightid]

    def set_specular(self, name, value):
        """
        设置指定光源的镜面反射（specular）颜色。

        Args:
            name (str): 光源名称
            value (np.array): 要设置的 (r, g, b) 镜面反射颜色

        Raises:
            AssertionError: 光源名称无效
            AssertionError: value 维度无效
        """
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name

        value = list(value)
        assert len(value) == 3, "Expected 3-dim value, got %s" % value

        self.model.light_specular[lightid] = value

    def get_ambient(self, name):
        """
        获取指定光源的环境光（ambient）颜色。

        Args:
            name (str): 光源名称

        Returns:
            np.array: 光源的 (r, g, b) 环境光颜色

        Raises:
            AssertionError: 光源名称无效
        """
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name

        return self.model.light_ambient[lightid]

    def set_ambient(self, name, value):
        """
        设置指定光源的环境光（ambient）颜色。

        Args:
            name (str): 光源名称
            value (np.array): 要设置的 (r, g, b) 环境光颜色

        Raises:
            AssertionError: 光源名称无效
            AssertionError: value 维度无效
        """
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name

        value = list(value)
        assert len(value) == 3, "Expected 3-dim value, got %s" % value

        self.model.light_ambient[lightid] = value

    def get_diffuse(self, name):
        """
        获取指定光源的漫反射（diffuse）颜色。

        Args:
            name (str): 光源名称

        Returns:
            np.array: 光源的 (r, g, b) 漫反射颜色

        Raises:
            AssertionError: 光源名称无效
        """
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name

        return self.model.light_diffuse[lightid]

    def set_diffuse(self, name, value):
        """
        设置指定光源的漫反射（diffuse）颜色。

        Args:
            name (str): 光源名称
            value (np.array): 要设置的 (r, g, b) 漫反射颜色

        Raises:
            AssertionError: 光源名称无效
            AssertionError: value 维度无效
        """
        lightid = self.get_lightid(name)
        assert lightid > -1, "Unkwnown light %s" % name

        value = list(value)
        assert len(value) == 3, "Expected 3-dim value, got %s" % value

        self.model.light_diffuse[lightid] = value

    def get_lightid(self, name):
        """
        获取指定光源在模型中的唯一 id。

        Args:
            name (str): 光源名称

        Returns:
            int: 光源 id。若未找到则返回 -1
        """
        return self.model.light_name2id(name)


class CameraModder(BaseModder):
    """
    用于修改 MuJoCo 仿真中相机属性的 Modder。

    相机的位置、朝向、视场角（fovy）等参数会被随机化，
    从而让视觉策略对不同相机标定误差和拍摄角度具有鲁棒性。

    Args:
        sim (MjSim): MjSim 对象

        random_state (None 或 RandomState): np.random.RandomState 实例

        camera_names (None 或 str 列表): 要参与随机化的相机名称列表。若为 None，则所有相机都会被随机化。

        randomize_position (bool): 若为 True，随机化相机位置

        randomize_rotation (bool): 若为 True，随机化相机朝向

        randomize_fovy (bool): 若为 True，随机化相机垂直视场角

        position_perturbation_size (float): 相机位置每个维度的扰动幅度

        rotation_perturbation_size (float): 相机朝向轴角随机化的幅度（弧度）。
            默认值 0.087 约等于 5°。

        fovy_perturbation_size (float): 相机垂直视场角（fovy）的扰动幅度，对应焦距/视野的变化

    Raises:
        AssertionError: [未选择任何随机化项]
    """

    def __init__(
        self,
        sim,
        random_state=None,
        camera_names=None,
        randomize_position=True,
        randomize_rotation=True,
        randomize_fovy=True,
        position_perturbation_size=0.01,
        rotation_perturbation_size=0.087,
        fovy_perturbation_size=5.0,
    ):
        super().__init__(sim, random_state=random_state)

        # 必须至少选择一项相机属性进行随机化，否则无意义
        assert randomize_position or randomize_rotation or randomize_fovy

        if camera_names is None:
            camera_names = self.sim.model.camera_names
        self.camera_names = camera_names

        self.randomize_position = randomize_position
        self.randomize_rotation = randomize_rotation
        self.randomize_fovy = randomize_fovy

        self.position_perturbation_size = position_perturbation_size
        self.rotation_perturbation_size = rotation_perturbation_size
        self.fovy_perturbation_size = fovy_perturbation_size

        self.save_defaults()

    def save_defaults(self):
        """
        保存当前相机的默认位置、朝向和视场角。
        """
        self._defaults = {k: {} for k in self.camera_names}
        for camera_name in self.camera_names:
            self._defaults[camera_name]["pos"] = np.array(self.get_pos(camera_name))
            self._defaults[camera_name]["quat"] = np.array(self.get_quat(camera_name))
            self._defaults[camera_name]["fovy"] = self.get_fovy(camera_name)

    def restore_defaults(self):
        """
        恢复相机的默认参数值。
        """
        for camera_name in self.camera_names:
            self.set_pos(camera_name, self._defaults[camera_name]["pos"])
            self.set_quat(camera_name, self._defaults[camera_name]["quat"])
            self.set_fovy(camera_name, self._defaults[camera_name]["fovy"])

    def randomize(self):
        """
        对仿真中所有请求随机化的相机属性进行随机化。
        """
        for camera_name in self.camera_names:
            if self.randomize_position:
                self._randomize_position(camera_name)

            if self.randomize_rotation:
                self._randomize_rotation(camera_name)

            if self.randomize_fovy:
                self._randomize_fovy(camera_name)

    def _randomize_position(self, name):
        """
        对指定相机的位置进行随机化。

        Args:
            name (str): 要随机化的相机名称
        """
        delta_pos = self.random_state.uniform(
            low=-self.position_perturbation_size,
            high=self.position_perturbation_size,
            size=3,
        )
        self.set_pos(
            name,
            self._defaults[name]["pos"] + delta_pos,
        )

    def _randomize_rotation(self, name):
        """
        对指定相机的朝向进行随机化。

        Args:
            name (str): 要随机化的相机名称
        """
        # 采样一个小的随机轴角增量旋转
        random_axis, random_angle = trans.random_axis_angle(
            angle_limit=self.rotation_perturbation_size, random_state=self.random_state
        )
        random_delta_rot = trans.quat2mat(trans.axisangle2quat(random_axis * random_angle))

        # 用该增量旋转对默认朝向进行扰动，并更新相机四元数
        # 注意 MuJoCo 使用 wxyz 四元数约定，因此中间计算需要转换
        base_rot = trans.quat2mat(trans.convert_quat(self._defaults[name]["quat"], to="xyzw"))
        new_rot = random_delta_rot.T.dot(base_rot)
        new_quat = trans.convert_quat(trans.mat2quat(new_rot), to="wxyz")
        self.set_quat(
            name,
            new_quat,
        )

    def _randomize_fovy(self, name):
        """
        对指定相机的垂直视场角（fovy）进行随机化。

        Args:
            name (str): 要随机化的相机名称
        """
        delta_fovy = self.random_state.uniform(
            low=-self.fovy_perturbation_size,
            high=self.fovy_perturbation_size,
        )
        self.set_fovy(
            name,
            self._defaults[name]["fovy"] + delta_fovy,
        )

    def get_fovy(self, name):
        """
        获取指定相机的垂直视场角（fovy）。

        Args:
            name (str): 相机名称

        Returns:
            float: 相机的垂直视场角，单位为度

        Raises:
            AssertionError: 相机名称无效
        """
        camid = self.get_camid(name)
        assert camid > -1, "Unknown camera %s" % name
        return self.model.cam_fovy[camid]

    def set_fovy(self, name, value):
        """
        设置指定相机的垂直视场角（fovy）。

        Args:
            name (str): 相机名称
            value (float): 相机的垂直视场角，单位为度

        Raises:
            AssertionError: 相机名称无效
            AssertionError: value 不合法
        """
        camid = self.get_camid(name)
        assert 0 < value < 180
        assert camid > -1, "Unknown camera %s" % name
        self.model.cam_fovy[camid] = value

    def get_quat(self, name):
        """
        获取指定相机的朝向四元数。

        Args:
            name (str): 相机名称

        Returns:
            np.array: 相机的 (w, x, y, z) 四元数朝向

        Raises:
            AssertionError: 相机名称无效
        """
        camid = self.get_camid(name)
        assert camid > -1, "Unknown camera %s" % name
        return self.model.cam_quat[camid]

    def set_quat(self, name, value):
        """
        设置指定相机的朝向四元数。

        Args:
            name (str): 相机名称
            value (np.array): 相机的 (w, x, y, z) 四元数朝向

        Raises:
            AssertionError: 相机名称无效
            AssertionError: value 维度无效
        """
        value = list(value)
        assert len(value) == 4, "Expectd value of length 4, instead got %s" % value
        camid = self.get_camid(name)
        assert camid > -1, "Unknown camera %s" % name
        self.model.cam_quat[camid] = value

    def get_pos(self, name):
        """
        获取指定相机的位置。

        Args:
            name (str): 相机名称

        Returns:
            np.array: 相机的 (x, y, z) 位置

        Raises:
            AssertionError: 相机名称无效
        """
        camid = self.get_camid(name)
        assert camid > -1, "Unknown camera %s" % name
        return self.model.cam_pos[camid]

    def set_pos(self, name, value):
        """
        设置指定相机的位置。

        Args:
            name (str): 相机名称
            value (np.array): 相机的 (x, y, z) 位置

        Raises:
            AssertionError: 相机名称无效
            AssertionError: value 维度无效
        """
        value = list(value)
        assert len(value) == 3, "Expected value of length 3, instead got %s" % value
        camid = self.get_camid(name)
        assert camid > -1
        self.model.cam_pos[camid] = value

    def get_camid(self, name):
        """
        获取指定相机在模型中的唯一 id。

        Args:
            name (str): 相机名称

        Returns:
            int: 相机 id。若未找到则返回 -1
        """
        return self.model.camera_name2id(name)


class TextureModder(BaseModder):
    """
    用于修改 MuJoCo 模型中纹理（texture）的 Modder。

    示例用法：
        sim = MjSim(...)
        modder = TextureModder(sim)
        modder.whiten_materials()  # 先把材质颜色设为白色，避免纹理颜色被材质调制
        modder.set_checker('some_geom', (255, 0, 0), (0, 0, 0))
        modder.rand_all('another_geom')

    注意：为了让纹理效果完全生效，需要将所有材质的 rgba 设为 [1, 1, 1, 1]，
          否则纹理颜色会被材质颜色调制。可调用 `whiten_materials` 辅助方法
          将所有材质颜色设为白色。

    Args:
        sim (MjSim): MjSim 对象

        random_state (RandomState): np.random.RandomState 实例

        geom_names (str 列表): 要参与随机化的 geom 名称列表。若为 None，则所有 geom 都会被随机化。

        randomize_local (bool): 若为 True，颜色变化会被限制在原始 RGB 附近；
            否则 RGB 颜色将在 [0, 1] 范围内完全随机采样。

        randomize_material (bool): 若为 True，同时随机化材质属性
            （reflectance、shininess、specular）。

        local_rgb_interpolation (float): 当 @randomize_local 为 True 时，
            控制颜色相对于基础 geom 颜色的扰动幅度。

        local_material_interpolation (float): 当 @randomize_local 和 @randomize_material
            都为 True 时，控制材质相对于基础材质的扰动幅度。

        texture_variations (str 列表): 纹理变化方式列表。每个字符串必须是
            'rgb'、'checker'、'noise'、'gradient' 之一，对应一种纹理随机化方式。
            对于每个拥有材质和纹理的 geom，会从该列表中随机选择一种方式应用。

        randomize_skybox (bool): 若为 True，同样对天空盒（skybox）应用纹理随机化。
    """

    def __init__(
        self,
        sim,
        random_state=None,
        geom_names=None,
        randomize_local=False,
        randomize_material=False,
        local_rgb_interpolation=0.1,
        local_material_interpolation=0.2,
        texture_variations=("rgb", "checker", "noise", "gradient"),
        randomize_skybox=True,
    ):
        super().__init__(sim, random_state=random_state)

        if geom_names is None:
            geom_names = self.sim.model.geom_names
        self.geom_names = geom_names

        self.randomize_local = randomize_local
        self.randomize_material = randomize_material
        self.local_rgb_interpolation = local_rgb_interpolation
        self.local_material_interpolation = local_material_interpolation
        self.texture_variations = list(texture_variations)
        self.randomize_skybox = randomize_skybox

        # 注册所有支持的纹理变化回调函数
        self._all_texture_variation_callbacks = {
            "rgb": self.rand_rgb,
            "checker": self.rand_checker,
            "noise": self.rand_noise,
            "gradient": self.rand_gradient,
        }
        # 只保留用户指定的纹理变化方式
        self._texture_variation_callbacks = {
            k: self._all_texture_variation_callbacks[k] for k in self.texture_variations
        }

        self.save_defaults()

    def save_defaults(self):
        """
        保存当前模型中所有纹理、材质和 geom 颜色的默认值。
        """
        # 为每个纹理创建 Texture 辅助对象
        self.textures = [Texture(self.model, i) for i in range(self.model.ntex)]
        # self._build_tex_geom_map()

        # 保存原始纹理位图的副本，作为后续随机化的基准
        self._default_texture_bitmaps = [np.array(text.bitmap) for text in self.textures]

        # 缓存棋盘格矩阵，用于快速生成 checker 纹理
        self._cache_checker_matrices()

        # 保存每个 geom 的默认纹理或颜色
        self._defaults = {k: {} for k in self.geom_names}
        if self.randomize_skybox:
            self._defaults["skybox"] = {}
        for name in self.geom_names:
            if self._check_geom_for_texture(name):
                # 若 geom 有纹理，保存其纹理位图
                tex_id = self._name_to_tex_id(name)
                self._defaults[name]["texture"] = self._default_texture_bitmaps[tex_id]
                # 同时保存材质属性（reflectance, shininess, specular）
                self._defaults[name]["material"] = self.get_material(name)
            else:
                # 若 geom 没有纹理，保存其 RGB 颜色
                self._defaults[name]["rgb"] = np.array(self.get_geom_rgb(name))

        if self.randomize_skybox:
            tex_id = self._name_to_tex_id("skybox")
            self._defaults["skybox"]["texture"] = self._default_texture_bitmaps[tex_id]

    def restore_defaults(self):
        """
        恢复所有纹理、材质和 geom 颜色到保存的默认值。
        """
        for name in self.geom_names:
            if self._check_geom_for_texture(name):
                self.set_texture(name, self._defaults[name]["texture"], perturb=False)
                self.set_material(name, self._defaults[name]["material"], perturb=False)
            else:
                self.set_geom_rgb(name, self._defaults[name]["rgb"])

        if self.randomize_skybox:
            self.set_texture("skybox", self._defaults["skybox"]["texture"], perturb=False)

    def randomize(self):
        """
        对所有 geom 的纹理/颜色以及天空盒进行随机化。

        本实现扩展了 mujoco-py 原版：即使 geom 没有材质（material），
        也会直接对其 geom_rgba 颜色进行随机化。
        """
        # 先把所有材质和 geom 颜色"漂白"为白色，确保纹理颜色不被材质调制
        self.whiten_materials()
        for name in self.geom_names:
            if self._check_geom_for_texture(name):
                # 若 geom 有合法纹理，则随机化纹理
                self._randomize_texture(name)
                # 若需要，同时随机化材质属性
                if self.randomize_material:
                    self._randomize_material(name)
            else:
                # 否则直接随机化 geom 颜色
                self._randomize_geom_color(name)

        if self.randomize_skybox:
            self._randomize_texture("skybox")

    def _randomize_geom_color(self, name):
        """
        对指定 geom 的颜色进行随机化。

        Args:
            name (str): 要随机化的 geom 名称
        """
        if self.randomize_local:
            # 局部随机化：在原始颜色与随机颜色之间线性插值
            random_color = self.random_state.uniform(0, 1, size=3)
            rgb = (1.0 - self.local_rgb_interpolation) * self._defaults[name][
                "rgb"
            ] + self.local_rgb_interpolation * random_color
        else:
            # 全局随机化：直接采样新的 RGB 颜色
            rgb = self.random_state.uniform(0, 1, size=3)
        self.set_geom_rgb(name, rgb)

    def _randomize_texture(self, name):
        """
        对指定 geom 的纹理进行随机化（随机选择一种纹理变化方式）。

        Args:
            name (str): 要随机化的 geom 名称
        """
        keys = list(self._texture_variation_callbacks.keys())
        choice = keys[self.random_state.randint(len(keys))]
        self._texture_variation_callbacks[choice](name)

    def _randomize_material(self, name):
        """
        对指定 geom 的材质属性进行随机化。

        Args:
            name (str): 要随机化的 geom 名称
        """
        # 天空盒没有材质，直接返回
        if name == "skybox":
            return
        # 获取材质 id
        mat_id = self._name_to_mat_id(name)
        # 随机采样 reflectance、shininess、specular
        material = self.random_state.uniform(0, 1, size=3)  # (reflectance, shininess, specular)
        self.set_material(name, material, perturb=self.randomize_local)

    def rand_checker(self, name):
        """
        为指定 geom 生成随机棋盘格（checker）纹理。

        Args:
            name (str): 要随机化的 geom 名称
        """
        rgb1, rgb2 = self.get_rand_rgb(2)
        self.set_checker(name, rgb1, rgb2, perturb=self.randomize_local)

    def rand_gradient(self, name):
        """
        为指定 geom 生成随机渐变（gradient）纹理。

        Args:
            name (str): 要随机化的 geom 名称
        """
        rgb1, rgb2 = self.get_rand_rgb(2)
        vertical = bool(self.random_state.uniform() > 0.5)
        self.set_gradient(name, rgb1, rgb2, vertical=vertical, perturb=self.randomize_local)

    def rand_rgb(self, name):
        """
        为指定 geom 生成随机纯色 RGB 纹理。

        Args:
            name (str): 要随机化的 geom 名称
        """
        rgb = self.get_rand_rgb()
        self.set_rgb(name, rgb, perturb=self.randomize_local)

    def rand_noise(self, name):
        """
        为指定 geom 生成随机噪声（noise）纹理。

        Args:
            name (str): 要随机化的 geom 名称
        """
        # 前景噪声像素占比：在 0.1 ~ 0.9 之间随机
        fraction = 0.1 + self.random_state.uniform() * 0.8
        rgb1, rgb2 = self.get_rand_rgb(2)
        self.set_noise(name, rgb1, rgb2, fraction, perturb=self.randomize_local)

    def whiten_materials(self):
        """
        辅助方法：将所有 geom 和材质的颜色设为白色。

        如果不把材质颜色漂白，纹理颜色会被材质颜色调制，导致随机化效果不理想。
        本方法扩展了 mujoco-py 原版，同时漂白了 geom_rgba。
        """
        for name in self.geom_names:
            # 将 geom 本身的颜色设为白色
            geom_id = self.model.geom_name2id(name)
            self.model.geom_rgba[geom_id, :] = 1.0

            if self._check_geom_for_texture(name):
                # 若 geom 有材质，将材质颜色也设为白色
                mat_id = self.model.geom_matid[geom_id]
                self.model.mat_rgba[mat_id, :] = 1.0

    def get_geom_rgb(self, name):
        """
        获取指定 geom 的 RGB 颜色。

        Args:
            name (str): geom 名称

        Returns:
            np.array: geom 的 (r, g, b) 颜色
        """
        geom_id = self.model.geom_name2id(name)
        return self.model.geom_rgba[geom_id, :3]

    def set_geom_rgb(self, name, rgb):
        """
        设置指定 geom 的 RGB 颜色。

        Args:
            name (str): geom 名称
            rgb (np.array): geom 的 (r, g, b) 颜色
        """
        geom_id = self.model.geom_name2id(name)
        self.model.geom_rgba[geom_id, :3] = rgb

    def get_rand_rgb(self, n=1):
        """
        随机生成一组或多组 RGB 颜色元组。

        Args:
            n (int): 要生成的 RGB 元组数量

        Returns:
            np.array 或 n 元组: 若 n > 1，返回 n 个 RGB 元组；否则返回单个 (r, g, b) 数组
        """

        def _rand_rgb():
            # 在 [0, 255] 范围内采样 uint8 类型的 RGB 值
            return np.array(self.random_state.uniform(size=3) * 255, dtype=np.uint8)

        if n == 1:
            return _rand_rgb()
        else:
            return tuple(_rand_rgb() for _ in range(n))

    def get_texture(self, name):
        """
        获取与指定 geom 关联的 Texture 对象。

        Args:
            name (str): geom 名称

        Returns:
            Texture: 与 geom 关联的纹理对象
        """
        tex_id = self._name_to_tex_id(name)
        texture = self.textures[tex_id]
        return texture

    def set_texture(self, name, bitmap, perturb=False):
        """
        设置与 geom @name 对应的纹理位图。

        若 @perturb 为 True，则使用计算出的位图对默认位图进行轻微扰动，
        而不是完全替换它。

        Args:
            name (str): geom 名称
            bitmap (np.array): 表示每个像素 RGB 值的三维数组
            perturb (bool): 是否对输入位图进行扰动
        """
        bitmap_to_set = self.get_texture(name).bitmap
        if perturb:
            # 局部扰动：在默认纹理与目标纹理之间插值
            bitmap = (1.0 - self.local_rgb_interpolation) * self._defaults[name][
                "texture"
            ] + self.local_rgb_interpolation * bitmap
        bitmap_to_set[:] = bitmap
        # 修改位图后需要上传到 GPU，渲染时才能看到变化
        self.upload_texture(name)

    def get_material(self, name):
        """
        获取与指定 geom 关联的材质属性。

        Args:
            name (str): geom 名称

        Returns:
            np.array: geom 的材质属性 (reflectance, shininess, specular)
        """
        mat_id = self._name_to_mat_id(name)
        # 材质以 (reflectance, shininess, specular) 元组形式返回
        material = np.array(
            (self.model.mat_reflectance[mat_id], self.model.mat_shininess[mat_id], self.model.mat_specular[mat_id])
        )
        return material

    def set_material(self, name, material, perturb=False):
        """
        设置与 geom @name 对应的材质属性。

        若 @perturb 为 True，则使用计算出的材质对默认材质进行轻微扰动，
        而不是完全替换它。

        Args:
            name (str): geom 名称
            material (np.array): geom 的材质属性 (reflectance, shininess, specular)
            perturb (bool): 是否对输入材质属性进行扰动
        """
        mat_id = self._name_to_mat_id(name)
        if perturb:
            # 局部扰动：在默认材质与新材质之间插值
            material = (1.0 - self.local_material_interpolation) * self._defaults[name][
                "material"
            ] + self.local_material_interpolation * material
        self.model.mat_reflectance[mat_id] = material[0]
        self.model.mat_shininess[mat_id] = material[1]
        self.model.mat_specular[mat_id] = material[2]

    def get_checker_matrices(self, name):
        """
        获取与 @name 关联的棋盘格模式矩阵。

        Args:
            name (str): geom 名称

        Returns:
            np.array: 表示 RGB 棋盘格模式的三维数组
        """
        tex_id = self._name_to_tex_id(name)
        return self._texture_checker_mats[tex_id]

    def set_checker(self, name, rgb1, rgb2, perturb=False):
        """
        使用两个棋盘格矩阵和两种颜色，为 geom @name 创建棋盘格纹理。

        Args:
            name (str): geom 名称
            rgb1 (3 数组): 棋盘格中一半格子的 (r, g, b) 颜色
            rgb2 (3 数组): 棋盘格中另一半格子的 (r, g, b) 颜色
            perturb (bool): 是否对最终棋盘格纹理进行扰动
        """
        cbd1, cbd2 = self.get_checker_matrices(name)
        rgb1 = np.asarray(rgb1).reshape([1, 1, -1])
        rgb2 = np.asarray(rgb2).reshape([1, 1, -1])
        bitmap = rgb1 * cbd1 + rgb2 * cbd2

        self.set_texture(name, bitmap, perturb=perturb)

    def set_gradient(self, name, rgb1, rgb2, vertical=True, perturb=False):
        """
        从 rgb1 到 rgb2 创建线性渐变纹理。

        Args:
            name (str): geom 名称
            rgb1 (3 数组): 渐变起始颜色
            rgb2 (3 数组): 渐变结束颜色
            vertical (bool): 若为 True，渐变沿正 y 方向；否则沿正 x 方向。
            perturb (bool): 是否对最终渐变纹理进行扰动
        """
        # 注意：MuJoCo 原生渐变使用 sigmoid，这里简化为线性渐变。
        # 如需更接近原生效果，可改用 tanh-sigmoid。
        bitmap = self.get_texture(name).bitmap
        h, w = bitmap.shape[:2]
        if vertical:
            p = np.tile(np.linspace(0, 1, h)[:, None], (1, w))
        else:
            p = np.tile(np.linspace(0, 1, w), (h, 1))

        new_bitmap = np.zeros_like(bitmap)
        for i in range(3):
            new_bitmap[..., i] = rgb2[i] * p + rgb1[i] * (1.0 - p)

        self.set_texture(name, new_bitmap, perturb=perturb)

    def set_rgb(self, name, rgb, perturb=False):
        """
        将 geom @name 的纹理位图设置为纯色 RGB。

        Args:
            name (str): geom 名称
            rgb (3 数组): 目标 (r, g, b) 颜色
            perturb (bool): 是否对最终颜色进行扰动
        """
        bitmap = self.get_texture(name).bitmap
        new_bitmap = np.zeros_like(bitmap)
        new_bitmap[..., :] = np.asarray(rgb)

        self.set_texture(name, new_bitmap, perturb=perturb)

    def set_noise(self, name, rgb1, rgb2, fraction=0.9, perturb=False):
        """
        将 geom @name 的纹理位图设置为噪声模式。

        Args:
            name (str): geom 名称
            rgb1 (3 数组): 背景颜色
            rgb2 (3 数组): 前景噪声颜色
            fraction (float): 前景噪声像素所占比例
            perturb (bool): 是否对最终噪声纹理进行扰动
        """
        bitmap = self.get_texture(name).bitmap
        h, w = bitmap.shape[:2]
        # 根据 fraction 生成随机 mask，决定哪些像素使用前景色
        mask = self.random_state.uniform(size=(h, w)) < fraction

        new_bitmap = np.zeros_like(bitmap)
        new_bitmap[..., :] = np.asarray(rgb1)
        new_bitmap[mask, :] = np.asarray(rgb2)

        self.set_texture(name, new_bitmap, perturb=perturb)

    def upload_texture(self, name, device_id=0):
        """
        将修改后的纹理上传到 GPU，使其在渲染中可见。

        Args:
            name (str): geom 名称
        """
        texture = self.get_texture(name)
        # 若离屏渲染上下文尚未创建，则临时创建一个用于上传纹理
        if self.sim._render_context_offscreen is None:
            render_context = MjRenderContextOffscreen(self.sim, device_id)
            render_context.upload_texture(texture.id)

    def _check_geom_for_texture(self, name):
        """
        判断指定 geom 是否同时关联了材质（material）和纹理（texture）。

        Args:
            name (str): geom 名称

        Returns:
            bool: 若 geom 同时关联了材质和纹理则返回 True，否则返回 False
        """
        geom_id = self.model.geom_name2id(name)
        mat_id = self.model.geom_matid[geom_id]
        if mat_id < 0:
            return False
        tex_id = self.model.mat_texid[mat_id]
        if tex_id < 0:
            return False
        return True

    def _name_to_tex_id(self, name):
        """
        根据 geom 名称获取对应的纹理 id。

        Args:
            name (str): geom 名称

        Returns:
            int: 与 geom 关联的纹理 id

        Raises:
            AssertionError: [geom 没有关联纹理]
        """

        # 天空盒需要单独处理：遍历所有纹理，找到类型为 skybox（type=2）的纹理
        if name == "skybox":
            skybox_tex_id = -1
            for tex_id in range(self.model.ntex):
                skybox_textype = 2
                if self.model.tex_type[tex_id] == skybox_textype:
                    skybox_tex_id = tex_id
            assert skybox_tex_id >= 0
            return skybox_tex_id

        assert self._check_geom_for_texture(name)
        geom_id = self.model.geom_name2id(name)
        mat_id = self.model.geom_matid[geom_id]
        tex_id = self.model.mat_texid[mat_id]
        return tex_id

    def _name_to_mat_id(self, name):
        """
        Helper function to get material id from geom name.

        Args:
            name (str): name of geom

        Returns:
            int: id of material associated with geom

        Raises:
            ValueError: [No material associated with skybox]
            AssertionError: [No material associated with geom]
        """

        # 天空盒没有材质，单独处理
        if name == "skybox":
            raise ValueError("Error: skybox has no material!")

        assert self._check_geom_for_texture(name)
        geom_id = self.model.geom_name2id(name)
        mat_id = self.model.geom_matid[geom_id]
        return mat_id

    def _cache_checker_matrices(self):
        """
        为每个纹理缓存两个棋盘格矩阵：
            [[1, 0, 1, ...],
             [0, 1, 0, ...],
             ...]
        和
            [[0, 1, 0, ...],
             [1, 0, 1, ...],
             ...]
        用于快速生成棋盘格纹理图案。
        """
        self._texture_checker_mats = []
        for tex_id in range(self.model.ntex):
            texture = self.textures[tex_id]
            h, w = texture.bitmap.shape[:2]
            self._texture_checker_mats.append(self._make_checker_matrices(h, w))

    def _make_checker_matrices(self, h, w):
        """
        快速生成用于创建棋盘格图案的二维二进制矩阵。

        Args:
            h (int): 矩阵期望高度
            w (int): 矩阵期望宽度

        Returns:
            2 元组:
                - (np.array): 表示棋盘格第一部分的二维数组
                - (np.array): 表示棋盘格第二部分的二维数组
        """
        re = np.r_[((w + 1) // 2) * [0, 1]]
        ro = np.r_[((w + 1) // 2) * [1, 0]]
        cbd1 = np.expand_dims(np.row_stack(((h + 1) // 2) * [re, ro]), -1)[:h, :w]
        cbd2 = np.expand_dims(np.row_stack(((h + 1) // 2) * [ro, re]), -1)[:h, :w]
        return cbd1, cbd2


# MuJoCo 纹理类型枚举（来自 mjtTexture）
MJT_TEXTURE_ENUM = ["2d", "cube", "skybox"]


class Texture:
    """
    用于操作 MuJoCo 纹理的辅助类。

    Args:
        model (MjModel): MuJoCo 模型
        tex_id (int): 纹理在模型中的 id
    """

    __slots__ = ["id", "type", "height", "width", "tex_adr", "tex_rgb"]

    def __init__(self, model, tex_id):
        self.id = tex_id
        self.type = MJT_TEXTURE_ENUM[model.tex_type[tex_id]]
        self.height = model.tex_height[tex_id]
        self.width = model.tex_width[tex_id]
        self.tex_adr = model.tex_adr[tex_id]
        # tex_rgb 是模型中所有纹理 RGB 数据共享的一维数组
        self.tex_rgb = model.tex_rgb

    @property
    def bitmap(self):
        """
        从 MuJoCo 仿真中获取与该纹理关联的颜色位图。

        Returns:
            np.array: 表示 RGB 纹理位图的三维数组 (height, width, 3)
        """
        size = self.height * self.width * 3
        data = self.tex_rgb[self.tex_adr : self.tex_adr + size]
        return data.reshape((self.height, self.width, 3))


class DynamicsModder(BaseModder):
    """
    用于修改 MuJoCo 模型各种动力学属性的 Modder，例如摩擦、阻尼、质量、惯量等。

    可以同时修改存储在 MjModel 中的参数（如摩擦、阻尼）和存储在 PyMjOption 中的
    优化器参数（如介质密度 density、粘度 viscosity）。

    使用方法：
        sim = MjSim(...)
        modder = DynamicsModder(sim)
        modder.mod("element1_name", "attr1", new_value1)
        modder.mod("element2_name", "attr2", new_value2)
        ...
        modder.update()

    注意：所有修改完成后必须调用 modder.update()，以确保修改被传播到仿真中。

    注意：可通过 modder.dynamics_parameters 查看支持随机化的完整参数列表。

    注意：修改 MjModel.opt 中的参数（如 density、viscosity）时，不需要指定 name，
          应在 mod(...) 中将 name 设为 None，因为 opt 没有 name 属性。

    Args:
        sim (MjSim): Mujoco 仿真实例

        random_state (RandomState): np.random.RandomState 实例

        randomize_density (bool): 若为 True，随机化全局介质密度

        randomize_viscosity (bool): 若为 True，随机化全局介质粘度

        density_perturbation_ratio (float): 密度随机化的相对（比例）幅度

        viscosity_perturbation_ratio: 粘度随机化的相对（比例）幅度

        body_names (None 或 str 列表): 要参与随机化的 body 名称列表。若为 None，则所有 body 都会被随机化。

        randomize_position (bool): 若为 True，随机化 body 位置

        randomize_quaternion (bool): 若为 True，随机化 body 朝向四元数

        randomize_inertia (bool): 若为 True，随机化 body 惯量（仅对质量非零的 body 有效）

        randomize_mass (bool): 若为 True，随机化 body 质量（仅对质量非零的 body 有效）

        position_perturbation_size (float): body 位置随机化的扰动幅度

        quaternion_perturbation_size (float): body 朝向随机化的扰动幅度（弧度）

        inertia_perturbation_ratio (float): body 惯量随机化的相对（比例）幅度

        mass_perturbation_ratio (float): body 质量随机化的相对（比例）幅度

        geom_names (None 或 str 列表): 要参与随机化的 geom 名称列表。若为 None，则所有 geom 都会被随机化。

        randomize_friction (bool): 若为 True，随机化 geom 摩擦

        randomize_solref (bool): 若为 True，随机化 geom 的 solref 接触求解器参数

        randomize_solimp (bool): 若为 True，随机化 geom 的 solimp 接触求解器阻抗参数

        friction_perturbation_ratio (float): geom 摩擦随机化的相对（比例）幅度

        solref_perturbation_ratio (float): geom solref 随机化的相对（比例）幅度

        solimp_perturbation_ratio (float): geom solimp 随机化的相对（比例）幅度

        joint_names (None 或 str 列表): 要参与随机化的 joint 名称列表。若为 None，则所有 joint 都会被随机化。

        randomize_stiffness (bool): 若为 True，随机化关节刚度

        randomize_frictionloss (bool): 若为 True，随机化关节摩擦损耗

        randomize_damping (bool): 若为 True，随机化关节阻尼

        randomize_armature (bool): 若为 True，随机化关节电枢（armature）

        stiffness_perturbation_ratio (float): 关节刚度随机化的相对（比例）幅度

        frictionloss_perturbation_size (float): 关节摩擦损耗随机化的绝对扰动幅度

        damping_perturbation_size (float): 关节阻尼随机化的绝对扰动幅度

        armature_perturbation_size (float): 关节电枢随机化的绝对扰动幅度
    """

    def __init__(
        self,
        sim,
        random_state=None,
        # Opt parameters
        randomize_density=True,
        randomize_viscosity=True,
        density_perturbation_ratio=0.1,
        viscosity_perturbation_ratio=0.1,
        # Body parameters
        body_names=None,
        randomize_position=True,
        randomize_quaternion=True,
        randomize_inertia=True,
        randomize_mass=True,
        position_perturbation_size=0.02,
        quaternion_perturbation_size=0.02,
        inertia_perturbation_ratio=0.02,
        mass_perturbation_ratio=0.02,
        # Geom parameters
        geom_names=None,
        randomize_friction=True,
        randomize_solref=True,
        randomize_solimp=True,
        friction_perturbation_ratio=0.1,
        solref_perturbation_ratio=0.1,
        solimp_perturbation_ratio=0.1,
        # Joint parameters
        joint_names=None,
        randomize_stiffness=True,
        randomize_frictionloss=True,
        randomize_damping=True,
        randomize_armature=True,
        stiffness_perturbation_ratio=0.1,
        frictionloss_perturbation_size=0.05,
        damping_perturbation_size=0.01,
        armature_perturbation_size=0.01,
    ):
        super().__init__(sim=sim, random_state=random_state)

        # 初始化相关变量
        self.dummy_bodies = set()
        # 找出所有没有质量的"虚拟" body（如 world body），这些 body 不应被随机化
        for body_name in self.sim.model.body_names:
            body_id = self.sim.model.body_name2id(body_name)
            if self.sim.model.body_mass[body_id] == 0:
                self.dummy_bodies.add(body_name)

        # 获取需要随机化的元素名称列表
        self.body_names = list(self.sim.model.body_names) if body_names is None else body_names
        self.geom_names = list(self.sim.model.geom_names) if geom_names is None else geom_names
        self.joint_names = list(self.sim.model.joint_names) if joint_names is None else joint_names

        # 配置随机化设置
        # 每个动力学随机化组包含若干参数，每个参数有以下设置：
        #   - "randomize": 是否启用该参数的随机化
        #   - "perturbation": 扰动幅度（可能是相对比例或绝对值）
        #   - "type": "ratio"（相对扰动）或 "size"（绝对扰动）
        #   - "clip": 扰动后数值的裁剪范围 (low, high)
        self.opt_randomizations = {
            "density": {
                "randomize": randomize_density,
                "perturbation": density_perturbation_ratio,
                "type": "ratio",
                "clip": (0.0, np.inf),
            },
            "viscosity": {
                "randomize": randomize_viscosity,
                "perturbation": viscosity_perturbation_ratio,
                "type": "ratio",
                "clip": (0.0, np.inf),
            },
        }

        self.body_randomizations = {
            "position": {
                "randomize": randomize_position,
                "perturbation": position_perturbation_size,
                "type": "size",
                "clip": (-np.inf, np.inf),
            },
            "quaternion": {
                "randomize": randomize_quaternion,
                "perturbation": quaternion_perturbation_size,
                "type": "size",
                "clip": (-np.inf, np.inf),
            },
            "inertia": {
                "randomize": randomize_inertia,
                "perturbation": inertia_perturbation_ratio,
                "type": "ratio",
                "clip": (0.0, np.inf),
            },
            "mass": {
                "randomize": randomize_mass,
                "perturbation": mass_perturbation_ratio,
                "type": "ratio",
                "clip": (0.0, np.inf),
            },
        }

        self.geom_randomizations = {
            "friction": {
                "randomize": randomize_friction,
                "perturbation": friction_perturbation_ratio,
                "type": "ratio",
                "clip": (0.0, np.inf),
            },
            "solref": {
                "randomize": randomize_solref,
                "perturbation": solref_perturbation_ratio,
                "type": "ratio",
                "clip": (0.0, 1.0),
            },
            "solimp": {
                "randomize": randomize_solimp,
                "perturbation": solimp_perturbation_ratio,
                "type": "ratio",
                "clip": (0.0, np.inf),
            },
        }

        self.joint_randomizations = {
            "stiffness": {
                "randomize": randomize_stiffness,
                "perturbation": stiffness_perturbation_ratio,
                "type": "ratio",
                "clip": (0.0, np.inf),
            },
            "frictionloss": {
                "randomize": randomize_frictionloss,
                "perturbation": frictionloss_perturbation_size,
                "type": "size",
                "clip": (0.0, np.inf),
            },
            "damping": {
                "randomize": randomize_damping,
                "perturbation": damping_perturbation_size,
                "type": "size",
                "clip": (0.0, np.inf),
            },
            "armature": {
                "randomize": randomize_armature,
                "perturbation": armature_perturbation_size,
                "type": "size",
                "clip": (0.0, np.inf),
            },
        }

        # Store defaults so we don't loss track of the original (non-perturbed) values
        self.opt_defaults = None
        self.body_defaults = None
        self.geom_defaults = None
        self.joint_defaults = None
        self.save_defaults()

    def save_defaults(self):
        """
        Grabs the current values for all parameters in sim and stores them as default values
        """
        self.opt_defaults = {
            None: {  # no name associated with the opt parameters
                "density": self.sim.model.opt.density,
                "viscosity": self.sim.model.opt.viscosity,
            }
        }

        self.body_defaults = {}
        for body_name in self.sim.model.body_names:
            body_id = self.sim.model.body_name2id(body_name)
            self.body_defaults[body_name] = {
                "position": np.array(self.sim.model.body_pos[body_id]),
                "quaternion": np.array(self.sim.model.body_quat[body_id]),
                "inertia": np.array(self.sim.model.body_inertia[body_id]),
                "mass": self.sim.model.body_mass[body_id],
            }

        self.geom_defaults = {}
        for geom_name in self.sim.model.geom_names:
            geom_id = self.sim.model.geom_name2id(geom_name)
            self.geom_defaults[geom_name] = {
                "friction": np.array(self.sim.model.geom_friction[geom_id]),
                "solref": np.array(self.sim.model.geom_solref[geom_id]),
                "solimp": np.array(self.sim.model.geom_solimp[geom_id]),
            }

        self.joint_defaults = {}
        for joint_name in self.sim.model.joint_names:
            joint_id = self.sim.model.joint_name2id(joint_name)
            # 找到与该关节关联的所有自由度（dof）索引
            dof_idx = [i for i, v in enumerate(self.sim.model.dof_jntid) if v == joint_id]
            self.joint_defaults[joint_name] = {
                "stiffness": self.sim.model.jnt_stiffness[joint_id],
                "frictionloss": np.array(self.sim.model.dof_frictionloss[dof_idx]),
                "damping": np.array(self.sim.model.dof_damping[dof_idx]),
                "armature": np.array(self.sim.model.dof_armature[dof_idx]),
            }

    def restore_defaults(self):
        """
        恢复当前 Modder 中保存的所有默认值。
        """
        # 遍历所有默认值分组，逐个调用 mod() 恢复到仿真中
        for group_defaults in (self.opt_defaults, self.body_defaults, self.geom_defaults, self.joint_defaults):
            for name, defaults in group_defaults.items():
                for attr, default_val in defaults.items():
                    self.mod(name=name, attr=attr, val=default_val)

        # 确保修改传播到仿真中
        self.update()

    def randomize(self):
        """
        对仿真中所有启用的动力学参数进行随机化。
        """
        # 依次处理 opt、body、geom、joint 四组参数
        for group_defaults, group_randomizations, group_randomize_names in zip(
            (self.opt_defaults, self.body_defaults, self.geom_defaults, self.joint_defaults),
            (self.opt_randomizations, self.body_randomizations, self.geom_randomizations, self.joint_randomizations),
            ([None], self.body_names, self.geom_names, self.joint_names),
        ):
            for name in group_randomize_names:
                # 随机化与该元素关联的所有参数
                for attr, default_val in group_defaults[name].items():
                    val = copy.copy(default_val)
                    settings = group_randomizations[attr]
                    if settings["randomize"]:
                        # 根据参数类型生成随机扰动，并裁剪到合法范围
                        perturbation = np.random.rand() if type(val) in {int, float} else np.random.rand(*val.shape)
                        perturbation = settings["perturbation"] * (-1 + 2 * perturbation)
                        val = val + perturbation if settings["type"] == "size" else val * (1.0 + perturbation)
                        val = np.clip(val, *settings["clip"])
                    # 应用修改
                    self.mod(name=name, attr=attr, val=val)

        # 确保修改传播到仿真中
        self.update()

    def update_sim(self, sim):
        """
        除了调用父类方法更新 sim 引用外，还根据新的 sim 重新保存默认值。

        Args:
            sim (MjSim): 新的 MjSim 对象
        """
        super().update_sim(sim=sim)
        self.save_defaults()

    def update(self):
        """
        将到目前为止所做的修改传播到仿真中。
        """
        # 调用 MuJoCo forward 以重新计算依赖修改参数的内部状态
        self.sim.forward()

    def mod(self, name, attr, val):
        """
        通用方法：将元素 @name 的动力学参数 @attr 修改为新值 @val。

        Args:
            name (str): 要修改参数的元素名称。可以是 body、geom 或 joint 名称。
                若修改 opt 参数，应设为 None。
            attr (str): 要修改的动力学参数名称。有效选项见 self.dynamics_parameters。
            val (int 或 float 或 n 维数组): 要设置的新值。类型应与参数期望类型一致。
        """
        # 确保参数有效，然后调用对应的 mod_<attr> 方法
        assert (
            attr in self.dynamics_parameters
        ), "Invalid dynamics parameter specified! Supported parameters are: {};" " requested: {}".format(
            self.dynamics_parameters, attr
        )
        # 使用 getattr 动态调用对应的修改方法，避免大量 if-else
        getattr(self, f"mod_{attr}")(name, val)

    def mod_density(self, name=None, val=0.0):
        """
        修改仿真的全局介质密度。
        详见 http://www.mujoco.org/book/XMLreference.html#option。

        Args:
            name (str): 应设为 None（opt 没有 name 属性）。
            val (float): 新的密度值。
        """
        # 确保输入形式正确
        assert name is None, "No name should be specified if modding density!"

        # 修改密度值
        self.sim.model.opt.density = val

    def mod_viscosity(self, name=None, val=0.0):
        """
        修改仿真的全局介质粘度。
        详见 http://www.mujoco.org/book/XMLreference.html#option。

        Args:
            name (str): 应设为 None（opt 没有 name 属性）。
            val (float): 新的粘度值。
        """
        # 确保输入形式正确
        assert name is None, "No name should be specified if modding density!"

        # 修改粘度值
        self.sim.model.opt.viscosity = val

    def mod_position(self, name, val=(0, 0, 0)):
        """
        修改指定 body 的相对位置。
        详见 http://www.mujoco.org/book/XMLreference.html#body。

        Args:
            name (str): body 名称。
            val (3 数组): 新的相对位置 (x, y, z)。
        """
        # 直接修改模型中的 body 位置
        body_id = self.sim.model.body_name2id(name)
        self.sim.model.body_pos[body_id] = np.array(val)

    def mod_quaternion(self, name, val=(1, 0, 0, 0)):
        """
        修改指定 body 的相对朝向（四元数）。
        详见 http://www.mujoco.org/book/XMLreference.html#body。

        注意：本方法会自动对输入四元数进行归一化。

        Args:
            name (str): body 名称。
            val (4 数组): 新的相对四元数 (w, x, y, z)。
        """
        # 对输入四元数归一化
        val = np.array(val) / np.linalg.norm(val)
        # 修改 body 朝向
        body_id = self.sim.model.body_name2id(name)
        self.sim.model.body_quat[body_id] = val

    def mod_inertia(self, name, val):
        """
        修改指定 body 的相对惯量。
        详见 http://www.mujoco.org/book/XMLreference.html#body。

        Args:
            name (str): body 名称。
            val (3 数组): 惯量矩阵对角线的新值 (ixx, iyy, izz)。
        """
        # 跳过虚拟 body（质量为 0），避免破坏模型结构
        if name not in self.dummy_bodies:
            body_id = self.sim.model.body_name2id(name)
            self.sim.model.body_inertia[body_id] = np.array(val)

    def mod_mass(self, name, val):
        """
        修改指定 body 的质量。
        详见 http://www.mujoco.org/book/XMLreference.html#body。

        Args:
            name (str): body 名称。
            val (float): 新的质量值。
        """
        # 跳过虚拟 body，防止破坏模型基础结构
        if name not in self.dummy_bodies:
            body_id = self.sim.model.body_name2id(name)
            self.sim.model.body_mass[body_id] = val

    def mod_friction(self, name, val):
        """
        修改指定 geom 的摩擦参数。
        详见 http://www.mujoco.org/book/XMLreference.html#geom。

        Args:
            name (str): geom 名称。
            val (3 数组): 新的摩擦值 (sliding, torsional, rolling)。
        """
        geom_id = self.sim.model.geom_name2id(name)
        self.sim.model.geom_friction[geom_id] = np.array(val)

    def mod_solref(self, name, val):
        """
        修改指定 geom 的接触求解器参数 solref。
        详见 http://www.mujoco.org/book/modeling.html#CSolver。

        Args:
            name (str): geom 名称。
            val (2 数组): 新的 solref 值 (timeconst, dampratio)。
        """
        geom_id = self.sim.model.geom_name2id(name)
        self.sim.model.geom_solref[geom_id] = np.array(val)

    def mod_solimp(self, name, val):
        """
        修改指定 geom 的接触求解器阻抗参数 solimp。
        详见 http://www.mujoco.org/book/modeling.html#CSolver。

        Args:
            name (str): geom 名称。
            val (5 数组): 新的 solimp 值 (dmin, dmax, width, midpoint, power)。
        """
        geom_id = self.sim.model.geom_name2id(name)
        self.sim.model.geom_solimp[geom_id] = np.array(val)

    def mod_stiffness(self, name, val):
        """
        修改指定关节的刚度（stiffness）。
        详见 http://www.mujoco.org/book/XMLreference.html#joint。

        注意：若关节原始刚度为 0，则忽略该修改，因为非刚性关节（自由旋转）
              与刚性关节在物理本质上是不同的。

        Args:
            name (str): 关节名称。
            val (float): 新的刚度值。
        """
        # 仅当关节原本就有刚度时才修改
        jnt_id = self.sim.model.joint_name2id(name)
        if self.sim.model.jnt_stiffness[jnt_id] != 0:
            self.sim.model.jnt_stiffness[jnt_id] = val

    def mod_frictionloss(self, name, val):
        """
        修改指定关节的摩擦损耗（frictionloss）。
        详见 http://www.mujoco.org/book/XMLreference.html#joint。

        注意：若请求的是自由关节（free joint），则忽略该修改，因为自由关节的
              空气阻力/阻尼已由介质密度和粘度隐式描述。

        Args:
            name (str): 关节名称。
            val (float): 新的摩擦损耗值。
        """
        # 仅对非自由关节修改
        jnt_id = self.sim.model.joint_name2id(name)
        if self.sim.model.jnt_type[jnt_id] != 0:
            dof_idx = [i for i, v in enumerate(self.sim.model.dof_jntid) if v == jnt_id]
            self.sim.model.dof_frictionloss[dof_idx] = val

    def mod_damping(self, name, val):
        """
        修改指定关节的阻尼（damping）。
        详见 http://www.mujoco.org/book/XMLreference.html#joint。

        注意：若请求的是自由关节，则忽略该修改，原因同上。

        Args:
            name (str): 关节名称。
            val (float): 新的阻尼值。
        """
        # 仅对非自由关节修改
        jnt_id = self.sim.model.joint_name2id(name)
        if self.sim.model.jnt_type[jnt_id] != 0:
            dof_idx = [i for i, v in enumerate(self.sim.model.dof_jntid) if v == jnt_id]
            self.sim.model.dof_damping[dof_idx] = val

    def mod_armature(self, name, val):
        """
        修改指定关节的电枢（armature）。
        详见 http://www.mujoco.org/book/XMLreference.html#joint。

        注意：若请求的是自由关节，则忽略该修改。

        Args:
            name (str): 关节名称。
            val (float): 新的电枢值。
        """
        # 仅对非自由关节修改
        jnt_id = self.sim.model.joint_name2id(name)
        if self.sim.model.jnt_type[jnt_id] != 0:
            dof_idx = [i for i, v in enumerate(self.sim.model.dof_jntid) if v == jnt_id]
            self.sim.model.dof_armature[dof_idx] = val

    @property
    def dynamics_parameters(self):
        """
        Returns:
            set: 本 Modder 支持随机化的所有动力学参数名称。
        """
        return {
            # Opt（优化器/全局）参数
            "density",
            "viscosity",
            # Body 参数
            "position",
            "quaternion",
            "inertia",
            "mass",
            # Geom 参数
            "friction",
            "solref",
            "solimp",
            # Joint 参数
            "stiffness",
            "frictionloss",
            "damping",
            "armature",
        }

    @property
    def opt(self):
        """
        Returns:
             PyMjOption: MjModel 的仿真选项（opt）对象
        """
        return self.sim.model.opt
