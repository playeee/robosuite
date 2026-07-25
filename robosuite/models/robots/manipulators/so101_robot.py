import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class SO101(ManipulatorModel):
    """
    SO101 是 SO-ARM100 项目中的 5 自由度桌面级机械臂 + 单指夹爪。

    本项目将其 MJCF 模型（so101_new_calib.xml）集成进 robosuite，
    用于在 Lift 等操作任务中进行接近真实硬件的仿真训练。

    Args:
        idn (int or str): 该机器人实例的唯一标识
    """

    arms = ["right"]

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/so101/robot.xml"), idn=idn)

        # SO101 使用 STS3215 舵机，默认阻尼已在外部 XML 中设置；
        # 这里微调 arm joints 阻尼，使低维 RL 策略更稳定。
        self.set_joint_attribute(attrib="damping", values=np.array([0.6, 0.6, 0.6, 0.6, 0.6]))

    @property
    def _eef_name(self):
        return {"right": "gripper"}

    @property
    def default_base(self):
        # SO101 自带底座，不需要额外的 RethinkMount
        return "NullBase"

    @property
    def default_gripper(self):
        # 夹爪本体在 robot.xml 中，gripper model 只补充 sites/sensors
        return {"right": "SO101Gripper"}

    @property
    def default_controller_config(self):
        return {"right": "default_so101"}

    @property
    def init_qpos(self):
        # 5 个 arm 关节的初始姿态。
        # 经测试，全零姿态会让末端执行器悬浮在桌面上方约 0.2m（z≈1.03），
        # 策略难以在稀疏奖励下学会“下降-接近-抓取”。
        # 这里采用一个“预备抓取”姿态：肩部俯仰 0.7 使大臂前倾、肘部微屈，
        # 让末端落在桌面附近（z≈0.87），更接近 cube 工作空间。
        return np.array([0.0, 0.7, -0.2, -0.2, -0.5])

    @property
    def base_xpos_offset(self):
        # SO101 是桌面级机械臂，直接放置在桌面上（z=0.8）而非像 Panda 那样站在地面
        return {
            "bins": (-0.5, -0.1, 0.8),
            "empty": (-0.6, 0, 0.8),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0.8),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 0.35))

    @property
    def _horizontal_radius(self):
        # 桌面工作空间半径约 0.35m
        return 0.35

    @property
    def arm_type(self):
        return "single"
