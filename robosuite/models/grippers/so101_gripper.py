import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class SO101Gripper(GripperModel):
    """
    SO101 单指夹爪的 robosuite 包装。

    夹爪的几何体、关节、驱动器定义在 grippers/so101_gripper.xml 中；
    通过覆盖 naming_prefix 与机器人本体保持一致（robot0_），从而能正确引用
    机器人本体 asset 中已加载的 mesh/material。

    Args:
        idn (int or str): 该夹爪实例的唯一标识，robosuite 传入形式为 "{robot_idn}_{arm}"
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/so101_gripper.xml"), idn=idn)
        self.current_action = np.zeros(self.dof)

    @property
    def naming_prefix(self):
        # 与机器人本体使用相同 prefix（如 "robot0_"），共享 robot.xml 中的 mesh/material
        return f"robot{str(self.idn).split('_')[0]}_"

    def format_action(self, action):
        # [-1, 1] -> [-1, 1]，直接透传给 SimpleGripController
        return action

    @property
    def init_qpos(self):
        # 初始半开
        return np.array([0.5])

    @property
    def _important_geoms(self):
        return {
            "left_finger": ["moving_jaw_so101_v1_collision"],
            "right_finger": ["wrist_roll_follower_so101_v1_collision"],
            "left_fingerpad": ["moving_jaw_so101_v1_collision"],
            "right_fingerpad": ["wrist_roll_follower_so101_v1_collision"],
        }
