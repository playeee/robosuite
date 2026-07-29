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
        # SO101 是单指夹爪（1-DOF hinge），jaw 内侧各有一个高摩擦 fingerpad
        # 薄板（fixed_jaw_pad / moving_jaw_pad），用于物理夹持和接触检测。
        #
        # 原始 mesh 碰撞体（fixed_jaw_collision / moving_jaw_collision）已设
        # contype=0 不参与碰撞检测，避免大面积面接触把 cube 推开。
        #
        # _check_grasp 要求 left_fingerpad 和 right_fingerpad 各自至少有一个
        # geom 与物体接触才判定为"抓取"。SO101 的两侧 pad 在夹爪闭合时能
        # 同时接触 cube，因此可以用各自的 pad 作为 fingerpad。
        # 同时保留对方的 pad 作为备选，确保只要任一侧接触即可通过判定。
        return {
            "left_finger": ["moving_jaw_pad", "moving_jaw_collision"],
            "right_finger": ["fixed_jaw_pad", "fixed_jaw_collision"],
            # _check_grasp 要求 left_fingerpad 和 right_fingerpad 两组各自至少
            # 有一个 geom 与 cube 接触才判 True。SO101 是 hinge 单指夹爪，必须
            # 活动颚 pad 和固定颚 pad 同时接触 cube 才算"夹住"。若两组都设为
            # 同一列表，则任一 pad 接触即判 True，会误判（如 pad 穿过 cube 时
            # 单侧接触也算抓取）。这里分别只取一个 pad，确保两侧同时接触。
            "left_fingerpad": ["moving_jaw_pad"],
            "right_fingerpad": ["fixed_jaw_pad"],
        }
