"""
SO101 真实感 Lift 训练脚本的辅助工具包。

包含：
    - 奖励塑形包装器
    - SB3 训练回调（日志、rollout 收集）
    - Rollout 轨迹分析工具
"""

from .callbacks import RewardBreakdownCallback, RolloutCollectorCallback, TrainingLoggerCallback
from .rollout_utils import analyze_rollouts
from .wrappers import (
    REWARD_COMPONENTS,
    REWARD_PENALTY_COMPONENTS,
    REWARD_PROGRESS_COMPONENTS,
    SO101LiftObservationWrapper,
    SO101LiftRewardShapingWrapper,
)

__all__ = [
    "SO101LiftObservationWrapper",
    "SO101LiftRewardShapingWrapper",
    "TrainingLoggerCallback",
    "RolloutCollectorCallback",
    "RewardBreakdownCallback",
    "analyze_rollouts",
    "REWARD_COMPONENTS",
    "REWARD_PROGRESS_COMPONENTS",
    "REWARD_PENALTY_COMPONENTS",
]
