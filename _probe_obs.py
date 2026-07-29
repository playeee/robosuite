"""探测环境观测空间维度，弄清楚 41 维观测的含义"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "robosuite", "demos"))

import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.wrappers.domain_randomization_wrapper import DomainRandomizationWrapper
from robosuite.utils.placement_samplers import UniformRandomSampler
from so101_realistic import SO101LiftObservationWrapper, SO101LiftRewardShapingWrapper

placement_initializer = UniformRandomSampler(
    name="SO101ObjectSampler",
    x_range=[-0.28, -0.12],
    y_range=[-0.08, 0.08],
    rotation=None,
    ensure_object_boundary_in_range=False,
    ensure_valid_placement=True,
    reference_pos=(0, 0, 0.8),
    z_offset=0.01,
)

env = suite.make(
    "Lift",
    robots="SO101",
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=20,
    horizon=200,
    reward_shaping=False,
    reward_scale=1.0,
    use_object_obs=False,
    initialization_noise=None,
    table_friction=(1.0, 5e-3, 1e-4),
    table_full_size=(0.8, 0.8, 0.05),
    placement_initializer=placement_initializer,
)
# robosuite env doesn't have observation_space directly
print(f"Base env type: {type(env)}")

env = GymWrapper(env)
print(f"GymWrapper obs space: {env.observation_space.shape}")
obs_before_wrapper, info = env.reset()
print(f"Obs before ObsWrapper shape: {obs_before_wrapper.shape}")

# Now add ObservationWrapper
env2 = SO101LiftObservationWrapper(env)
obs_after_wrapper, info = env2.reset()
print(f"Obs after ObsWrapper shape: {obs_after_wrapper.shape}")

# The difference is the 3-dim rel_pos (cube - eef)
print(f"\nBefore wrapper obs: {obs_before_wrapper}")
print(f"\nAfter wrapper obs: {obs_after_wrapper}")
print(f"\nLast 3 dims (rel_pos): {obs_after_wrapper[-3:]}")

# 获取底层环境的观测键名
base_env = env2
while hasattr(base_env, 'env'):
    base_env = base_env.env
print(f"\nBase env type: {type(base_env)}")

# 获取 EEF 和 cube 位置
eef_site_id = base_env.robots[0].eef_site_id["right"]
eef_pos = np.array(base_env.sim.data.site_xpos[eef_site_id])
cube_pos = np.array(base_env.sim.data.body_xpos[base_env.cube_body_id])
print(f"\nEEF position: {eef_pos}")
print(f"Cube position: {cube_pos}")
print(f"EEF-Cube rel (cube-eef): {cube_pos - eef_pos}")
print(f"Last 3 obs dims: {obs_after_wrapper[-3:]}")
print(f"Match: {np.allclose(cube_pos - eef_pos, obs_after_wrapper[-3:], atol=0.01)}")

# 获取 robosuite 环境的观测键
if hasattr(base_env, 'observation_names'):
    print(f"\nObservation names: {base_env.observation_names}")
    
# 通过 _setup_observables 获取 obs 名称
try:
    obs_dict = base_env._get_observation()
    print(f"\nObs dict keys: {list(obs_dict.keys()) if isinstance(obs_dict, dict) else 'not a dict'}")
except:
    pass

# 找 GymWrapper 中的 obs 键名
gym_env = env  # env is GymWrapper
if hasattr(gym_env, '_obs_keys'):
    print(f"\nGymWrapper obs keys: {gym_env._obs_keys}")
else:
    # search
    for attr in dir(gym_env):
        if 'key' in attr.lower() or 'obs' in attr.lower():
            val = getattr(gym_env, attr)
            if isinstance(val, (list, tuple)) and len(val) > 0 and isinstance(val[0], str):
                print(f"  {attr}: {val}")

# Also try to look at the GymWrapper source
import inspect
src = inspect.getsource(GymWrapper.__init__)
print(f"\nGymWrapper init source (first 500 chars):\n{src[:500]}")
