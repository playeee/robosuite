"""确定观测向量中各分量的确切索引位置"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "robosuite", "demos"))

import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.utils.placement_samplers import UniformRandomSampler
from so101_realistic import SO101LiftObservationWrapper

placement_initializer = UniformRandomSampler(
    name="SO101ObjectSampler",
    x_range=[-0.28, -0.12], y_range=[-0.08, 0.08],
    rotation=None, ensure_object_boundary_in_range=False,
    ensure_valid_placement=True, reference_pos=(0, 0, 0.8), z_offset=0.01,
)

env = suite.make("Lift", robots="SO101", has_renderer=False,
    has_offscreen_renderer=False, use_camera_obs=False,
    control_freq=20, horizon=200, reward_shaping=False, reward_scale=1.0,
    use_object_obs=False, initialization_noise=None,
    table_friction=(1.0, 5e-3, 1e-4), table_full_size=(0.8, 0.8, 0.05),
    placement_initializer=placement_initializer)

# Get obs dict from robosuite env directly
obs_dict = env.reset()
print("Robosuite obs dict keys and shapes:")
for k, v in obs_dict.items():
    if hasattr(v, 'shape'):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    else:
        print(f"  {k}: type={type(v)}")

# Now let's look at the "robot0_proprio-state" which is what GymWrapper uses
proprio = obs_dict["robot0_proprio-state"]
print(f"\nrobot0_proprio-state: shape={proprio.shape}")
print(f"  values: {proprio}")

# Get EEF position from simulation
eef_site_id = env.robots[0].eef_site_id["right"]
eef_pos = np.array(env.sim.data.site_xpos[eef_site_id])
print(f"\nEEF pos from sim: {eef_pos}")

# Get gripper qpos
gripper_idx = env.robots[0]._ref_gripper_joint_pos_indexes["right"]
gripper_qpos = np.array([env.sim.data.qpos[x] for x in gripper_idx])
print(f"Gripper qpos: {gripper_qpos}")

# Let's look at all the observation modality names
if hasattr(env, '_obs_modality_map'):
    print(f"\nObs modality map: {env._obs_modality_map}")

# Try to find observation name to index mapping
# The GymWrapper concatenates obs_dict values for keys in self.keys
# Let's trace through the order
env_gym = GymWrapper(env)
print(f"\nGymWrapper keys: {env_gym.keys}")

# For realistic_state mode, it's just ["robot0_proprio-state"]
# The proprio-state is constructed in the robot's observation method
# Let's look at what's in it
robot = env.robots[0]
print(f"\nRobot type: {type(robot)}")
if hasattr(robot, 'observation_names'):
    print(f"Robot observation_names: {robot.observation_names}")

# Let's manually trace the observation construction
# The robot creates observations through _setup_observables
# Let's get all the observable sensors
for obs_name in sorted(env.observation_names):
    val = obs_dict.get(obs_name)
    if val is not None:
        print(f"  {obs_name}: shape={np.array(val).shape}, first_vals={np.array(val).flat[:5]}")
