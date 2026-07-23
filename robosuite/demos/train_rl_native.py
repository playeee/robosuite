import numpy as np
import robosuite as suite

# 1. 创建基础环境
env = suite.make(
    "Lift",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=20,
    horizon=200,
    reward_shaping=True,
)

# 2. 查看动作空间和观测结构
print("Action spec:", env.action_spec)          # (low, high)
print("Action dim:", env.action_spec[0].shape)  # e.g. (7,) for OSC_POSE

# 3. 原生 RL 循环
for episode in range(10):
    obs = env.reset()  # obs 是 OrderedDict
    episode_reward = 0

    for t in range(env.horizon):
        # 随机动作（实际训练时换成你的策略网络）
        action = np.random.uniform(
            env.action_spec[0], env.action_spec[1]
        )

        # step 返回 4 元组
        obs, reward, done, info = env.step(action)
        episode_reward += reward

        if done:
            break

    print(f"Episode {episode}, reward: {episode_reward:.2f}")