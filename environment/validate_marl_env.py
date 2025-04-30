import gymnasium as gym
from marl_env import MARLFormationEnv

# Register and make if you like, or instantiate directly
env = MARLFormationEnv()
obs, _ = env.reset()
print("Sample obs:", {i: v.shape for i, v in obs.items()})
actions = {i: env.action_space.sample() for i in obs}
obs2, rewards, dones, infos = env.step(actions)
print("Next obs:", {i: v.shape for i, v in obs2.items()})
print("Rewards:", rewards)
print("Dones:", dones)
env.close()
