import numpy as np
import matplotlib.pyplot as plt
from environment.formation_env import FormationEnv
from ray.rllib.algorithms.ppo import PPO

# Load trained policy
trainer = PPO.from_checkpoint("checkpoints/")
policy = trainer.get_policy("agent_0")  # All agents share this policy

# Visualize in PyBullet GUI
env = FormationEnv(gui=True)
obs, _ = env.reset()
for _ in range(500):
    actions = {i: policy.compute_single_action(obs[i])[0] for i in range(4)}
    obs, _, _, _, _ = env.step([actions[i] for i in range(4)])
env.close()

# Plot trajectories
obs, _ = env.reset()
traj = {i: [] for i in range(4)}
for _ in range(500):
    actions = {i: policy.compute_single_action(obs[i])[0] for i in range(4)}
    obs, _, done, _, _ = env.step([actions[i] for i in range(4)])
    for i in range(4):
        traj[i].append(obs[i][0:3])  # x,y,z
    if done: break

plt.figure(figsize=(6,6))
for i in range(4):
    arr = np.array(traj[i])
    plt.plot(arr[:,0], arr[:,1], label=f"Drone {i}")
plt.xlabel("X Position"); plt.ylabel("Y Position")
plt.legend(); plt.grid(); plt.show()
