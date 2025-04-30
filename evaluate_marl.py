import numpy as np
import matplotlib.pyplot as plt
from environment.formation_env import FormationEnv
from stable_baselines3 import PPO

env = FormationEnv(gui=True)
model = PPO.load("ppo_marl_formation_v1")  # Or your latest model

obs, _ = env.reset()
traj = {i: [] for i in range(env.NUM_DRONES)}

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _, _ = env.step(action)
    for i in range(env.NUM_DRONES):
        traj[i].append(obs[i][:3])  # x, y, z
    if done:
        break
env.close()

plt.figure(figsize=(6,6))
for i, path in traj.items():
    arr = np.array(path)
    plt.plot(arr[:,0], arr[:,1], label=f"Drone {i}")
plt.title("Drone XY Trajectories")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.axis("equal")
plt.legend()
plt.grid(True)
plt.show()
