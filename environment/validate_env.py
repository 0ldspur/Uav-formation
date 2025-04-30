import numpy as np
from environment.formation_env import FormationEnv

env = FormationEnv(gui=False)
obs, _ = env.reset()
print("Obs shape:", obs.shape)        # expect (4,16)
print("Act shape:", env.action_space.sample().shape)  # e.g. (4,)
env.close()
