import time
from environment.formation_env import FormationEnv

env = FormationEnv(gui=True)
obs, _ = env.reset()
print("Press Ctrl+C to exit")

try:
    while True:
        actions, _ = env.action_space.sample(), None
        obs, _, _, _, _ = env.step(actions)
        time.sleep(1/240)
except KeyboardInterrupt:
    pass
finally:
    env.close()
