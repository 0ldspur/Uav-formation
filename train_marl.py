from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from environment.formation_env import FormationEnv

# Create training and evaluation environments
train_env = Monitor(FormationEnv(gui=False))
eval_env  = Monitor(FormationEnv(gui=False))

# Periodically evaluate and save best model
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_model/",
    log_path="./ppo_marl_tensorboard/",
    eval_freq=5000,
    deterministic=True,
    render=False
)

# Instantiate PPO
model = PPO(
    "MlpPolicy",
    train_env,
    verbose=1,
    tensorboard_log="./ppo_marl_tensorboard/",
    device="auto"
)

# Train for 200 000 timesteps
model.learn(
    total_timesteps=200_000,
    callback=eval_callback
)
model.save("ppo_marl_formation_v2")
