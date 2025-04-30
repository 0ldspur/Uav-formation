# train_mappo.py

import gymnasium as gym
from ray.tune.registry import register_env
from environment.marl_env import MARLFormationEnv
from ray.rllib.algorithms.ppo import PPOConfig

# 1) Register the env
register_env("marl_form", lambda cfg: MARLFormationEnv(cfg))

# 2) Instantiate one shared policy per agent
env_template = MARLFormationEnv()
num_agents = len(env_template.agents)
policies = {
    f"agent_{i}": (
        None,
        env_template.observation_space,
        env_template.action_space,
        {}
    ) for i in range(num_agents)
}

# 3) Build multi-agent PPO (MAPPO) config
config = (
    PPOConfig()
    .environment(env="marl_form")
    .framework("torch")
    .env_runners(
        num_env_runners=2,
        num_envs_per_env_runner=1
    )
    .multi_agent(
        policies=policies,
        policy_mapping_fn=lambda agent_id, **_: f"agent_{agent_id}"
    )
    .training(
        lr=5e-4,
        gamma=0.99,
        entropy_coeff=0.01,
        train_batch_size=4000,
        minibatch_size=256,
        num_sgd_iter=10
    )
    .resources(num_gpus=0)
)

# 4) Train
if __name__ == "__main__":
    trainer = config.build()
    for i in range(200):
        result = trainer.train()
        if i % 20 == 0:
            print(f"Iter {i} | reward_mean={result['episode_reward_mean']:.2f}")
            ckpt = trainer.save()
            print("Saved checkpoint:", ckpt)
