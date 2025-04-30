import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .formation_env import FormationEnv

class MARLFormationEnv(gym.Env):
    def __init__(self, config=None):
        super().__init__()
        # Underlying PyBullet env with all 4 drones
        self.env = FormationEnv(gui=False)
        # Agent IDs
        self.agents = list(range(self.env.NUM_DRONES))
        # Number of agents
        self.num_agents = len(self.agents)

        # Vectorized obs/action spaces from FormationEnv
        base_obs = self.env.observation_space   # shape (4,132)
        base_act = self.env.action_space        # shape (4,1)

        # Per‐agent spaces: drop the first dimension
        obs_low, obs_high = base_obs.low[0], base_obs.high[0]
        act_low, act_high = base_act.low[0], base_act.high[0]

        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high,
            shape=base_obs.shape[1:], dtype=base_obs.dtype
        )
        self.action_space = spaces.Box(
            low=act_low, high=act_high,
            shape=base_act.shape[1:], dtype=base_act.dtype
        )

        # Expose dicts for RLlib
        self.observation_spaces = {i: self.observation_space for i in self.agents}
        self.action_spaces      = {i: self.action_space      for i in self.agents}

    def reset(self, seed=None, options=None):
        vec_obs, _ = self.env.reset()
        obs_dict = {i: vec_obs[i] for i in self.agents}
        return obs_dict, {}

    def step(self, action_dict):
        # Build (4,1) array of actions
        acts = np.stack([action_dict[i] for i in self.agents], axis=0)
        vec_obs, reward, terminated, truncated, info = self.env.step(acts)
        done = terminated or truncated

        obs_dict   = {i: vec_obs[i]    for i in self.agents}
        rew_dict   = {i: reward        for i in self.agents}
        term_dict  = {i: terminated    for i in self.agents}
        trunc_dict = {i: truncated     for i in self.agents}
        info_dict  = {i: {}            for i in self.agents}

        # RLlib requires “__all__” flags
        term_dict["__all__"]  = done
        trunc_dict["__all__"] = done

        return obs_dict, rew_dict, term_dict, trunc_dict, info_dict

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()
