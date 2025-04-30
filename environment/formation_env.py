import numpy as np
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class FormationEnv(MultiHoverAviary):
    def __init__(self, gui=False):
        initial_positions = np.array([[0,0,1],[1,0,1],[1,1,1],[0,1,1]])
        super().__init__(
            num_drones=4,
            drone_model=DroneModel.CF2X,
            initial_xyzs=initial_positions,
            physics=Physics.PYB,
            pyb_freq=240,
            ctrl_freq=240,
            obs=ObservationType.KIN,
            act=ActionType.ONE_D_RPM,
            gui=gui
        )
        self.target_positions = initial_positions

    def _computeReward(self):
        positions = np.array([self._getDroneStateVector(i)[0:3] for i in range(4)])
        dists = np.linalg.norm(positions - self.target_positions, axis=1)
        return -np.sum(dists)  # Negative sum of distances
