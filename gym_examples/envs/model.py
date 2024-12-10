import numpy as np

from gym import utils
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box


class ModelEnv(MujocoEnv, utils.EzPickle):
    """
    ### Observation space
    0: rotation x
    1: rotation y
    2: x pos Ball
    3: y pos Ball
    4: z pos Ball
    5: 
    6: hinge?
    7: hinge?
    8: 
    9:
    10: 
    11: x vel ball
    12: y vel ball
    13: z vel ball
    14: 
    15: 
    16: 

    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64) #shape=(17,)
        MujocoEnv.__init__(
            self,
            "model.xml", #inverted_pendulum
            2,
            observation_space=observation_space,
            **kwargs
        )

    def step(self, a):
        # reward = 1.0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        terminated = bool(not np.isfinite(ob).all() or (ob[4] < -0.4) or (abs(ob[3]) > 0.6) or (abs(ob[2]) > 0.6))
        reward = -1 if terminated else 0.1 
        if self.render_mode == "human":
            self.render()
        extractor=[0,1,2,3,4,11,12,13]
        ob = ob[extractor]
        return ob, reward, terminated, False, {}

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        ob = self._get_obs()
        extractor=[0,1,2,3,4,11,12,13]
        ob = ob[extractor]
        return ob

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()

    def viewer_setup(self):
        assert self.viewer is not None
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent