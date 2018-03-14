import os
import datetime

import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env


def body_index(model, body_name):
    return model.body_names.index(body_name)


def body_pos(model, body_name):
    ind = body_index(model, body_name)
    return model.body_pos[ind]


class YumiReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.high = np.array([40, 35, 30, 20, 15, 10, 10])
        self.low = -self.high
        self.wt = 0.0
        self.we = 0.0
        root_dir = os.path.dirname(__file__)
        xml_path = os.path.join(root_dir, 'yumi', 'yumi.xml')
        mujoco_env.MujocoEnv.__init__(self, xml_path, 1)
        utils.EzPickle.__init__(self)

        # Manually define this to let a be in [-1, 1]^d
        self.action_space = spaces.Box(low=-np.ones(7) * 2, high=np.ones(7) * 2, dtype=np.float32)
        self.set_task_params(0.9, 0.0, 0.0, 0.2)

    def set_task_params(self, wt, x, y, z):
        self.wt = wt
        self.we = 1 - wt
        qpos = self.init_qpos
        qpos[-3:] = [x, y, z]
        qvel = self.init_qvel
        self.set_state(qpos, qvel)

    def step(self, a):
        a_real = a * self.high / 2
        self.do_simulation(a_real, self.frame_skip)
        reward = self._reward(a_real)
        done = False
        ob = self._get_obs()
        return ob, reward, done, {}

    def _reward(self, a):
        eef = self.get_body_com('gripper_r_base')
        goal = self.get_body_com('goal')
        goal_distance = np.linalg.norm(eef - goal)
        q_norm = np.linalg.norm(self.sim.data.qpos.flat[:7]) / 7
        if goal_distance > 0.025:
            reward = -(
                0.3 * self.wt * np.linalg.norm(eef - goal) * 2.0 +
                0.3 * self.we * np.linalg.norm(a) / 40 +
                q_norm
            )
        else:
            reward = 0.0
        return reward

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[:7],
            np.clip(self.sim.data.qvel.flat[:7], -10, 10)
        ])

    def reset_model(self):
        low  = np.array([-1.0,-0.3,-0.4,-0.4])
        high = np.array([ 0.4, 0.6, 0.4, 0.4])
        #self.init_qpos[:4] = np.random.uniform(low, high)
        self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = 2.0
        self.viewer.cam.elevation = -30
        self.viewer.cam.azimuth = 180
