import os
import sys
import datetime

import jinja2
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


def rotate(x, θ):
    R = np.array([[np.cos(θ), -np.sin(θ)],
                  [np.sin(θ),  np.cos(θ)]])
    res = np.zeros(3)
    res[:2] = R @ x[:2]
    res[-1] = x[-1] + θ
    return res


def to_frame(new_frame, x, translate):
    x3 = np.zeros(3)
    x3[:len(x)] = x
    θ = new_frame[-1]
    if translate:
        x3[:2] -= new_frame[:2]
    res = rotate(x3, -θ)
    return res[:len(x)]


class SimplePusherEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, rotation_x=None, n_sim_steps=16):
        np.random.seed(os.getpid() + datetime.datetime.now().microsecond)
        self.n_steps = 0
        self.n_steps_max = int(1600 / n_sim_steps)
        if rotation_x is None:
            self.rotation_x = 0.05 * (2 * np.random.rand() - 1)
        else:
            self.rotation_x = rotation_x
        print(f'INFO: Creating SimplePusher environment with rotation_x: {self.rotation_x}', file=sys.stderr)
        utils.EzPickle.__init__(self)
        base_path = os.path.abspath(os.path.dirname(__file__))
        template_path = os.path.join(base_path, 'assets', 'simple_pusher_template.xml')
        with open(template_path, 'r') as f:
            template = jinja2.Template(f.read())
        xml_path = os.path.join(base_path, 'assets', f'simple_pusher_{os.getpid()}.xml')
        with open(xml_path, 'w') as f:
            f.write(template.render(rotation_x=self.rotation_x) + '\n')
        mujoco_env.MujocoEnv.__init__(self, xml_path, n_sim_steps)
        os.remove(xml_path)

    def step(self, a):
        if self.n_steps < self.n_steps_max - 1:
            done = False
        else:
            done = True
        self.n_steps += 1
        obs = self._get_obs()
        goal_in_obj_frame = obs[-3:]
        action_in_world_frame = to_frame(goal_in_obj_frame, a, translate=False)
        self.do_simulation(action_in_world_frame, self.frame_skip)
        obs_ = self._get_obs()
        reward = self._reward(obs, obs_)
        return obs_, reward, done, {}

    def _reward(self, obs, obs_):
        distance_weights = np.array([1, 1, 0.1])
        def rew(obs):
            eef = obs[:2]
            goal = obs[4:]
            obj = np.array([0.0, 0.0, 0.0])
            eef_obj_rew = -np.linalg.norm(eef - obj[:2])
            obj_goal_rew = -np.linalg.norm((obj - goal) * distance_weights)
            return 10 * eef_obj_rew + 100 * obj_goal_rew

        goal = obs_[4:]
        goal_reward = 10 * np.exp(-32 * np.linalg.norm(goal * distance_weights))

        reward = rew(obs)
        reward_ = rew(obs_)

        return (reward_ - reward) + goal_reward

    def viewer_setup(self):
        self.viewer.cam.distance = 1.2
        self.viewer.cam.elevation = -80
        self.viewer.cam.azimuth = 90

    def reset_model(self):
        self.n_steps = 0

        qpos = self.init_qpos

        # eef
        qpos[0] = np.random.randn() * 0.05
        qpos[1] = -0.25

        # object
        qpos[2] = np.random.randn() * 0.05
        qpos[3] = np.random.randn() * 0.01 - 0.1
        qpos[4] = np.random.randn() * 0.4

        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        qpos = self.sim.data.qpos.flat
        qvel = self.sim.data.qvel.flat

        # in world/goal frame
        goal = np.array([0.0, 0.0, 0.0])
        eef = qpos[:2]
        eef_vel = qvel[:2]
        obj = qpos[2:5]

        return np.concatenate([
            to_frame(obj, eef, translate=True),
            to_frame(obj, eef_vel, translate=False),
            to_frame(obj, goal, translate=True),
        ])

if __name__ == '__main__':
    env = SimplePusherEnv(rotation_x=-0.04)
    while True:
        obs = env.reset()
        for _ in range(200):
            eef = obs[:2]
            u = -eef[:2]
            u *= 0.1
            obs, r, _, _ = env.step(u)
            env.render()
