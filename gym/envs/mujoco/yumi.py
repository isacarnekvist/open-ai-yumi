import os

import cvxopt
import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env


cvxopt.solvers.options['show_progress'] = False


class YumiReacherEnv():
    def __init__(self):
        raise NotImplementedError


class YumiPusherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.high = np.array([40, 35, 30, 20, 15, 10, 10])
        self.low = -self.high
        self.wt = 0.0
        self.we = 0.0
        self.controller = YumiController(self)
        root_dir = os.path.dirname(__file__)
        xml_path = os.path.join(root_dir, 'yumi', 'yumi_pusher.xml')

        mujoco_env.MujocoEnv.__init__(self, xml_path, 1)
        utils.EzPickle.__init__(self)

        # Manually define this to let a be in [-2.0, 2.0]^d
        self.action_space = spaces.Box(low=-np.ones(2) * 2.0, high=np.ones(2) * 2.0, dtype=np.float32)


    def seed(self, seed=0):
        np.random.seed(seed)

    def step(self, a):
        if self.action_space is not None:
            a_clipped = np.clip(a, self.action_space.low, self.action_space.high)
        else:
            a_clipped = a
        a_normed = 0.07 * a_clipped / 2.0
        u = self.controller(a_normed[0] + 0.04, a_normed[1] - 0.10)  # action is where to place end effector in goal frame
        self.do_simulation(u, self.frame_skip)

        done = False
        obs = self._get_obs()
        reward = self._reward(obs)
        return obs, reward, done, {}

    def _reward(self, obs):
        eef = obs[:2]
        obj = obs[4:6]
        eef_obj_distance = np.linalg.norm(obj - eef)
        goal_distance = np.linalg.norm(obj)

        reward = -(
            0.1 * eef_obj_distance +
            0.9 * goal_distance
        )
        return reward

    def _get_obs(self):
        eef_x, eef_y, _ = self.body_pos('gripper_r_base')
        eef_x_dot, eef_y_dot, _ = self.body_vel('gripper_r_base')
        obj_x, obj_y, obj_θ = self.sim.data.qpos[-3:]
        θ_quadrant = (obj_θ % (np.pi / 2)) * 4
        obs = np.array([eef_x - 0.04, eef_y + 0.1,                  # eef position (in goal frame)
                        eef_x_dot, eef_y_dot,                       # eef velocities
                        obj_x, obj_y,                               # obj position (in goal frame)
                        np.cos(θ_quadrant), np.sin(θ_quadrant)])    # transformation that ignores symmetries

        # Goal is at (0.04, -0.1)

        # Rotation invariant form
        # Three frames:
        # - World frame                  : w
        # - Goal frame                   : g
        # - New/non-rotaded object frame : n
        # goal frame:
        ogx = obj_x
        ogy = obj_y
        phi = np.arctan2(ogy, ogx)                  # object polar rotation coordinate in goal frame
        θ_quadrant = ((obj_θ - phi) % (np.pi / 2))  # rotation in "object" frame
        x_ = eef_x - (obj_x + 0.04)
        y_ = eef_y - (obj_y - 0.1)
        eef_n_x = np.cos(-phi) * x_ - np.sin(-phi) * y_
        eef_n_y = np.sin(-phi) * x_ + np.cos(-phi) * y_

        return obs * np.array([10.0, 10.0, 1, 1, 1, 1, 1, 1])

    def reset_model(self):
        self.init_qpos[:7] = np.array([-0.84266723,
                                        0.36546416,
                                        0.00895722,
                                        0.00415251,
                                       -0.51505002,
                                        0.13013919,
                                        0.10625207])

        # cube init
        self.init_qpos[-3:-1] = np.random.randn(2) * 0.03
        self.init_qpos[-1] = np.random.uniform(-np.pi / 4, np.pi / 4)
        self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = 1.5
        self.viewer.cam.elevation = -30
        self.viewer.cam.azimuth = 180

    def body_index(self, body_name):
        return self.model.body_names.index(body_name)

    def body_pos(self, body_name):
        ind = self.body_index(body_name)
        return self.data.body_xpos[ind]

    def body_vel(self, body_name):
        ind = self.body_index(body_name)
        return self.data.body_xvelp[ind]

    def body_quat(self, body_name):
        ind = self.body_index(body_name)
        return model.body_quat[ind]


    def body_frame(self, body_name):
        """
        Returns the rotation matrix to convert to the frame of the named body
        """
        ind = self.body_index(body_name)
        b = self.data.body_xpos[ind]
        q = self.data.body_xquat[ind]
        qr, qi, qj, qk = q
        s = np.square(q).sum()
        R = np.array([
            [1 - 2 * s * (qj ** 2 + qk ** 2), 2 * s * (qi * qj - qk * qr), 2 * s * (qi * qk + qj * qr)],
            [2 * s * (qi * qj + qk * qr), 1 - 2 * s * (qi ** 2 + qk ** 2), 2 * s * (qj * qk - qi * qr)],
            [2 * s * (qi * qk - qj * qr), 2 * s * (qj * qk + qi * qr), 1 - 2 * s * (qi ** 2 + qj ** 2)]
        ])
        return R

    def rotational_axis(self, body_name, local_axis):
        R = self.body_frame(body_name)
        return R @ local_axis


    def reference_vector(self, body_name, eef_name):
        pj = self.body_pos(body_name)
        pe = self.body_pos(eef_name)
        return pe - pj

    def jacobian(self):
        J = np.zeros((6, 7))
        for j, body_name in enumerate([
            'yumi_link_1_r',
            'yumi_link_2_r',
            'yumi_link_3_r',
            'yumi_link_4_r',
            'yumi_link_5_r',
            'yumi_link_6_r',
            'yumi_link_7_r',
        ]):
            k = self.rotational_axis(body_name, np.array([[0, 0, 1]]).T)
            r = self.reference_vector(body_name, 'gripper_r_base')
            b = self.body_pos(body_name)
            c = np.cross(k.reshape(1, 3), r.reshape(1, 3)).flatten()
            J[:3, j] = c.flatten()
            J[3:, j] = k.flatten()

        return J

    def jacobian_inv(self):
        J = np.zeros((6, 7))
        for j, body_name in enumerate([
            'yumi_link_1_r',
            'yumi_link_2_r',
            'yumi_link_3_r',
            'yumi_link_4_r',
            'yumi_link_5_r',
            'yumi_link_6_r',
            'yumi_link_7_r',
        ]):
            k = self.rotational_axis(body_name, np.array([[0, 0, 1]]).T)
            r = self.reference_vector(body_name, 'gripper_r_base')
            b = self.body_pos(body_name)
            c = np.cross(k.reshape(1, 3), r.reshape(1, 3)).flatten()
            J[:3, j] = c.flatten()
            J[3:, j] = k.flatten()

        return J.T @ np.linalg.inv(J @ J.T + np.eye(6) * 1e-9)


class YumiController:

    def __init__(self, env):
        self.e_prev = None

        self.qlim_lower = np.array([-2.94088,
                                    -2.50455,
                                    -2.94088,
                                    -2.15548,
                                    -5.06145,
                                    -1.53589,
                                    -3.99680]) + 0.07
        self.qlim_upper = np.array([2.940880,
                                    0.759218,
                                    2.940880,
                                    1.396260,
                                    5.061450,
                                    2.408550,
                                    3.996800]) - 0.07
        self.G = np.hstack((np.vstack((np.eye(7), -np.eye(7))), np.zeros((14, 6))))

        θ = 0.0
        R = np.array([[np.cos(θ), -np.sin(θ),  0],
                      [np.sin(θ),  np.cos(θ),  0],
                      [        0,          0,  1]])
        self.Fd = np.array([[-1,  0,  0],
                            [ 0,  1,  0],
                            [ 0,  0, -1]]) @ R

        self.env = env

    def __call__(self, x, y):
        q = self.env.sim.data.qpos[:7]
        h = 1.5 * np.concatenate([self.qlim_upper - q, q - self.qlim_lower]).reshape(-1, 1) # joint limit constraint
        limit_distance = min((q - self.qlim_lower).min(), (self.qlim_upper - q).min())
        if limit_distance < 0:
            print('angle limit:', limit_distance)

        e = np.zeros(6)
        xc = self.env.body_pos('gripper_r_base')
        F = self.env.body_frame('gripper_r_base')
        e[:3] = np.array([x, y, 0.15]) - xc
        e[3:] = 0.5 * np.cross(F.T, self.Fd.T).sum(axis=0)

        # limit the norm of the error in x and y, helps end effector stay in z-plane
        xy_error_norm = np.linalg.norm(e[:2])
        if xy_error_norm > 0.005:
            e[:2] = 0.002 * e[:2] / xy_error_norm

        J_inv = self.env.jacobian_inv()
        if self.e_prev is None:
            self.e_prev = e
        de = e - self.e_prev
        Kp = 64.0 * np.array([1, 1, 1, 0.25, 0.25, 0.25])
        Kd = 64.0 * np.array([1, 1, 1, 1, 1, 1])
        u = J_inv @ (Kp * (e + Kd * de))
        self.e_prev = e

        P = np.eye(7 + 6)
        P[7:, 7:] *= 128.0  # slack punishment
        q = np.zeros((7 + 6, 1))
        b = (Kp * (e + Kd * de)).reshape(-1, 1)
        A = np.hstack((self.env.jacobian(), np.eye(6)))
        P, q, G, h, A, b = map(cvxopt.matrix, [P, q, self.G, h, A, b])
        res = cvxopt.solvers.qp(P, q, G=G, h=h, A=A, b=b)
        u = np.array(res['x'])[:7, 0]

        if u.max() > 2:
            u = 2 * u / u.max()

        return u


if __name__ == '__main__':
    from datetime import datetime
    import gym
    env = gym.make('YumiPusher-v0')
    for _ in range(10):
        env.seed(datetime.now().second)
        o = env.reset()
        xs = []
        ys = []
        for θ in np.linspace(0, 600, 200):
            θ = θ / 200
            a = [2 * np.cos(θ), 2 * np.sin(θ)]
            for _ in range(int(600 / 200)):
                o, r, _, _ = env.step(a)
                env.render()
