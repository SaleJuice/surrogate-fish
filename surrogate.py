from typing import Optional
import copy, time, pickle
import numpy as np

import gymnasium as gym
from gymnasium import register

from utils import *


def analyze_init_condition(val:list):
    assert isinstance(val, list), f"'{val}' is NOT list!"
    if len(val) == 1:
        return val[0]
    elif len(val) == 2:
        return np.random.uniform(*val)
    elif len(val) >= 3:
        return np.random.choice(val)
    else:
        raise NotImplementedError


def convert_observation_to_space(observation):
    '''
    This is copied from offical codes of 'gym'.
    '''
    from collections import OrderedDict

    if isinstance(observation, dict):
        space = gym.spaces.Dict(OrderedDict([(key, convert_observation_to_space(value)) for key, value in observation.items()]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float("inf"), dtype=np.float32)
        high = np.full(observation.shape, float("inf"), dtype=np.float32)
        space = gym.spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


register(
    id='PositionControl-v0',
    entry_point='surrogate:PositionControlEnv',
    max_episode_steps=40,
    kwargs={'init_fish_condition':([0.0], [0.0], [0.0]), 'init_goal_condition':([-1.5, 1.5], [-1.5, 1.5]), 'sparse_reward':False}
)

register(
    id='PositionControl-v1',
    entry_point='surrogate:PositionControlEnv',
    max_episode_steps=40,
    kwargs={'init_fish_condition':([0.0], [0.0], [0.0]), 'init_goal_condition':([-1.5, 1.5], [-1.5, 1.5]), 'sparse_reward':True}
)

class PositionControlEnv(gym.Env):
    '''
        Y
        |
        |   o-> (robotic fish)
        |
        * - - - - X
       O
    '''
    def __init__(self, init_fish_condition=([0.0], [0.0], [0.0]), init_goal_condition=([-1.5, 1.5], [-1.5, 1.5]), sparse_reward=False, **kwargs) -> None:
        # model params based on specific robotic fish
        self.h = 1 / 2  # [s]
        self.dt = self.h  # [s]

        self.M = 7
        self.V_max = 0.3  # [m/s]
        self.D = np.array([[0.14, 0.13], [0.15, 0.16], [0.15, 0.15], [0.15, 0.17], [0.15, 0.16], [0.14, 0.16], [0.16, 0.16]])  # [m/s]
        self.A = np.array([[-0.26, -0.41], [-0.10, -0.24], [-0.06, -0.15], [0.01, -0.04], [0.11, 0.11], [0.20, 0.13], [0.24, 0.38]])  # [rad/s]
        self.X_m = np.array([[-0.26, -0.26], [-0.18, -0.18], [-0.09, -0.09], [0, 0], [0.18, 0.18], [0.26, 0.26], [0.35, 0.35]])  # [rad]
        
        # params
        self.init_fish_condition = init_fish_condition
        self.init_goal_condition = init_goal_condition
        self.sparse_reward = sparse_reward

        self.delta_p = 0.05  # m

        self.__dict__.update(kwargs)

        # space
        self.observation_space = convert_observation_to_space(self.reset()[0])
        self.action_space = gym.spaces.Discrete(self.M)
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.state = {'world':{}, 'robot':{}, 'scalar':{}}
        self.state['scalar']['k'] = 0
        self.state['scalar']['ta'] = int((self.state['scalar']['k']) % 2)
        self.state['scalar']['v'] = 0.0  # [m/s]
        
        self.state['world']['robot_pose'] = [np.array([analyze_init_condition(self.init_fish_condition[0]), analyze_init_condition(self.init_fish_condition[1]), analyze_init_condition(self.init_fish_condition[2])])]
        self.state['world']['task_goal_pose'] = np.array([analyze_init_condition(self.init_goal_condition[0]), analyze_init_condition(self.init_goal_condition[1]), 0])

        # task related processing
        tf_from_robot_to_world = homogeneous_transformation_matrix(self.state['world']['robot_pose'][-1][:-1], self.state['world']['robot_pose'][-1][-1])
        self.state['robot']['task_goal_pose'] = coordinate_transfer(np.linalg.inv(tf_from_robot_to_world), self.state['world']['task_goal_pose'])

        return self._get_obs(), {}
    
    def step(self, action):
        self.state['scalar']['k'] += 1
        self.state['scalar']['ta'] = int((self.state['scalar']['k']) % 2)  # ta = 0 (when k = 0, 2, 4, ...), ta = 1 (when k = 1, 3, 5, ...)
        self.state['scalar']['v'] = self.V_max if self.state['scalar']['k'] > 40 else (1 - 1 / (1.15 ** self.state['scalar']['k'])) * self.V_max

        linear_vel = self.state['scalar']['v'] / self.V_max * self.D[int(action)][self.state['scalar']['ta']]  # m/s
        angular_vel = self.A[int(action)][self.state['scalar']['ta']]  # [rad/s]
        
        self.state['world']['robot_pose'].append(copy.deepcopy(self.state['world']['robot_pose'][-1]))

        dt = 0.001  # [s]
        for k in range(int(self.h / dt)):
            dot_pose = np.array([linear_vel * np.cos(self.state['world']['robot_pose'][-1][2]), linear_vel * np.sin(self.state['world']['robot_pose'][-1][2]), angular_vel])
            self.state['world']['robot_pose'][-1] += dot_pose * dt
        
            # task related processing
            tf_from_robot_to_world = homogeneous_transformation_matrix(self.state['world']['robot_pose'][-1][:-1], self.state['world']['robot_pose'][-1][-1])
            self.state['robot']['task_goal_pose'] = coordinate_transfer(np.linalg.inv(tf_from_robot_to_world), self.state['world']['task_goal_pose'])
            
            terminated = False  # only current reward
            terminated = terminated or np.abs(self.state['world']['robot_pose'][-1][0]) > 1.8 or np.abs(self.state['world']['robot_pose'][-1][1]) > 1.8  # out of pool area
            terminated = terminated or np.linalg.norm(self.state['robot']['task_goal_pose'][:-1]) < self.delta_p

            truncated = False  # with future reward

            if terminated or truncated:
                break
        
        if self.sparse_reward:
            reward = 1 if np.linalg.norm(self.state['robot']['task_goal_pose'][:-1]) < self.delta_p else -0.01
        else:
            reward = - np.linalg.norm(self.state['robot']['task_goal_pose'][:-1]) * (k+1) * dt

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        NotImplementedError
    
    def close(self):
        NotImplementedError
    
    def _get_obs(self):
        diff_p = np.linalg.norm(self.state['robot']['task_goal_pose'][:-1])
        eta = np.arctan2(self.state['robot']['task_goal_pose'][1], self.state['robot']['task_goal_pose'][0])

        obs = np.concatenate((
            np.array([self.state['scalar']['ta']]),
            np.array([self.state['scalar']['v']]),
            np.array([diff_p]),
            np.array([eta]),
            ), axis=0)
        return obs


register(
    id='PathFollowingLine-v0',
    entry_point='surrogate:PathFollowingEnv',
    max_episode_steps=100,
    kwargs={'goal_path_type':'line', 'init_pose_condition':([-0.25, 0.25], [-0.25, 0.25], [-np.pi/2, np.pi/2])}
)

register(
    id='PathFollowingEight-v0',
    entry_point='surrogate:PathFollowingEnv',
    max_episode_steps=100,
    kwargs={'goal_path_type':'eight', 'init_pose_condition':([-0.25, 0.25], [-0.25, 0.25], [-np.pi/2, np.pi/2])}
)

class PathFollowingEnv(gym.Env):
    def __init__(self, goal_path_type:str, init_pose_condition=([0.0], [0.0], [0.0]), **kwargs) -> None:
        self.h = 1 / 2
        self.dt = self.h

        # model params based on specific robotic fish
        self.M = 7  # number of actions
        self.V_max = 0.3  # maximum linear velocity  # m/s
        self.D = np.array([[0.14, 0.13], [0.15, 0.16], [0.15, 0.15], [0.15, 0.17], [0.15, 0.16], [0.14, 0.16], [0.16, 0.16]])  # linear velocity  # m/s
        self.A = np.array([[-0.26, -0.41], [-0.10, -0.24], [-0.06, -0.15], [0.01, -0.04], [0.11, 0.11], [0.20, 0.13], [0.24, 0.38]])  # angular velocity  # rad/s
        self.X_m = np.array([[-0.26, -0.26], [-0.18, -0.18], [-0.09, -0.09], [0, 0], [0.18, 0.18], [0.26, 0.26], [0.35, 0.35]])  # bias param of CPG  # rad
        
        # task params
        self.delta_p = 0.05  # m
        self.delta_phi = 0.1  # rad

        self.R_E = 0.643  # m
        self.R_B = 0.5  # m
        
        self.R_L = 1.0  # m

        self.goal_path_type = goal_path_type
        self.init_pose_condition = init_pose_condition

        self.__dict__.update(kwargs)
        
        # space
        self.action_space = gym.spaces.Discrete(self.M)
        self.observation_space = convert_observation_to_space(self.reset()[0])
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.state = {'world':{}, 'robot':{}, 'scalar':{}}
        self.state['scalar']['k'] = 0
        self.state['scalar']['ta'] = int((self.state['scalar']['k'] + 1) % 2)
        self.state['scalar']['v'] = 0.0

        if self.goal_path_type == 'line':
            x = np.arange(0, 8, 0.01)
            self.state['world']['task_goal_pose'] = np.vstack([x, np.zeros_like(x), np.zeros_like(x)]).T  # line traj
        elif self.goal_path_type == 'line-short':
            x = np.arange(-1.0, 1.0, 0.01)
            self.state['world']['task_goal_pose'] = np.vstack([x, x, np.zeros_like(x)]).T  # line traj
        elif self.goal_path_type == 'eight': 
            with open('./paths/eight.pkl', 'rb') as fp:
                self.state['world']['task_goal_pose'] = pickle.load(fp)

        self.state['world']['robot_pose'] = [self.state['world']['task_goal_pose'][0] + np.array([analyze_init_condition(self.init_pose_condition[0]), analyze_init_condition(self.init_pose_condition[1]), analyze_init_condition(self.init_pose_condition[2])])]
        
        self.state['scalar']['d'] = 0.0
        self.state['scalar']['psi'] = 0.0
        self.state['scalar']['beta'] = 0.0


        # task related processing
        path = self.state['world']['task_goal_pose'][:, :-1]
        p = self.state['world']['robot_pose'][-1]
        n_R_L = int(self.R_L / np.mean(np.linalg.norm(np.diff(path, axis=0), axis=1)))
        
        self.ran = np.array([-n_R_L, n_R_L]).astype(int)
        self.ran[0] = np.max([0, self.ran[0]])
        self.ran[1] = np.min([path.shape[0], self.ran[1]])

        P_i = self.ran[0] + np.linalg.norm(path[self.ran[0]:self.ran[1]+1] - p[:-1], axis=1).argmin()
        P = path[P_i]

        if P_i == 0:
            l_P = (path[P_i + 1] - path[P_i]) / 2
        elif P_i == len(path) - 1:
            l_P = (path[P_i] - path[P_i - 1]) / 2
        else:
            l_P = (path[P_i + 1] - path[P_i - 1]) / 2
        theta_l_P = np.arctan2(l_P[1], l_P[0])

        q_i = P_i + np.abs(np.linalg.norm(path[P_i:self.ran[1]+1] - p[:-1], axis=1) - self.R_E).argmin()
        q = path[q_i]

        pq = q - p[:-1]
        theta_pq = np.arctan2(pq[1], pq[0])

        pP= P - p[:-1]
        theta_pP = np.arctan2(pP[1], pP[0])

        self.ran = np.array([P_i - n_R_L, P_i + n_R_L]).astype(int)
        self.ran[0] = np.max([0, self.ran[0]])
        self.ran[1] = np.min([path.shape[0], self.ran[1]])

        self.state['scalar']['d'] = np.sign(mapping_period(theta_pP - theta_l_P, [-np.pi, np.pi])) * np.linalg.norm(pP, axis=0)
        self.state['scalar']['psi'] = mapping_period(p[2] - theta_l_P, [-np.pi, np.pi])
        self.state['scalar']['beta'] = mapping_period(theta_pq - theta_l_P, [-np.pi, np.pi])

        return self._get_obs(), {}
    
    def step(self, action):
        self.state['scalar']['k'] += 1
        self.state['scalar']['ta'] = int((self.state['scalar']['k'] + 1) % 2)
        self.state['scalar']['v'] = self.V_max if self.state['scalar']['k'] > 40 else (1 - 1 / (1.15 ** self.state['scalar']['k'])) * self.V_max
        
        # ta = 1 (when k = 0, 2, 4, ...), ta = 0 (when k = 1, 3, 5, ...)
        linear_vel = self.state['scalar']['v'] / self.V_max * self.D[int(action)][self.state['scalar']['ta']]  # m/s
        angular_vel = self.A[int(action)][self.state['scalar']['ta']]  # rad/s

        dot_pose = np.array([linear_vel * np.cos(self.state['world']['robot_pose'][-1][2]), linear_vel * np.sin(self.state['world']['robot_pose'][-1][2]), angular_vel])
        self.state['world']['robot_pose'].append(self.state['world']['robot_pose'][-1] + dot_pose * self.h)
        

        # task related processing
        path = self.state['world']['task_goal_pose'][:, :-1]
        p = self.state['world']['robot_pose'][-1]
        n_R_L = int(self.R_L / np.mean(np.linalg.norm(np.diff(path, axis=0), axis=1)))

        P_i = self.ran[0] + np.linalg.norm(path[self.ran[0]:self.ran[1]+1] - p[:-1], axis=1).argmin()
        P = path[P_i]

        if P_i == 0:
            l_P = (path[P_i + 1] - path[P_i]) / 2
        elif P_i == len(path) - 1:
            l_P = (path[P_i] - path[P_i - 1]) / 2
        else:
            l_P = (path[P_i + 1] - path[P_i - 1]) / 2
        theta_l_P = np.arctan2(l_P[1], l_P[0])

        q_i = P_i + np.abs(np.linalg.norm(path[P_i:self.ran[1]+1] - p[:-1], axis=1) - self.R_E).argmin()
        q = path[q_i]

        pq = q - p[:-1]
        theta_pq = np.arctan2(pq[1], pq[0])

        pP= P - p[:-1]
        theta_pP = np.arctan2(pP[1], pP[0])

        self.ran = np.array([P_i - n_R_L, P_i + n_R_L]).astype(int)
        self.ran[0] = np.max([0, self.ran[0]])
        self.ran[1] = np.min([path.shape[0], self.ran[1]])

        self.state['scalar']['d'] = np.sign(mapping_period(theta_pP - theta_l_P, [-np.pi, np.pi])) * np.linalg.norm(pP, axis=0)
        self.state['scalar']['psi'] = mapping_period(p[2] - theta_l_P, [-np.pi, np.pi])
        self.state['scalar']['beta'] = mapping_period(theta_pq - theta_l_P, [-np.pi, np.pi])

        
        terminated = False  # only current reward
        terminated = terminated or np.abs(self.state['scalar']['d']) > self.R_B
        terminated = terminated or P_i == len(path) - 1

        truncated = False  # with future reward
        
        reward = -1 if np.abs(self.state['scalar']['d']) > self.R_B else 1 - np.abs(self.state['scalar']['d']) / self.R_B

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        NotImplementedError
    
    def close(self):
        NotImplementedError
    
    def _get_obs(self):
        obs = np.concatenate((
            np.array([self.state['scalar']['ta']]),
            np.array([self.state['scalar']['v']]),
            np.array([self.state['scalar']['d']]),
            np.array([self.state['scalar']['psi']]),
            np.array([self.state['scalar']['beta']]),
            ), axis=0)
        return obs


register(
    id='PoseRegulation-v0',
    entry_point='surrogate:PoseRegulationEnv',
    max_episode_steps=100,
    kwargs={'init_goal_condition':([-1.5, 1.5], [-1.5, 1.5], [-np.pi, np.pi]), 'novel_init_method':True}
)

register(
    id='PoseRegulationEasy-v0',
    entry_point='surrogate:PoseRegulationEnv',
    max_episode_steps=30,
    kwargs={'init_goal_condition':([1.0, 1.5], [-np.pi/4, np.pi/4], [-np.pi/6, np.pi/6])}
)

register(
    id='PoseRegulationMiddle-v0',
    entry_point='surrogate:PoseRegulationEnv',
    max_episode_steps=60,
    kwargs={'init_goal_condition':([1.0, 2.0], [-np.pi/2, np.pi/2], [-np.pi/2, np.pi/2])}
)

register(
    id='PoseRegulationHard-v0',
    entry_point='surrogate:PoseRegulationEnv',
    max_episode_steps=120,
    kwargs={'init_goal_condition':([0.0, 2.0], [-np.pi, np.pi], [-np.pi, np.pi])}
)

class PoseRegulationEnv(gym.Env):
    def __init__(self, init_goal_condition=([1.0], [1.0], [np.pi/2]), novel_init_method:bool=False, **kwargs) -> None:
        self.h = 1 / 2
        self.dt = self.h

        # model params based on specific robotic fish
        self.M = 7
        self.V_max = 0.3  # m/s
        self.D = np.array([[0.14, 0.13], [0.15, 0.16], [0.15, 0.15], [0.15, 0.17], [0.15, 0.16], [0.14, 0.16], [0.16, 0.16]])  # m/s
        self.A = np.array([[-0.26, -0.41], [-0.10, -0.24], [-0.06, -0.15], [0.01, -0.04], [0.11, 0.11], [0.20, 0.13], [0.24, 0.38]])  # rad/s
        self.X_m = np.array([[-0.26, -0.26], [-0.18, -0.18], [-0.09, -0.09], [0, 0], [0.18, 0.18], [0.26, 0.26], [0.35, 0.35]])  # rad
        
        # task params
        self.delta_p = 0.05  # m
        self.delta_phi = 0.1  # rad

        self.init_goal_condition = init_goal_condition
        self.novel_init_method = novel_init_method

        self.__dict__.update(kwargs)

        # space
        self.action_space = gym.spaces.Discrete(self.M)
        self.observation_space = convert_observation_to_space(self.reset()[0])
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.state = {'world':{}, 'robot':{}, 'scalar':{}}
        self.state['scalar']['k'] = 0
        self.state['scalar']['ta'] = int((self.state['scalar']['k'] + 1) % 2)
        self.state['scalar']['v'] = 0.0

        if self.novel_init_method:
            x, y, theta = analyze_init_condition(self.init_goal_condition[0]), analyze_init_condition(self.init_goal_condition[1]), analyze_init_condition(self.init_goal_condition[2])
            self.state['world']['robot_pose'] = [np.array([x, y, theta])]
            self.state['world']['task_goal_pose'] = np.array([0.0, 0.0, 0.0])
        else:
            self.state['world']['robot_pose'] = [np.array([0.0, 0.0, 0.0])]
            delta_p, delta_phi, eta = analyze_init_condition(self.init_goal_condition[0]), analyze_init_condition(self.init_goal_condition[1]), analyze_init_condition(self.init_goal_condition[2])
            self.state['world']['task_goal_pose'] = np.array([delta_p * np.cos(eta), delta_p * np.sin(eta), delta_phi])

        # task related processing
        tf_from_robot_to_world = homogeneous_transformation_matrix(self.state['world']['robot_pose'][-1][:-1], self.state['world']['robot_pose'][-1][-1])
        self.state['robot']['task_goal_pose'] = coordinate_transfer(np.linalg.inv(tf_from_robot_to_world), self.state['world']['task_goal_pose'])

        return self._get_obs(), {}
    
    def step(self, action):
        self.state['scalar']['k'] += 1
        self.state['scalar']['ta'] = int((self.state['scalar']['k'] + 1) % 2)
        self.state['scalar']['v'] = self.V_max if self.state['scalar']['k'] > 40 else (1 - 1 / (1.15 ** self.state['scalar']['k'])) * self.V_max

        # ta = 1 (when k = 0, 2, 4, ...), ta = 0 (when k = 1, 3, 5, ...)
        linear_vel = self.state['scalar']['v'] / self.V_max * self.D[int(action)][self.state['scalar']['ta']]  # m/s
        angular_vel = self.A[int(action)][self.state['scalar']['ta']]  # rad/s

        self.state['world']['robot_pose'].append(copy.deepcopy(self.state['world']['robot_pose'][-1]))

        dt = self.h / 100
        for _ in range(int(self.h / dt)):
            dot_pose = np.array([linear_vel * np.cos(self.state['world']['robot_pose'][-1][2]), linear_vel * np.sin(self.state['world']['robot_pose'][-1][2]), angular_vel])
            self.state['world']['robot_pose'][-1] += dot_pose * dt

            # task related processing
            tf_from_robot_to_world = homogeneous_transformation_matrix(self.state['world']['robot_pose'][-1][:-1], self.state['world']['robot_pose'][-1][-1])
            self.state['robot']['task_goal_pose'] = coordinate_transfer(np.linalg.inv(tf_from_robot_to_world), self.state['world']['task_goal_pose'])

            terminated = False  # only current reward
            terminated = terminated or (np.linalg.norm(self.state['robot']['task_goal_pose'][:-1]) < self.delta_p and np.abs(self.state['robot']['task_goal_pose'][2]) < self.delta_phi)

            truncated = False  # with future reward

            if terminated or truncated:
                break

        reward = 1 if terminated else -0.001

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        NotImplementedError
    
    def close(self):
        NotImplementedError
    
    def _get_obs(self):
        diff_p = np.linalg.norm(self.state['robot']['task_goal_pose'][:-1])
        diff_phi = self.state['robot']['task_goal_pose'][2]
        eta = np.arctan2(self.state['robot']['task_goal_pose'][1], self.state['robot']['task_goal_pose'][0])

        obs = np.concatenate((
            np.array([self.state['scalar']['ta']]),
            np.array([self.state['scalar']['v']]),
            np.array([diff_p]),
            np.array([diff_phi]),
            np.array([eta]),
            ), axis=0)
        return obs


if __name__ == '__main__':
    from gymnasium.wrappers import TimeLimit
    np.set_printoptions(precision=3, suppress=True, floatmode='fixed', linewidth=150)  # useful for printing
    
    env = TimeLimit(PositionControlEnv(init_fish_condition=([0.0], [0.0], [0.0]), init_goal_condition=([1.0], [1.0])), max_episode_steps=120)
    # env = TimeLimit(PoseRegulationEnv(init_goal_condition=([1.8], [1.57], [2.55])), max_episode_steps=80)
    # env = TimeLimit(PathFollowingEnv(goal_path_type='line', init_pose_condition=([-0.25, 0.25], [-0.25, 0.25], [-np.pi/2, np.pi/2])), max_episode_steps=100)
    print(f"obs_dims: {env.observation_space.shape[0]} | act_nums: {env.action_space.n}")

    
    obs, info = env.reset()
    while True:        
        act = env.action_space.sample()
        next_obs, rew, ter, tru, info = env.step(act)

        print(obs, act, rew, next_obs, ter, tru, info)
        
        obs = next_obs
        
        if ter or tru:
            break