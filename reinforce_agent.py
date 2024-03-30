import os, uuid, random
import numpy as np

import torch
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym


class StateValueNetwork(torch.nn.Module):
    def __init__(self, state_dims:int, hidden_size:list):
        super().__init__()
        self.layers = []
        self.layers.append(torch.nn.Linear(state_dims, hidden_size[0]))
        for i in range(len(hidden_size)-1):
            self.layers.append(torch.nn.Linear(hidden_size[i], hidden_size[i+1]))
        self.layers.append(torch.nn.Linear(hidden_size[-1], 1))
        
        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, state):
        x = state
        for i in range(len(self.layers)-1):
            x = torch.nn.functional.relu(self.layers[i](x))
        x = self.layers[-1](x)
        return x


class DiscretePolicyNetwork(torch.nn.Module):
    def __init__(self, state_dims:int, action_nums:int, hidden_size:list):
        super().__init__()
        self.layers = []
        self.layers.append(torch.nn.Linear(state_dims, hidden_size[0]))
        for i in range(len(hidden_size)-1):
            self.layers.append(torch.nn.Linear(hidden_size[i], hidden_size[i+1]))
        self.layers.append(torch.nn.Linear(hidden_size[-1], action_nums))
        
        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, state):
        x = state
        for i in range(len(self.layers)-1):
            x = torch.nn.functional.relu(self.layers[i](x))
        x = torch.nn.functional.softmax(self.layers[-1](x), dim=-1)
        return x


class ReinforceAgent(object):
    '''REINFORCE algorithm for discrete action space
    '''
    
    def __init__(self, obs_dims:int, act_nums:int, hidden_dims:list, gamma:float=0.99, v_lr:float=1e-2, pi_lr:float=1e-3, with_baseline:bool=True, device:str='cpu', **kwargs):
        self.gamma = gamma
        self.with_baseline = with_baseline
        self.device = torch.device(device)

        self.__dict__.update(kwargs)
        
        self.v_net = StateValueNetwork(obs_dims, hidden_dims).to(self.device)
        self.v_net_optim = torch.optim.Adam(self.v_net.parameters(), lr=v_lr)

        self.pi_net = DiscretePolicyNetwork(obs_dims, act_nums, hidden_dims).to(self.device)
        self.pi_net_optim = torch.optim.Adam(self.pi_net.parameters(), lr=pi_lr)

    @property
    def name(self):
        return 'REINFORCE'

    def set_train(self):
        self.v_net.train()
        self.pi_net.train()

    def set_eval(self):
        self.v_net.eval()
        self.pi_net.eval()
    
    def save_model(self, dir):
        if not os.path.exists(dir): 
            os.makedirs(dir)
        torch.save(self.v_net.state_dict(), os.path.join(dir, "v.pth"))
        torch.save(self.pi_net.state_dict(), os.path.join(dir, "pi.pth"))

    def load_model(self, dir):
        assert os.path.exists(dir), f"Directory '{dir}' of weights and biases is NOT exist."
        self.v_net.load_state_dict(torch.load(os.path.join(dir, "v.pth")))
        self.pi_net.load_state_dict(torch.load(os.path.join(dir, "pi.pth")))

    def get_action(self, obs:np.ndarray, deterministic=False) -> torch.Tensor:
        prob = self.pi_net(torch.FloatTensor(obs).to(self.device))
        if deterministic:
            act = torch.argmax(prob, dim=1)
        else:
            act = Categorical(prob).sample()
        return act.detach().cpu().numpy()
    
    def update(self, one_episode_traj):
        self.set_train()

        obs, act, rew, next_obs, ter, tru = one_episode_traj
        
        # calculate the discount return
        mc_v = [rew[-1]]
        for r in reversed(rew[:-1]):
            mc_v.append(r + self.gamma * mc_v[-1])
        mc_v.reverse()
        
        obs = torch.FloatTensor(obs).to(self.device)
        act = torch.LongTensor(act).unsqueeze(1).to(self.device)  # 'LongTensor' for discrete action 
        mc_v = torch.FloatTensor(mc_v).unsqueeze(1).to(self.device)

        # update state-value function (V function) using MC (Monte Carlo) method
        curr_v = self.v_net(obs)
        v_loss = torch.nn.MSELoss()(curr_v, mc_v.detach())  # can using 'SmoothL1Loss' for less sensitive to outliers

        self.v_net_optim.zero_grad()
        v_loss.backward()
        self.v_net_optim.step()

        # update policy function (pi function)
        advantage = (mc_v - curr_v).detach()
        if self.with_baseline:
            self.pi_net(obs)
            pi_loss = (-torch.log(self.pi_net(obs).gather(dim=1, index=act)) * advantage).sum()
        else:
            pi_loss = (-torch.log(self.pi_net(obs).gather(dim=1, index=act)) * mc_v).sum()
        
        self.pi_net_optim.zero_grad()
        pi_loss.backward()
        self.pi_net_optim.step()
        
        return {'loss/v':v_loss.detach().cpu().numpy(), 'loss/pi':pi_loss.detach().cpu().numpy()}


if __name__ == '__main__':
    # hyper-params
    max_episodes = 1e4

    # env
    env_name = ['CartPole-v1', 'LunarLander-v2'][0]
    env = gym.make(env_name)

    # agent
    assert isinstance(env.action_space, gym.spaces.Discrete), "Only support DISCRETE action space yet."
    agent = ReinforceAgent(env.observation_space.shape[0], env.action_space.n, hidden_dims=[64], gamma=0.96, v_lr=1e-2, pi_lr=1e-3, with_baseline=True, device='cuda')

    # logger
    uid = str(uuid.uuid1()).split('-')[0]
    logger = SummaryWriter(log_dir=f"./tensorboard/{env_name}/{agent.name}/{uid}")
    
    # training
    steps, episodes, returns = 0, 0, []
    while episodes <= max_episodes:
        episode_rew = 0
        episode_len = 0
        buffer = []

        obs, _ = env.reset()
        while True:  # one rollout
            
            act = agent.get_action(obs, deterministic=False)
            next_obs, rew, ter, tru, _ = env.step(act)
            
            buffer.append((obs, act, rew, next_obs, ter, tru))
            obs = next_obs

            steps += 1
            episode_len += 1
            episode_rew += rew
            
            if ter or tru:
                break

        episodes += 1

        loss_log = agent.update(map(np.stack, zip(*buffer)))

        returns.append(episode_rew)
        average_return = np.array(returns).mean() if len(returns) <= 50 else np.array(returns[-51:-1]).mean()

        # verbose
        print(f"Steps: {steps} | Episodes: {episodes} | Episode Length: {episode_len} | Episode Reward: {episode_rew} | Average Return: {average_return}")
        
        # logging
        logger.add_scalar('episodic/return', episode_rew, episodes)
        logger.add_scalar('episodic/length', episode_len, episodes)
        logger.add_scalar('episodic/return(average)', average_return, episodes)
        if loss_log is not None:
            for key, value in loss_log.items():
                logger.add_scalar(key, value, episodes)

        # save model
        if episodes % 100 == 0 or (len(returns) > 50 and average_return >= 0.94):
            agent.save_model(f'./chkpts/{uid}/{episodes}/')

        if len(returns) > 50 and average_return >= 0.94:
            print(f"Training SUCCESSFUL!({uid})")
            break
