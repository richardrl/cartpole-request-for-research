# REINFORCE algo implementation

# Pg. 270 of Andrew Sutton's Reinforcement Learning 2018 draft
import os

import gym
import torch
import torch.nn.functional as F

import torch.optim
from tensorboardX import SummaryWriter
import numpy as np
import torch.nn as nn
from config import *
import argparse

from datetime import datetime

# parser= argparse.ArgumentParser()
# parser.add_argument('--exp_prefix', required=True, help="Prefix for experiment. Log files saved ./runs/<exp_prefix>...")
# parser.add_argument('--lr_p', required=False, type=float, default=.01, help="Learning rate for policy optimizer. Default .01")
# parser.add_argument('--lr_v', required=False, type=float, default=.1, help="Learning rate for value optimizer. Default .1")
# parser.add_argument('--manual_seed', required=False, type=int, default=None, help="Manual random seed. Default None")
# parser.add_argument('--num_runs', required=False, type=int, default=1, help="# runs. Default 1")
# parser.add_argument('--out_dir', required=False, type=str, help="Outdir for video files")
#
# opt = parser.parse_args()
# print(opt)
#
#
# if opt.manual_seed:
#     torch.manual_seed(opt.manual_seed)


def policy(theta, s):
    """
    Input
    Theta: (4, 2) weights

    s: (4, ) state/observation vector

    Output

    Dist: (2, ) probability distribution over binary actions

    """
    return F.softmax(torch.matmul(torch.from_numpy(s).float(), theta))

class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 10, bias=True),
            nn.ReLU(True),
            nn.Linear(10, 1, bias=True)
            )

    def forward(self, s):
        """
        Input
        State: (?, 4) state vector

        Output
        Expected return: (?) predicted return from state s
        """
        if type(s) != torch.Tensor:
            s = torch.from_numpy(s).float()
        if len(s.size())==2: # shape (?,1) -> (?,)
            return self.model(s).squeeze(1)
        return self.model(s)

def initialize_params():
    import torch.nn.init
    return torch.nn.init.xavier_uniform_(torch.zeros((4,2), requires_grad=True))

def sample_bernoulli_distribution(dist):
    return 0 if torch.rand(1)[0] < dist[0] else 1

def episode_loss(theta, states, sampled_actions, advantages, gamma=.97):
    """
    Inputs
    theta: (4, 2)
    --- Weights for policy network

    states: (T, 4)

    sampled_actions: (T, 2)
    --- These are the actions actually taken during the trajectory rollouts

    advantages: (T, 1)
    --- Reward-to-go - baseline

    sample_policy: (T, 1)
    --- pi(A_t|s_t;params). Policy evaluated at sampled action A_t
    

    Return
    loss: -(expected return)
    """
    # assert len(states.size()) == 3
    action_logits = torch.matmul(states, theta) #(N, T, 2)

    # Use one-hot-encoded sampled_actions to zero out the non-sampled action
    # import ipdb; ipdb.set_trace()
    sample_policy = torch.sum(F.log_softmax(action_logits, dim=1)*sampled_actions, dim=1) #(N, T, 1) = reduce(N,T,2) * (N,T,2)
    # gammas = torch.from_numpy(np.asarray([gamma**i for i in range(states.size(0))])).float()
    # import ipdb; ipdb.set_trace()
    preloss = torch.mean(advantages * sample_policy) # = reduce(reduce[(N, T, 1) * (N, T, 1), dim=1], dim=0)
    return -preloss, preloss # Positive * positive * negative

def list2tensor(lst):
    tmp = torch.from_numpy(np.asarray(lst)).float()
    # if tmp.size().__len__() == 1:
    #     return tmp.unsqueeze(1)
    return tmp

def collect_trajectory(env, theta, value_network):
    obs =env.reset()
    done=False
    states = []
    sampled_actions = []
    rewards = []
    rewards_to_go_arr = []
    advantages = []
    while not done:
        # Sampled_action: scalar action 
        # Dist = [action0_prob, action1_prob]
        dist = policy(theta, obs)
        sampled_action = sample_bernoulli_distribution(dist)
        states.append(obs)
        obs, reward, done, info = env.step(sampled_action)
        one_hot_sampled_action = np.zeros([2])
        one_hot_sampled_action[sampled_action] = 1
        sampled_actions.append(one_hot_sampled_action)
        rewards.append(reward)

    rewards = np.asarray(rewards)

    gamma = .97
    # gammas = np.asarray([gamma**i for i in range(rewards.__len__())])
    # discounted_rewards = gammas * rewards

    # advantages = np.cumsum(discounted_rewards[::-1])[::-1].copy()

    # Build advantages from last reward, 2nd to last reward, etc...

    ep_len= states.__len__()
    for start_time in range(ep_len):
        rewards_to_go = 0
        gamma_t = 1
        for timestep in range(start_time, ep_len-start_time):
            rewards_to_go += rewards[timestep]*gamma_t
            gamma_t = gamma*gamma_t
        rewards_to_go_arr.append(rewards_to_go)
        baseline = value_network(states[start_time])
        # advantage = rewards_to_go - baseline.detach().numpy()
        advantage = rewards_to_go
        advantages.append(advantage)

    total_return = np.sum(rewards)
    return list2tensor(states), list2tensor(sampled_actions), list2tensor(rewards_to_go_arr), list2tensor(advantages), total_return

def main(exp_prefix, lr_p, lr_v, run_num):
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    try:
        os.makedirs(F"./runs/{exp_prefix}")
    except OSError:
        pass
    logdir = F"./runs/{exp_prefix}/lr_p{lr_p}-lr_v{lr_v}-run_num{run_num}-{current_time}"
    writer = SummaryWriter(logdir)

    env = gym.make("CartPole-v1")
    env = gym.wrappers.Monitor(env, main_config['OutDir'], force=True)

    max_batches = 10000

    gamma = .97
    episode_max_step = 500

    batch_size = 1
    theta = initialize_params()

    popt=torch.optim.Adam([theta], lr=lr_p)

    vn = ValueNet()
    vopt = torch.optim.Adam(vn.parameters(), lr=lr_v)

    for episode in range(max_batches):
        states, sampled_actions, rewards_to_go_arr, advantages, total_return = collect_trajectory(env, theta, vn)
        ploss, preloss = episode_loss(theta, states, sampled_actions, advantages)
        writer.add_scalar("policy preloss ", preloss, episode)
        writer.add_scalar("policy loss ", ploss, episode)
        writer.add_scalar("episode return ", total_return, episode)

        popt.zero_grad()
        ploss.backward()
        popt.step()

        writer.add_scalar("policy gradient norm", torch.norm(theta.grad, 2), episode)
        print("episode return: " + str(total_return))
        if total_return > 495:
            print("Total return > 195: completed in " + str(episode) + " episodes")
            break

        vopt.zero_grad()
        # import ipdb; ipdb.set_trace()
        vloss = F.mse_loss(vn(states), rewards_to_go_arr)
        vloss.backward()
        vopt.step()

    writer.close()

if __name__ == "__main__":
    # Set seed
    # torch.manual_seed(1)
    import sys
    if main_config.getint('NumRuns') == 1:
        main(main_config['ExpPrefix'], main_config.getfloat('LrP'), main_config.getfloat('LrV'), 1)
    else:
        for run_num in range(main_config.getint('NumRuns')):
            main(main_config['ExpPrefix'], main_config.getfloat('LrP'), main_config.getfloat('LrV'), run_num)