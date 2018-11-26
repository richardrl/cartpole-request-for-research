# REINFORCE algo implementation

# Pg. 270 of Andrew Sutton's Reinforcement Learning 2018 draft

import gym
import torch
import torch.nn.functional as F
import torch.optim
from tensorboardX import SummaryWriter
import numpy as np

def policy(theta, s):
    """
    Input
    Theta: (4, 2) weights

    s: (4, ) state/observation vector

    Output

    Dist: (2, ) probability distribution over binary actions

    """
    return F.softmax(torch.matmul(torch.from_numpy(s).float(), theta))

def value(w, s):
    pass

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

    states: (?, 4)

    sampled_actions: (?, 2)
    --- These are the actions actually taken during the trajectory rollouts

    advantages: (?, 1)
    --- Reward-to-go - baseline

    sample_policy: pi(A_t|s_t;params). Policy evaluated at sampled action A_t
    

    Return
    loss: -(expected return)
    """
    action_logits = torch.matmul(states, theta)

    # Use one-hot-encoded sampled_actions to zero out the non-sampled action
    # import ipdb; ipdb.set_trace()
    sample_policy = torch.sum(F.log_softmax(action_logits, dim=1)*sampled_actions, dim=1) 
    # gammas = torch.from_numpy(np.asarray([gamma**i for i in range(states.size(0))])).float()
    # import ipdb; ipdb.set_trace()
    return -torch.mean(advantages * sample_policy) # Positive * positive * negative

def list2tensor(lst):
    return torch.from_numpy(np.asarray(lst)).float()

def collect_trajectory(env, theta):
    obs =env.reset()
    done=False
    states = []
    sampled_actions = []
    rewards = []
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
        advantage = 0
        gamma_t = 1
        for timestep in range(start_time, ep_len-start_time):
            advantage += rewards[timestep]*gamma_t
            gamma_t = gamma*gamma_t
        advantages.append(advantage)

    total_return = np.sum(rewards)
    return list2tensor(states), list2tensor(sampled_actions), list2tensor(advantages), total_return

def main():
    writer = SummaryWriter()

    env = gym.make("CartPole-v0")

    max_batches = 10000

    gamma = .97
    episode_max_step = 200

    batch_size = 1
    theta = initialize_params()

    opt=torch.optim.Adam([theta], lr=.01)
    for episode in range(max_batches):
        states, sampled_actions, advantages, total_return = collect_trajectory(env, theta)
        loss = episode_loss(theta, states, sampled_actions, advantages)
        writer.add_scalar("episode loss ", loss, episode)
        writer.add_scalar("episode return ", total_return, episode)
        print("episode return: " + str(total_return))
        if total_return > 195:
            print("Total return > 195: completed in " + str(episode) + " episodes")
            break
        opt.zero_grad()
        loss.backward()
        writer.add_scalar("gradient norm", torch.norm(theta.grad, 2), episode)
        opt.step()
    writer.close()

if __name__ == "__main__":
    # Set seed
    torch.manual_seed(1)
    import sys
    for run in range(int(sys.argv[1])):
        main()