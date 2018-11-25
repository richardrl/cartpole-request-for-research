# REINFORCE algo implementation

# Pg. 270 of Andrew Sutton's Reinforcement Learning 2018 draft

import gym
import torch
import torch.nn.functional as F
import torch.optim
from tensorboardX import SummaryWriter

def running_avg(arr, N, cur_iter):
    arr = arr.unsqueeze(0)
    return torch.sum(arr.narrow(0, cur_iter - N, N))/N

def policy(params, obs):
    """
    Params: (2, 4) weights

    Obs: (4, ) state/observation vector

    Output: (2, ) action logits vector

    """
    return F.softmax(torch.matmul(params, torch.from_numpy(obs).float()))

def initialize_params():
    return torch.rand((2, 4), requires_grad=True)

def main():
    writer = SummaryWriter()
    params = initialize_params()


    env = gym.make("CartPole-v0")

    max_episodes = 10000
    episodes_rewards = torch.zeros(max_episodes)

    gamma = .99
    episode_max_step = 200

    episode_rewards = torch.zeros(episode_max_step)
    episode_obs = torch.zeros(episode_max_step, 4)
    episode_actions = torch.zeros(episode_max_step)

    opt=torch.optim.Adam([params])

    for ep in range(max_episodes):
        obs = env.reset()
        done = False
        step_count = 0
        while not done:
            action = torch.argmax(policy(params, obs)).detach().numpy()
            obs, reward, done, info = env.step(action)
            episode_rewards[step_count] = reward
            episode_obs[step_count] = torch.from_numpy(obs)
            episode_actions[step_count] = torch.from_numpy(action)
            step_count += 1
        gamma_t = 1
        cumsum = torch.cumsum(episode_rewards, dim=0)
        for ep_step in range(episode_rewards.nonzero().size(0)):   
            gamma_t = gamma_t*gamma
            loss_fnx = gamma * cumsum[ep_step] * F.log_softmax(torch.matmul(params, \
                episode_obs[ep_step].float()))[episode_actions[ep_step].long()]
            opt.zero_grad()
            loss_fnx.backward()
            opt.step()
        # Episode complete, reset variables

        episodes_rewards[ep] = torch.sum(episode_rewards)
        # if ep % 10 == 0 and ep > 0:
        #     print("Average reward (10 trials): " + str(running_avg(episodes_rewards, 10, ep)))
        cum_ep_reward = torch.sum(episode_rewards)
        writer.add_scalar("episode_reward", cum_ep_reward, ep)
        print("episode: "+ str(ep) + " reward:" +str(cum_ep_reward))
        episode_rewards = torch.zeros(episode_max_step) # Reset episode rewards
    print("Max total reward: " + str(torch.max(episodes_rewards)))
    writer.close()

if __name__ == "__main__":
    main()