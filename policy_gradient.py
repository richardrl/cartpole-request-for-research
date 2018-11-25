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

def policy(theta, s):
    """
    Theta: (2, 4) weights

    s: (4, ) state/observation vector

    Output: (2, ) action logits vector

    """
    return F.softmax(torch.matmul(theta, torch.from_numpy(s).float()))

def value(w, s):
    pass

def initialize_params():
    return torch.rand((2, 4), requires_grad=True)

def sample_bernoulli_distribution(dist):
    return 1 if torch.rand(1)[0] < dist[1] else 0 

def main():
    writer = SummaryWriter()
    params = initialize_params()


    env = gym.make("CartPole-v0")

    max_batches = 10000

    gamma = .97
    episode_max_step = 200

    batch_size = 1

    opt=torch.optim.Adam([params], lr=2**-13)

    episode_rewards_n = torch.zeros([batch_size, episode_max_step])
    episode_obs_n = torch.zeros([batch_size, episode_max_step, 4])
    episode_actions_n = torch.zeros([batch_size, episode_max_step])

    for batch in range(max_batches):
        traj_idx = 0
        while traj_idx < batch_size:       
            step_count = 0
            obs = env.reset()
            done = False
            while not done:
                dist = policy(params, obs)
                # print("dist " + str(dist))
                action = sample_bernoulli_distribution(dist)
                obs, reward, done, info = env.step(action)
                # env.render()
                episode_rewards_n[traj_idx][step_count] = reward
                # import ipdb; ipdb.set_trace()
                episode_obs_n[traj_idx][step_count] = torch.from_numpy(obs)
                episode_actions_n[traj_idx][step_count] = action
                step_count += 1
            if step_count > 0:
                traj_idx += 1
        gamma_t = 1

        # Build loss for policy gradient update
        loss_fnx = 0

        for traj_idx in range(batch_size):
            episode_rewards = episode_rewards_n[traj_idx]
            episode_obs = episode_obs_n[traj_idx]
            episode_actions = episode_actions_n[traj_idx]
            traj_loss = 0
            T = episode_rewards.nonzero().size(0)
            for ep_step in range(T):
                gamma_t = gamma_t*gamma
                partial_loss = F.log_softmax(torch.matmul(params, \
                    episode_obs[ep_step].float()))[episode_actions[ep_step].long()]
                # import ipdb; ipdb.set_trace()
                G = torch.sum(episode_rewards.narrow(0, ep_step, episode_rewards.size(0)-ep_step))
                traj_loss += gamma_t* G * partial_loss
                # print("params grad " + str(params.grad))
                # print("partial loss" + str(partial_loss))
                # print("loss fnx " + str(loss_fnx))
                # print("action taken " + str(episode_actions[ep_step].long()))
                # print("\n")
            loss_fnx += traj_loss/T

        # opt.zero_grad()
        # params.grad.data.zero_()
        opt.zero_grad()
        loss_fnx /= batch_size
        loss_fnx = loss_fnx
        loss_fnx.backward()
        opt.step()
        # Episode complete, reset variables

        average_ep_reward = torch.mean(torch.sum(episode_rewards_n, dim=1))
        writer.add_scalar("average_reward_over_batch", average_ep_reward, batch)
        print("batch: "+ str(batch) + " reward:" +str(average_ep_reward))
        if average_ep_reward > 195:
            break

        episode_rewards_n = torch.zeros([batch_size, episode_max_step])
        episode_obs_n = torch.zeros([batch_size, episode_max_step, 4])
        episode_actions_n = torch.zeros([batch_size, episode_max_step])
    writer.close()

if __name__ == "__main__":
    main()