import gym
import numpy as np

def running_avg(arr, N, cur_iter):
    return np.sum(arr[cur_iter-N:cur_iter])/N

def forward(params, obs):
    """
    Params: (4, ) weights

    """
    sign = np.sign(np.dot(params, obs))
    return 1 if sign > 0 else 0

def get_test_params(params):
    return np.random.normal(loc=params, scale=1, size=params.shape)

def main():
    env = gym.make("CartPole-v0")
    obs = env.reset()

    done =False
    max_iters = 100000
    rewards = np.zeros(max_iters)

    episode_reward = 0
    episode = 0
    parameters = np.random.uniform(low=-1, high=+1, size=(4,))
    new_params = get_test_params(parameters)

    for _ in range(max_iters):
        if done:
            env.reset()
            rewards[episode] = episode_reward
            if episode_reward > np.max(rewards):
                parameters = new_params
            print("episode: "+ str(episode) + " reward:" +str(episode_reward))
            episode_reward = 0
            episode += 1
            if episode % 10 == 0:
                print("Average reward (10 trials): " + str(running_avg(rewards, 10, episode)))
            done = False
            new_params = get_test_params(parameters)
        obs, reward, done, info = env.step(forward(new_params, obs))
        episode_reward+=reward
    print("Max total reward: " + str(np.max(rewards)))

if __name__ == "__main__":
    main()