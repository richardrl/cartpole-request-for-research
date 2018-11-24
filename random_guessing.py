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

def main():
    env = gym.make("CartPole-v0")
    env.reset()

    done =False
    max_iters = 100000
    rewards = np.zeros(max_iters)

    episode_reward = 0
    episode = 0
    parameterConfigurations = np.random.randn(10000, 4)
    # parameterConfigurations = np.random.uniform(low=-1, high=+1, size=(10000, 4))

    # for _ in range(max_iters):
    #     if done:
    #         env.reset()
    #         rewards[episode] = episode_reward
    #         print("episode: "+ str(episode) + " reward:" +str(episode_reward))
    #         episode_reward = 0
    #         episode += 1
    #         if episode % 10 == 0:
    #             print("Average reward (10 trials): " + str(running_avg(rewards, 10, episode)))
    #     obs, reward, done, info = env.step(int(np.rint(np.random.uniform(0, 1))))
    #     print("obs: " + str(obs))
    #     episode_reward+=reward
    #     env.render()
    # import ipdb; ipdb.set_trace()
    for episode in range(parameterConfigurations.shape[0]):
        obs = env.reset()
        while not done:
            # obs, reward, done, info = env.step(forward(parameterConfigurations[episode], obs))
            obs, reward, done, info = env.step(forward(np.array([-0.05224341, 0.32173872, 1.4241636, 0.86725748]), obs))
            episode_reward+=reward
        rewards[episode] = episode_reward
        print("episode: "+ str(episode) + " reward:" +str(episode_reward))
        episode_reward = 0
        done=False
    print("Max total reward: " + str(np.max(rewards)))
    print("Params: " + str(parameterConfigurations[np.argmax(rewards)]))

if __name__ == "__main__":
    main()