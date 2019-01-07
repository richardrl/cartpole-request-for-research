# Usage
python <script.py> <flags> (see file)

Records Tensorboard output logs to "runs" subdirectory. 

1. Random guessing
2. Hill climbing
3. REINFORCE (vanilla policy gradient) algorithm 

# Notes
- Make sure obs/state is recorded BEFORE env.step() to produce correct (s,a,r) tuple for trajectories
- Without fixed seed: compared to Tensorflow version, PyTorch version has much higher variance wrt episodes till convergence
- With fixed seed: PyTorch version has lower variance and usually converges 1/4 of episodes
- Intuition for using a value function as a baseline:

Imagine without a baseline, the advantage associated with state s: (G_s - b_s) where G_s is reward to go from s and b_s is baseline evaluated at s

Let's say we collect a bunch of trajectories and G_s is usually 100.

Now, let's say we run our algorithm loop a few times and update the policy, and during a new trajectory we see state s again.

Except this time, due to variance associated with stochastic transition dynamics or action selection, G'_s is 10. That means our policy is updated based on an outlier target of 10

Clearly, the Monte Carlo nature of these trajectories is leading to huge variance in the target returns we use for updating our policy. How do we reduce that variance?

Let's reimagine the scenario again with a value estimation network baseline, denoted VN.

Since G_s was usually 100 historically, our value network will predict a return ~100 for state s -> VN(s) = 100

Without the baseline, target: 10

With the baseline, target: 100-10 = 90

Clearly the baseline has a huge regularizing effect

# References
- OpenAI Requests for Research Cartpole: https://openai.com/requests-for-research/#cartpole
- Benchmarking Deep Reinforcement Learning for Continuous Control pg 3: https://arxiv.org/pdf/1604.06778.pdf
- DeepRL HW2: http://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw2.pdf
- Vanilla Policy Gradient: https://spinningup.openai.com/en/latest/algorithms/vpg.html#id2
