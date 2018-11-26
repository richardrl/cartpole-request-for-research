# Usage
python <script.py>

# Random guessing
# Hill climbing
# REINFORCE (vanilla policy gradient) algorithm 
- Ala page 270 of Sutton et. al 2018 "Reinforcement Learning" implemented in PyTorch

# Notes
- Make sure obs/state is recorded BEFORE env.step() to produce correct (s,a,r) tuple for trajectories
- Without fixed seed: compared to Tensorflow version, PyTorch version has much higher variance wrt episodes till convergence
- With fixed seed: PyTorch version has lower variance and usually converges 1/4 of episodes

# References
- OpenAI Requests for Research Cartpole: https://openai.com/requests-for-research/#cartpole
- Benchmarking Deep Reinforcement Learning for Continuous Control pg 3: https://arxiv.org/pdf/1604.06778.pdf
- DeepRL HW2: http://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw2.pdf
