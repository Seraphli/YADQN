import gym, numpy as np
from agent.MLPDQN import MLPDQN

env = gym.make("CartPole-v0")
agent = MLPDQN(env)
s = env.reset()
episode_rewards = [0.0]
loss = 0
for step in range(100000):
    eps = max(min(1.0 + (0.02 - 1.0) * step / 10000, 1.0), 0.02)
    a = agent.take_action(s, eps)
    s_, r, t, info = env.step(a)
    agent.store_transition(s, a, r, t, s_)
    s = s_
    episode_rewards[-1] += r
    if t:
        s = env.reset()
        episode_rewards.append(0)

    if step > 1000:
        loss = agent.train()

    if step > 1000 and step % 500 == 0:
        agent.update_target()

    if t and len(episode_rewards) % 10 == 0:
        mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
        num_episodes = len(episode_rewards)
        print('t: %.2e, n:%d, r:%.2f, ep:%d, l:%.3e' % (step, num_episodes, mean_100ep_reward, int(eps * 100), loss))
