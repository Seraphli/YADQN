import yaml, os, gym, numpy as np
from gym import wrappers
from agent.MLPDQN import MLPDQN
from utility.utility import disable_other_log, init_logger, get_path, load_config
from utility.exp_replay import PriorityQueue, Sequence


class Game():
    def __init__(self):
        disable_other_log()
        self.env_name = os.path.basename(os.path.dirname(__file__))
        with open('cfg.yml') as f:
            self.cfg = yaml.load(f)
        self.logger = init_logger(self.env_name)

    def train(self):
        cfg = self.cfg['Train']
        env = gym.make(self.env_name)
        agent = MLPDQN({'env': env, 'env_name': self.env_name, 'logger': self.logger})
        agent.load()
        s = env.reset()
        episode_rewards = [0.0]
        episode_w_rewards = [0.0]
        episode_len = [0]
        loss = 0
        save_mean_reward = None
        render_enable = False
        replay_empty = True
        queue = PriorityQueue()
        seq = Sequence()
        for step in range(cfg['TimeStep']):
            if self.train_terminal(locals()):
                break
            eps = max(min(cfg['Epsilon']['Start'] + (cfg['Epsilon']['End'] - cfg['Epsilon']['Start']) *
                          step / cfg['Epsilon']['Total'], cfg['Epsilon']['Start']), cfg['Epsilon']['End'])
            a = agent.take_action(s, eps)
            s_, r, t, info = env.step(a)
            if render_enable:
                env.render()
            wrap_r = r
            # if a == 2:
            #     wrap_r = r - 0.7
            # else:
            #     wrap_r = r
            # wrap_r = r + (1 - s_[0]) * 20 + (1 - s_[1]) * 20
            # agent.store_transition(s, a, wrap_r, t, s_)
            seq.append(s, a, wrap_r, t, s_)
            s = s_
            episode_rewards[-1] += r
            episode_w_rewards[-1] += wrap_r
            episode_len[-1] += 1
            if t:
                s = env.reset()
                mean_100ep_w_reward = np.mean(episode_w_rewards[-101:-1])
                episode_rewards.append(0)
                episode_w_rewards.append(0)
                episode_len.append(0)
                queue.push(-abs(seq.score - mean_100ep_w_reward), seq)
                if len(queue) > 10:
                    queue.pop()

                seq = Sequence()

            if t and len(episode_rewards) % 20 == 0:
                for _seq in queue:
                    for _sample in _seq[1]:
                        agent.store_transition(*_sample)

                replay_empty = False

            if t and not replay_empty:
                losses = []
                for _ in range(200):
                    loss = agent.train()
                    losses.append(loss)
                loss = np.mean(losses)

            if step > 1000 and step % 2000 == 0:
                agent.update_target()

            if t and len(episode_rewards) % 10 == 0:
                mean_100ep_reward = np.mean(episode_rewards[-101:-1])
                mean_100ep_w_reward = np.mean(episode_w_rewards[-101:-1])
                mean_10ep_len = np.mean(episode_len[-11:-1])
                num_episodes = len(episode_rewards)
                self.logger.info('t: %.2e, n:%d, r:%.2f, wr:%.2f, len:%.2f, ep:%d, l:%.3e' % (
                    step, num_episodes, mean_100ep_reward, mean_100ep_w_reward, mean_10ep_len, int(eps * 100), loss))
            if t and len(episode_rewards) % 10 == 0 and eps == cfg['Epsilon']['End'] \
                    and (not save_mean_reward or save_mean_reward < mean_100ep_reward):
                agent.save()
                save_mean_reward = mean_100ep_reward
                self.logger.info('Saved')

    def train_terminal(self, lcls):
        if 'mean_100ep_reward' in lcls.keys() and lcls['mean_100ep_reward'] > 200:
            return True
        return False

    def eval(self):
        env = gym.make(self.env_name)
        tmp_path = get_path('tmp/' + self.env_name + '-experiment-%05d' % np.random.randint(99999))
        self.logger.info('Result will store at `' + tmp_path + '`')
        env = wrappers.Monitor(env, tmp_path)
        agent = MLPDQN({'env': env, 'env_name': self.env_name, 'logger': self.logger})
        agent.load()
        for i_episode in range(1000):
            observation = env.reset()
            for t in range(10000):
                action = agent.take_action(observation, 0)
                observation, reward, done, info = env.step(action)
                if done:
                    self.logger.info("Episode finished after {} timesteps".format(t + 1))
                    break
        env.close()
        gym.upload(tmp_path, api_key=load_config('OpenAI.yml')['APIKEY'])

    def main(self):
        self.train()
        self.eval()


if __name__ == '__main__':
    Game().main()
