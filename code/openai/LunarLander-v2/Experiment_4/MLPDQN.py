import tensorflow as tf, numpy as np
from utility.exp_replay import ExpReplay
from utility.loss import huber_loss, minimize_and_clip
from utility.utility import get_path
from utility.model import mlp


class MLPDQN(object):
    def __init__(self, params):
        self._env = params['env']
        self.save_path = get_path('tf_log/' + params['env_name'] + '-' + params['experiment'])
        self.logger = params['logger']
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.graph = self.build_graph()
        self.saver = tf.train.Saver()
        self.replay = ExpReplay(50000)
        self.sess.run(tf.global_variables_initializer())

    def clear_storage(self):
        self.replay = ExpReplay(50000)

    def take_action(self, obs, eps):
        if np.random.random() < eps:
            return np.random.randint(self._env.action_space.n)
        return self.sess.run(self.graph['act'], feed_dict={self.graph['s']: np.array(obs)[None]})[0]

    def build_graph(self):
        s = tf.placeholder(tf.float32, [None] + list(self._env.observation_space.shape))
        a = tf.placeholder(tf.int32, [None])
        r = tf.placeholder(tf.float32, [None])
        t = tf.placeholder(tf.float32, [None])
        s_ = tf.placeholder(tf.float32, [None] + list(self._env.observation_space.shape))

        network_shape = [list(self._env.observation_space.shape)[0], 256, 512, 256, self._env.action_space.n]
        with tf.variable_scope('q_net', reuse=False):
            q, q_net_w = mlp(s, network_shape)
        q_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_net')

        act = tf.argmax(q, axis=1)

        with tf.variable_scope('q_target_net', reuse=False):
            q_target, q_target_w = mlp(s_, network_shape)
        q_target_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_target_net')

        q_selected = tf.reduce_sum(q * tf.one_hot(a, self._env.action_space.n), 1)
        q_target_best = tf.reduce_max(q_target, 1)

        q_target_best_masked = (1.0 - t) * q_target_best

        q_target_selected = r + 1.0 * q_target_best_masked
        error = q_selected - tf.stop_gradient(q_target_selected)
        loss = tf.reduce_mean(huber_loss(error))

        opt = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = minimize_and_clip(opt, loss, q_var)

        update_target_expr = [tf.assign(t, o) for t, o in zip(q_target_w, q_net_w)]

        return {'act': act, 's': s, 'a': a, 'r': r, 't': t, 's_': s_, 'train_op': train_op,
                'loss': loss, 'update_target_expr': update_target_expr}

    def store_transition(self, s, a, r, t, s_):
        self.replay.add(s, a, r, float(t), s_)

    def train(self):
        s, a, r, t, s_ = self.replay.batch(512)
        _, loss = self.sess.run([self.graph['train_op'], self.graph['loss']],
                                feed_dict={self.graph['s']: s, self.graph['a']: a, self.graph['r']: r,
                                           self.graph['t']: t, self.graph['s_']: s_, })
        return loss

    def update_target(self):
        self.sess.run(self.graph['update_target_expr'])

    def load(self):
        checkpoint = tf.train.get_checkpoint_state(self.save_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            self.logger.info('Successfully loaded: %s' % checkpoint.model_checkpoint_path)
            return True
        else:
            self.logger.info('Could not find old network weights')
            return False

    def save(self):
        self.saver.save(self.sess, self.save_path + '/model.ckpt')
