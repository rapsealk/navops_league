#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os

import tensorflow as tf
import numpy as np

physical_devices = tf.config.experimental.list_physical_devices("GPU")
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)
tf.keras.backend.set_floatx('float32')
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

gamma = 0.98


"""
class ActorCritic(tf.keras.Model):
    def __init__(self, learning_rate=2e-4):
        super(ActorCritic, self).__init__()
        self.data = []

        self.dense = tf.keras.layers.Dense(256, activation='relu', input_shape=(-1, 4))
        self.pi = tf.keras.layers.Dense(2)
        self.value = tf.keras.layers.Dense(1)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def call(self, inputs):
        x = self.dense(inputs)
        pi = tf.nn.softmax(self.pi(x) + 1e-10)
        value = self.value(x)
        return pi, value

    def get_pi(self, inputs):
        x = self.dense(inputs)
        return tf.nn.softmax(self.pi(x))

    def get_value(self, inputs):
        x = self.dense(inputs)
        return self.value(x)

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        states = []
        actions = []
        rewards = []
        state_primes = []
        dones = []
        for transition in self.data:
            s, a, r, s_prime, done = transition
            states.append(s)
            actions.append([a])
            rewards.append([r/100.0])
            state_primes.append(s_prime)
            dones.append([bool(done)])

        self.data = []

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        state_primes = np.array(state_primes)
        dones = np.array(dones)

        return states, actions, rewards, state_primes, dones

    def train(self):
        s, a, r, s_prime, done = self.make_batch()

        with tf.GradientTape() as tape:
            td_target = r + gamma * self.get_value(s_prime) * done
            v = self.get_value(s)
            delta = td_target - v   # FIXME: Huber

            pi = self.get_pi(s)
            pi_a = tf.gather(pi, a, axis=1)
            loss = -tf.math.log(pi_a) * delta + tf.losses.Huber()(td_target, v)

            grads = tape.gradient(loss, self.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 40.0)
            _ = self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # print('Loss:', loss)
        return np.sum(loss)
"""


class ActorCriticLSTM(tf.keras.Model):
    def __init__(self, inputs=22, outputs=6, learning_rate=2e-4):
        super(ActorCriticLSTM, self).__init__()
        self.recurrent = tf.keras.layers.LSTM(1024, input_shape=(4, inputs),
                                              recurrent_initializer='he_uniform',
                                              stateful=False,
                                              return_sequences=False)
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        self.pi = tf.keras.layers.Dense(outputs)
        self.value = tf.keras.layers.Dense(1)

        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

        # Agent
        # self.transitions = []

    def call(self, inputs):
        # print('model.call.inputs.shape:', inputs.shape)
        x = self.recurrent(inputs)
        x = self.dropout1(self.dense1(x))
        x = self.dropout2(self.dense2(x))
        pi = tf.nn.softmax(self.pi(x) + 1e-10)
        value = self.value(x)
        return pi, value

    def reset(self):
        self.recurrent.reset_states()

    def get_pi(self, inputs):
        x = self.recurrent(inputs)
        x = self.dropout1(self.dense1(x))
        x = self.dropout2(self.dense2(x))
        return tf.nn.softmax(self.pi(x) + 1e-10)

    def get_value(self, inputs):
        x = self.recurrent(inputs)
        x = self.dropout1(self.dense1(x))
        x = self.dropout2(self.dense2(x))
        return self.value(x)

    """
    def store_transition(self, data):
        self.transitions.append(data)
    def batch(self):
        states = []
        actions = []
        rewards = []
        states_ = []
        dones = []
        for transition in self.transitions:
            s, a, r, s_, done = transition
            states.append(s)
            actions.append([a])
            rewards.append([r])
            states_.append(s_)
            dones.append([bool(done)])
        self.transitions = []
        states = np.asarray(states)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        states_ = np.asarray(states_)
        dones = np.asarray(dones)
        return states, actions, rewards, states_, dones
    """

    def train(self, s, a, r, s_, done, gamma=0.99):
        """
        s, a, r, s_, done = self.batch()
        with tf.GradientTape() as tape:
            td_target = r + gamma * self.get_value(s_) * done
            v = self.get_value(s)
            delta = td_target - v   # FIXME: Huber
            pi = self.get_pi(s)
            pi_a = tf.gather(pi, a, axis=1)
            loss = -tf.math.log(pi_a) * delta + tf.losses.Huber()(td_target, v)
            grads = tape.gradient(loss, self.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 40.0)
            _ = self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        """

        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            s = tf.squeeze(s)
            s_ = tf.squeeze(s_)
            pi, value = self(s)
            _, value_ = self(s_)
            td_target = r + gamma * self.get_value(s_) * done
            v = self.get_value(s)
            delta = td_target - v

            pi = self.get_pi(s)
            # pi_a = tf.gather(pi, tf.cast(a, dtype=tf.int32), axis=1)
            pi_a = tf.gather(pi, np.argmax(a), axis=1)
            delta = tf.cast(delta, dtype=tf.float32)
            td_target = tf.cast(td_target, dtype=tf.float32)
            v = tf.cast(v, dtype=tf.float32)
            loss = -tf.math.log(pi_a) * delta + tf.losses.Huber()(td_target, v)

            grads = tape.gradient(loss, self.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 40.0)
            _ = self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.save_weights(os.path.join(os.path.dirname(__file__), 'actorcritic_lstm.h5'))

        return loss


def main():
    pass
    """
    env = gym.make('CartPole-v1')
    model = ActorCritic()
    print_interval = 20
    score = 0.0
    losses = 0.0

    for n_epi in range(10000):
        done = False
        s = env.reset()

        while not done:
            prob, _ = model(s[np.newaxis, :])
            prob = np.squeeze(prob)
            a = np.random.choice(prob.shape[0], 1, p=prob)[0]
            s_prime, r, done, info = env.step(a)
            model.put_data((s, a, r, s_prime, done))

            s = s_prime
            score += r

        losses += model.train()

        if n_epi % print_interval == 0 and n_epi != 0:
            print('# of episode: {}, average score: {}, average loss: {}'.format(n_epi, score / print_interval, losses / print_interval))
            score = 0.0
            losses = 0.0

        env.reset()
    """


if __name__ == "__main__":
    main()
