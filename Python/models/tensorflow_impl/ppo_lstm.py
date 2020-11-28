#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os

import numpy as np
import tensorflow as tf


class ProximalPolicyOptimizationLSTM(tf.keras.Model):

    def __init__(self, n):
        super(ProximalPolicyOptimizationLSTM, self).__init__()

        self.recurrent = tf.keras.layers.LSTM(1024, input_shape=(4, 22),
                                              recurrent_initializer='he_uniform',
                                              stateful=False,
                                              return_sequences=False)
        self.dense1 = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        self.dense_policy = tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_uniform')
        self.policy = tf.keras.layers.Dense(n, activation='softmax')
        self.dense_value = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform')
        self.value = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.recurrent(inputs)
        x = self.dropout1(self.dense1(x))
        x = self.dropout2(self.dense2(x))
        p = self.policy(self.dense_policy(x))
        v = self.value(self.dense_value(x))
        return p, v


class Agent:

    def __init__(self, n, gamma=0.99, lambda_=0.95, learning_rate=1e-3):
        self.gamma = gamma
        self.lambda_ = lambda_

        self.n = n
        self.model = ProximalPolicyOptimizationLSTM(n)
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

        self.batch_size = 128
        self.epsilon = 0.2
        self.normalize = True

    def get_action(self, state):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        policy, _ = self.model(state)
        policy = np.array(policy)[0]
        action = np.random.choice(policy.shape[-1], p=policy)
        return action

    def update(self, states, actions, next_states, rewards, dones):
        policy, value = self.model(tf.convert_to_tensor(states, dtype=tf.float32))
        _, next_value = self.model(tf.convert_to_tensor(next_states, dtype=tf.float32))

        policy = policy.numpy()
        values = tf.squeeze(value).numpy()
        if not values.shape:
            values = values[np.newaxis]
        next_values = tf.squeeze(next_value).numpy()
        if not next_values.shape:
            next_values = next_values[np.newaxis]

        advantages, target_values = self.generalized_advantage_estimator(values, next_values, rewards, dones,
                                                                         gamma=self.gamma, lambda_=self.lambda_,
                                                                         normalize=self.normalize)

        for i in range(3):  # epochs
            samples = np.arange(len(states))
            np.random.shuffle(samples)

            total_loss_ = 0

            for b in range(np.ceil(len(states) / self.batch_size).astype(np.uint8)):
                batch = samples[b*self.batch_size:(b+1)*self.batch_size]
                batch_states = [states[i] for i in batch]
                batch_actions = np.array([actions[i] for i in batch])
                batch_target_values = [target_values[i] for i in batch]
                batch_advantages = [advantages[i] for i in batch]
                batch_policy = [policy[i] for i in batch]

                with tf.GradientTape() as tape:
                    tape.watch(self.model.trainable_variables)

                    train_policy, train_value = self.model(tf.convert_to_tensor(batch_states, dtype=tf.float32))
                    train_value = tf.squeeze(train_value)
                    train_advantages = tf.convert_to_tensor(batch_advantages, dtype=tf.float32)
                    train_target_values = tf.convert_to_tensor(batch_target_values, dtype=tf.float32)
                    train_actions = tf.convert_to_tensor(batch_actions, dtype=tf.uint8)
                    train_old_policy = tf.convert_to_tensor(batch_policy, dtype=tf.float32)

                    entropy = tf.reduce_mean(-train_policy * tf.math.log(train_policy + 1e-8)) * 0.1
                    # onehot_action = tf.one_hot(train_actions, self.n)
                    onehot_action = tf.cast(train_actions, dtype=tf.float32)
                    # print('onehot_action:', onehot_action, onehot_action.shape)
                    prob = tf.reduce_sum(train_policy * onehot_action, axis=1)
                    old_prob = tf.reduce_sum(train_old_policy * onehot_action, axis=1)
                    log_pi = tf.math.log(prob + 1e-8)
                    log_old_pi = tf.math.log(old_prob + 1e-8)

                    ratio = tf.exp(log_pi - log_old_pi)
                    clipped_ratio = tf.clip_by_value(ratio, clip_value_min=1-self.epsilon, clip_value_max=1+self.epsilon)
                    minimum = tf.minimum(tf.multiply(train_advantages, clipped_ratio), tf.multiply(train_advantages, ratio))
                    loss_pi = -tf.reduce_mean(minimum) + entropy
                    loss_value = tf.reduce_mean(tf.square(train_target_values - train_value))
                    total_loss = loss_pi + loss_value

                grads = tape.gradient(total_loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                total_loss_ += total_loss

            return total_loss_

    def generalized_advantage_estimator(self, values, next_values, rewards, dones,
                                        gamma=0.99, lambda_=0.95, normalize=True):
        deltas = [r + gamma * (1 - d) * v_ - v
                  for r, v, v_, d in zip(rewards, values, next_values, dones)]
        deltas = np.stack(deltas)

        gaes = np.copy(deltas)
        for t in reversed(range(deltas.shape[0]-1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lambda_ * gaes[t+1]
        target_values = gaes + values

        if normalize:
            gaes = (gaes - np.mean(gaes)) / (np.std(gaes) + 1e-8)

        return gaes, target_values

    def save(self, path=os.path.join(os.path.dirname(__file__), 'model.h5')):
        self.model.save_weights(path)
        print('Model saved at', path)


if __name__ == "__main__":
    pass
