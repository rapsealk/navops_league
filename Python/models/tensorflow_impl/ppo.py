#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import gym
import numpy as np
import tensorflow as tf


class ProximalPolicyOptimization(tf.keras.Model):

    def __init__(self):
        super(ProximalPolicyOptimization, self).__init__()

        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.dense_policy = tf.keras.layers.Dense(64, activation='relu')
        self.dense_value = tf.keras.layers.Dense(64, activation='relu')
        self.policy = tf.keras.layers.Dense(2, activation='softmax')
        self.value = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        policy = self.dense_policy(x)
        policy = self.policy(policy)
        value = self.dense_value(x)
        value = self.value(value)

        return policy, value


class Agent:

    def __init__(self, n, rollout=128, gamma=0.99, lambda_=0.95, learning_rate=1e-3):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.learning_rate = learning_rate

        self.n = n
        self.model = ProximalPolicyOptimization()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.rollout = rollout
        self.batch_size = 128
        self.epsilon = 0.2
        self.normalize = True

    def get_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        policy, _ = self.model(state)
        policy = np.array(policy)[0]
        action = np.random.choice(self.n, p=policy)
        return action

    def update(self, state, next_state, reward, done, action):
        policy, value = self.model(tf.convert_to_tensor(state, dtype=tf.float32))
        _, next_value = self.model(tf.convert_to_tensor(next_state, dtype=tf.float32))
        policy = policy.numpy()
        value = tf.squeeze(value).numpy()
        next_value = tf.squeeze(next_value).numpy()

        advantages, target = self.generalized_adavantage_estimator(value, next_value, reward, done,
                                                                   gamma=self.gamma, lambda_=self.lambda_, normalize=self.normalize)

        for _ in range(3):  # (self.epoch):
            samples = np.arange(self.rollout)
            np.random.shuffle(samples)
            samples = samples[:self.batch_size]

            batch_state = [state[i] for i in samples]
            batch_action = [action[i] for i in samples]
            batch_target = [target[i] for i in samples]
            batch_advantages = [advantages[i] for i in samples]
            batch_policy = [policy[i] for i in samples]

            with tf.GradientTape() as tape:
                tape.watch(self.model.trainable_variables)

                train_policy, train_value = self.model(tf.convert_to_tensor(batch_state, dtype=tf.float32))
                train_value = tf.squeeze(train_value)
                train_advantages = tf.convert_to_tensor(batch_advantages, dtype=tf.float32)
                train_target = tf.convert_to_tensor(batch_target, dtype=tf.float32)
                train_action = tf.convert_to_tensor(batch_action, dtype=tf.uint8)
                train_old_policy = tf.convert_to_tensor(batch_policy, dtype=tf.float32)

                entropy = tf.reduce_mean(-train_policy * tf.math.log(train_policy + 1e-8)) * 0.1
                onehot_action = tf.one_hot(train_action, self.n)
                prob = tf.reduce_sum(train_policy * onehot_action, axis=1)
                old_prob = tf.reduce_sum(train_old_policy * onehot_action, axis=1)
                log_pi = tf.math.log(prob + 1e-8)
                log_old_pi = tf.math.log(old_prob + 1e-8)

                ratio = tf.exp(log_pi - log_old_pi)
                clipped_ratio = tf.clip_by_value(ratio, clip_value_min=1-self.epsilon, clip_value_max=1+self.epsilon)
                minimum = tf.minimum(tf.multiply(train_advantages, clipped_ratio), tf.multiply(train_advantages, ratio))
                loss_pi = -tf.reduce_mean(minimum) + entropy
                loss_value = tf.reduce_mean(tf.square(train_target - train_value))
                total_loss = loss_pi + loss_value

            grads = tape.gradient(total_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def generalized_adavantage_estimator(self, values, next_values, rewards, dones,
                                         gamma=0.99, lambda_=0.95, normalize=True):
        deltas = [r + gamma * (1 - d) * nv - v
                  for r, v, nv, d in zip(rewards, values, next_values, dones)]
        deltas = np.stack(deltas)

        gaes = np.copy(deltas)
        for t in reversed(range(deltas.shape[0]-1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lambda_ * gaes[t+1]
        target = gaes + values

        if normalize:
            gaes = (gaes - np.mean(gaes)) / (np.std(gaes) + 1e-8)

        return gaes, target


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    rollout = 128
    agent = Agent(n=env.action_space.n, rollout=rollout)
    episode = 0
    score = 0
    state = env.reset()

    while True:
        states = []
        next_states = []
        actions = []
        rewards = []
        dones = []

        for _ in range(rollout):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            env.render()

            score += reward
            reward = 0

            if done:
                if score == 500:
                    reward = 1
                else:
                    reward = -1

            states.append(state)
            next_states.append(next_state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            state = next_state

            if done:
                print('Episode: %d, score: %f' % (episode, score))
                episode += 1
                score = 0

                state = env.reset()

        """
        print('Before update:')
        states = np.array(states) #np.asarray(states)
        print('- states.shape:', len(states), states[0].shape, type(states[0]))
        next_states = np.array(next_states) # np.asarray(next_states)
        print('- next_states.shape:', len(next_states), next_states[0].shape, type(next_states[0]))
        print('- rewards.shape:', len(rewards), rewards[0].shape, type(rewards[0]))
        print('- dones.shape:', len(dones), dones[0].shape, type(dones[0]))
        print('- actions.shape:', len(actions), actions[0].shape, type(actions[0]))
        """
        agent.update(states, next_states, rewards, dones, actions)
