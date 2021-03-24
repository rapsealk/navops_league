#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import pathlib
from datetime import datetime

import numpy as np
import tensorflow as tf

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# tf.debugging.set_log_device_placement(True)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError:
        pass


def convert_to_tensor(device, *args):
    return map(np.array, args)


def tf_gather(x, indices):
    indices = tf.cast(indices, dtype=tf.int64)
    return tf.gather_nd(x, tuple(enumerate(indices)))


class MultiHeadLstmActorCriticModel(tf.keras.Model):

    def __init__(self, input_size, output_sizes, hidden_size=256, activation='swish'):
        super(MultiHeadLstmActorCriticModel, self).__init__()
        self._rnn_input_size = 256
        self._rnn_output_size = 128

        if activation == 'relu':
            configs = dict(activation='relu', kernel_initializer='he_uniform')
        elif activation in ('swish', 'silu'):
            configs = dict(activation='swish', kernel_initializer='glorot_uniform')
        else:
            raise NotImplementedError(f'Activation function "{activation}" is not supported.')

        self.encoder = tf.keras.layers.Dense(units=self._rnn_input_size, **configs)
        self.rnn = tf.keras.Sequential([
            tf.keras.layers.LSTM(units=self._rnn_input_size, return_sequences=True),
            tf.keras.layers.LSTM(units=self._rnn_input_size, return_sequences=True),
            tf.keras.layers.LSTM(units=self._rnn_input_size, return_sequences=True),
            tf.keras.layers.LSTM(units=self._rnn_output_size)
        ])
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size, **configs),
            tf.keras.layers.Dense(hidden_size, **configs)
        ])

        movement_action_size, attack_action_size = output_sizes

        self.actor_movement_h = tf.keras.layers.Dense(hidden_size, **configs)
        self.actor_movement_h2 = tf.keras.layers.Dense(hidden_size, **configs)
        self.actor_movement = tf.keras.layers.Dense(movement_action_size, **configs)

        self.critic_movement_h = tf.keras.layers.Dense(hidden_size, **configs)
        self.critic_movement_h2 = tf.keras.layers.Dense(hidden_size, **configs)
        self.critic_movement = tf.keras.layers.Dense(movement_action_size)

        self.actor_attack_h = tf.keras.layers.Dense(hidden_size, **configs)
        self.actor_attack_h2 = tf.keras.layers.Dense(hidden_size, **configs)
        self.actor_attack = tf.keras.layers.Dense(attack_action_size, **configs)

        self.critic_attack_h = tf.keras.layers.Dense(hidden_size, **configs)
        self.critic_attack_h2 = tf.keras.layers.Dense(hidden_size, **configs)
        self.critic_attack = tf.keras.layers.Dense(attack_action_size)

        self.softmax = tf.keras.layers.Softmax(axis=1)

    def call(self, x):
        raise NotImplementedError()

    def get_policy(self, inputs, masking=True):
        x = self.encoder(tf.expand_dims(inputs, axis=0))
        x = tf.reshape(x, (-1, 1, self._rnn_input_size))
        x = self.rnn(x)
        x = self.flatten(x)
        x = self.fc(x)

        x_p_move = self.actor_movement_h(x)
        x_p_move = self.actor_movement_h2(x_p_move)
        logit_movement = self.actor_movement(x_p_move)
        if masking:
            mask = self._generate_mask(inputs, logit_movement.shape[-1])
        else:
            mask = None
        prob_movement = self.softmax(logit_movement, mask)

        x_p_attack = self.actor_attack_h(x)
        x_p_attack = self.actor_attack_h2(x_p_attack)
        logit_attack = self.actor_attack(x_p_attack)
        prob_attack = tf.nn.softmax(logit_attack, axis=-1)

        return prob_movement, prob_attack

    def value(self, x):
        x = self.encoder(tf.expand_dims(x, axis=0))
        x = tf.reshape(x, (-1, 1, self._rnn_input_size))
        x = self.rnn(x)
        x = self.flatten(x)
        x = self.fc(x)

        x_v_move = self.critic_movement_h(x)
        x_v_move = self.critic_movement_h2(x_v_move)
        value_movement = self.critic_movement(x_v_move)

        x_v_attack = self.critic_attack_h(x)
        x_v_attack = self.critic_attack_h2(x_v_attack)
        value_attack = self.critic_attack(x_v_attack)

        return value_movement, value_attack

    def _generate_mask(self, x, output_size):
        rudder_state_max = -3
        rudder_state_min = -7
        engine_state_max = -8
        engine_state_min = -12
        rudder_action_right = 4
        rudder_action_left = 3
        engine_action_forward = 1
        engine_action_backward = 2
        if x.ndim == 1:
            mask = np.ones(output_size)
            if x[rudder_state_max] == 1.0:
                x[rudder_action_right] = 0
            elif x[rudder_state_min] == 1.0:
                x[rudder_action_left] = 0
            if x[engine_state_max] == 1.0:
                x[engine_action_forward] = 0
            elif x[engine_state_min] == 1.0:
                x[engine_action_backward] = 0
        elif x.ndim == 2:
            mask = np.ones((x.shape[0], output_size))
            mask[np.where(x[:, rudder_state_max] == 1.0), rudder_action_right] = 0
            mask[np.where(x[:, rudder_state_min] == 1.0), rudder_action_left] = 0
            mask[np.where(x[:, engine_state_max] == 1.0), engine_action_forward] = 0
            mask[np.where(x[:, engine_state_min] == 1.0), engine_action_backward] = 0
        return mask.astype(np.bool)


class MultiHeadAcerAgent:

    def __init__(
        self,
        model,
        buffer,
        c_sampling_ratio=1.0,
        learning_rate=3e-5,
        cuda=True
    ):
        if cuda and (gpus := tf.config.experimental.list_physical_devices('GPU')):
            self._device = gpus[0]  # tf.device('gpu')
        else:
            self._device = tf.device('cpu')

        self._model = model
        self._optim = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self._gamma = 0.98
        self._c_sampling_ratio = c_sampling_ratio

        self._buffer = buffer

    def get_action(self, state):
        with tf.device('cpu'):
            prob_movement, prob_attack = self._model.get_policy(state)
            # action_movement = tf.random.categorical(tf.expand_dims(prob_movement, axis=0), num_samples=1)
            action_movement = tf.random.categorical(prob_movement, num_samples=1)
            action_movement = tf.squeeze(action_movement).numpy()
            # action_attack = tf.random.categorical(tf.expand_dims(prob_attack, axis=0), num_samples=1)
            action_attack = tf.random.categorical(prob_attack, num_samples=1)
            action_attack = tf.squeeze(action_attack).numpy()
            # prob_movement, prob_attack = map(lambda x: tf.squeeze(x), (prob_movement, prob_attack))
            prob_movement = tf.squeeze(prob_movement)
            prob_attack = tf.squeeze(prob_attack)
        return (action_movement, prob_movement[action_movement]), (action_attack, prob_attack[action_attack])

    # @tf.function
    def train(self, batch_size=4, on_policy=False):
        s, a, r, s_, a_prob, dones, begins = [], [], [], [], [], [], []

        for batch in self._buffer.sample(batch_size, on_policy=on_policy):
            for step in batch:
                for i, data in enumerate(step):
                    s.append(data[0])
                    a.append(data[1])
                    r.append(data[2])
                    s_.append(data[3])
                    a_prob.append(data[4])
                    dones.append(data[5])
                    begins.append(i == 0)

        s, a, r, s_, a_prob, dones, begins = convert_to_tensor(self._device, s, a, r, s_, a_prob, dones, begins)
        r = tf.expand_dims(r, axis=1)
        dones = tf.expand_dims(dones, axis=1)

        # a_movement = tf.expand_dims(a[:, 0], axis=1)
        a_movement = a[:, 0]
        a_movement_prob = tf.expand_dims(a_prob[:, 0], axis=1)
        # a_attack = tf.expand_dims(a[:, 1], axis=1)
        a_attack = a[:, 1]
        a_attack_prob = tf.expand_dims(a_prob[:, 1], axis=1)

        with tf.GradientTape() as tape:
            q_movement, q_attack = self._model.value(s)
            # q_movement = tf.squeeze(q_movement)
            # q_attack = tf.squeeze(q_attack)
            # print(f'q_movement: {q_movement.shape}, q_attack: {q_attack.shape}')
            # q_movement, q_attack = map(lambda logit: tf.squeeze(logit, axis=1), self._model.value(s))
            q_movement_a = tf_gather(q_movement, a_movement)    # tf.gather(q_movement, axis=1, indices=tf.cast(a_movement, dtype=tf.int64))
            q_attack_a = tf_gather(q_attack, a_attack)  # tf.gather(q_attack, axis=1, indices=tf.cast(a_attack, dtype=tf.int64))
            pi_movement, pi_attack = self._model.get_policy(s, masking=False)
            # pi_movement = tf.squeeze(pi_movement, axis=1)

            ### gather
            ###pi_movement_a = tf.gather(pi_movement, axis=1, indices=tf.cast(a_movement, dtype=tf.int64))
            pi_movement_a = tf_gather(pi_movement, a_movement)

            # pi_attack = tf.squeeze(pi_attack, axis=1)
            ###pi_attack_a = tf.gather(pi_attack, axis=1, indices=tf.cast(a_attack, dtype=tf.int64))
            pi_attack_a = tf_gather(pi_attack, a_attack)

            value_movement = tf.math.reduce_sum((q_movement * pi_movement), axis=1, keepdims=True)
            #value_movement = tf.math.reduce_sum(tf.math.multiply(q_movement, pi_movement), axis=1, keepdims=True)
            value_movement = value_movement.numpy()     # tf.stop_gradient(value_movement)
            value_attack = tf.math.reduce_sum((q_attack * pi_attack), axis=1, keepdims=True)
            #value_attack = tf.math.reduce_sum(tf.math.multiply(q_attack, pi_attack), axis=1, keepdims=True)
            value_attack = value_attack.numpy()         # tf.stop_gradient(value_attack)

            # Movement Loss
            # print(f'pi_movement: {pi_movement.shape}, a_movement_prob: {a_movement_prob}')
            rho_movement = tf.math.divide_no_nan(pi_movement.numpy(), a_movement_prob)   # _no_nan
            print(f'rho_movement: {rho_movement.shape}')
            print(f'a_movement: {a_movement.shape}')
            rho_movement_a = tf_gather(rho_movement, a_movement)    #####rho_movement_a = tf.gather(rho_movement, axis=1, indices=tf.cast(a_movement, dtype=tf.int64))
            print(f'rho_movement_a: {rho_movement_a.shape}')
            rho_movement_bar = tf.clip_by_value(rho_movement_a, clip_value_max=self._c_sampling_ratio, clip_value_min=-np.inf)
            print(f'rho_movement_bar: {rho_movement_bar.shape}')
            correction_coeff_movement = tf.math.divide(1 - self._c_sampling_ratio, rho_movement)
            correction_coeff_movement = tf.clip_by_value(correction_coeff_movement, clip_value_max=np.inf, clip_value_min=0)

            print(f'rho_movement_bar: {rho_movement_bar.shape}')
            print(f'value_movement: {value_movement.shape} {value_movement.dtype}, dones: {dones.shape} {dones.dtype}')
            q_retrace_movement = value_movement[-1] * tf.cast(dones[-1], dtype=tf.float32)
            q_retraces_movement = []
            for i in reversed(range(len(r))):
                # print(f'[TensorFlow] q_retrace_movement: {q_retrace_movement.shape}, dones: {dones.shape}')
                # print(f'[TensorFlow] r: {r.shape}, r[i]: {r[i].shape}')
                q_retrace_movement = r[i] + self._gamma * q_retrace_movement
                # q_retraces_movement.append(q_retrace_movement.numpy())
                q_retraces_movement.append(q_retrace_movement.numpy())
                # q_retrace_movement = tf.math.add(value_movement[i], tf.math.multiply(rho_movement_bar[i], tf.math.subtract(q_retrace_movement, q_movement_a[i])))
                q_retrace_movement = rho_movement_bar[i] * (q_retrace_movement - q_movement_a[i]) + value_movement[i]
                if begins[i] and i != 0:
                    q_retrace_movement = value_movement[i-1] * tf.cast(dones[i-1], dtype=tf.float32)
            #print(f'q_retraces_movement[0]: {q_retraces_movement[0]} -> ', end='')
            #q_retraces_movement.reverse()
            #print(q_retraces_movement[0])
            # print(f'q_retraces_movement: {type(q_retraces_movement)} {len(q_retraces_movement)} ({q_retraces_movement[0]})')
            #q_retraces_movement = np.asarray(q_retraces_movement, dtype=np.float32)
            #print(f'q_retraces_movement: {q_retraces_movement.shape} {q_retraces_movement.dtype}')
            #q_retraces_movement = tf.convert_to_tensor([q_retraces_movement], dtype=tf.float32)
            #print(f'q_retraces_movement: {type(q_retraces_movement)} {q_retraces_movement.shape} ({q_retraces_movement[0]})')
            # q_retraces_movement = tf.expand_dims(tf.convert_to_tensor(q_retraces_movement), axis=1)
            # q_retraces_movement = np.expand_dims(np.asarray(q_retraces_movement), axis=1)
            # q_retraces_movement = tf.convert_to_tensor(q_retraces_movement, dtype=tf.float32)

            loss_movement_1 = -tf.math.multiply(tf.math.multiply(rho_movement_bar, tf.math.log(pi_movement_a)), tf.math.subtract(q_retraces_movement, value_movement))
            loss_movement_2 = -correction_coeff_movement * pi_movement.numpy() * tf.math.log(pi_movement) * tf.math.subtract(q_movement.numpy(), value_movement)
            loss_movement = loss_movement_1 + tf.reduce_sum(loss_movement_2, axis=1) + tf.keras.losses.huber(q_movement_a, q_retraces_movement)
            print(f'[TensorFlow] LossMovement#1: {loss_movement_1}')
            print(f'[TensorFlow] LossMovement#2: {loss_movement_2}')
            print(f'[TensorFlow] LossMovement: {loss_movement}')

            # Attack Loss
            rho_attack = tf.math.divide_no_nan(pi_attack.numpy(), a_attack_prob)   # _no_nan
            rho_attack_a = tf_gather(rho_attack, a_attack)  # tf.gather(rho_attack, axis=1, indices=tf.cast(a_attack, dtype=tf.int64))
            rho_attack_bar = tf.clip_by_value(rho_attack_a, clip_value_max=self._c_sampling_ratio, clip_value_min=-np.inf)
            correction_coeff_attack = tf.math.divide(1 - self._c_sampling_ratio, rho_attack)
            correction_coeff_attack = tf.clip_by_value(correction_coeff_attack, clip_value_max=np.inf, clip_value_min=0)

            q_retrace_attack = value_attack[-1] * tf.cast(dones[-1], dtype=tf.float32)
            q_retraces_attack = []
            for i in reversed(range(len(r))):
                q_retrace_attack = r[i] + self._gamma * q_retrace_attack
                # q_retrace_attack = tf.math.add(r[i], tf.math.multiply(tf.constant(self._gamma, dtype=tf.float32), q_retrace_attack))
                q_retraces_attack.append(q_retrace_attack.numpy())
                q_retrace_attack = tf.math.add(value_attack[i], tf.math.multiply(rho_attack_bar[i], tf.math.subtract(q_retrace_attack, q_attack_a[i])))
                if begins[i] and i != 0:
                    q_retrace_attack = value_attack[i-1] * tf.cast(dones[i-1], dtype=tf.float32)
            q_retraces_attack.reverse()
            # q_retraces_attack = tf.expand_dims(tf.convert_to_tensor(q_retraces_attack), axis=1)
            q_retraces_attack = np.expand_dims(np.asarray(q_retraces_attack), axis=1)
            q_retraces_attack = tf.convert_to_tensor(q_retraces_attack, dtype=tf.float32)

            loss_attack_1 = -tf.math.multiply(tf.math.multiply(rho_attack_bar, tf.math.log(pi_attack_a)), tf.math.subtract(q_retraces_attack, value_attack))
            loss_attack_2 = -correction_coeff_attack * pi_attack.numpy() * tf.math.log(pi_attack) * tf.math.subtract(q_attack.numpy(), value_attack)
            loss_attack = loss_attack_1 + tf.reduce_sum(loss_attack_2, axis=1) + tf.keras.losses.huber(q_attack_a, q_retraces_attack)
            print(f'[TensorFlow] LossAttack#1: {loss_attack_1}')
            print(f'[TensorFlow] LossAttack#2: {loss_attack_2}')
            print(f'[TensorFlow] LossAttack: {loss_attack}')

            # Total Loss
            print(f'[TensorFlow] Movement Loss: {loss_movement}')
            print(f'[TensorFlow] Attack Loss: {loss_attack}')
            total_loss = loss_movement + loss_attack
            loss_value = tf.reduce_mean(total_loss).numpy()

        grads = tape.gradient(total_loss, self._model.trainable_variables)
        # Gradient clipping for learning stability
        grads, _ = tf.clip_by_global_norm(grads, 40.0)
        self._optim.apply_gradients(zip(grads, self._model.trainable_variables))

        return loss_value

    def save(self, path):
        self._model.save_weights(path)

    def load(self, path):
        self._model.load_weights(path)

    @property
    def buffer(self):
        return self._buffer


def main():
    x = np.random.uniform(-1.0, 1.0, (3,))
    softmax = tf.keras.layers.Softmax()
    mask = np.asarray([True, False, True], dtype=bool)
    print(x)
    print(softmax(x).numpy())
    print(softmax(x, mask).numpy())
    mask = None
    print(softmax(x, mask).numpy())

    return

    print(tf.test.is_gpu_available(cuda_only=True, min_cuda_compute_capability=None))
    model = MultiHeadLstmActorCriticModel()
    x = np.random.uniform(-1.0, 1.0, (4,))
    y = model(x[np.newaxis, :])
    """
    with tf.GradientTape() as tape:
        grads = tape.gradient(y, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 40.0)
    op = self.optimizer.apply_gradient(zip(grads, self.trainable_variables))
    """
    print(x)
    print(y)


if __name__ == "__main__":
    main()
