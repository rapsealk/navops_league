#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
import json
import importlib
from collections import deque
from datetime import datetime
from itertools import count

import gym
import gym_navops   # noqa: F401
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from memory import ReplayBuffer
from utils import generate_id   # SlackNotification, Atomic
from rating import EloRating
from plotboard import WinRateBoard


with open(os.path.join(os.path.dirname(__file__), 'config.json')) as f:
    config = json.loads(''.join(f.readlines()))
    SLACK_API_TOKEN = config["slack"]["token"]

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=3)
parser.add_argument('--env', type=str, default='NavOpsMultiDiscrete-v2')
parser.add_argument('--no-graphics', action='store_true', default=False)
parser.add_argument('--worker-id', type=int, default=0)
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--buffer-size', type=int, default=10000) # 42533 bytes -> 10000 (12GB)
parser.add_argument('--time-horizon', type=int, default=32)   # 2048
parser.add_argument('--seq-len', type=int, default=64)  # 0.1s per state-action
parser.add_argument('--learning-rate', type=float, default=3e-5)
parser.add_argument('--no-logging', action='store_true', default=False)
parser.add_argument('--framework', choices=['pytorch', 'tensorflow'], default='pytorch')
args = parser.parse_args()

if args.framework == 'tensorflow':
    models_impl = importlib.import_module('models.tensorflow_impl')
    # tf.summary.SummaryWriter
elif args.framework == 'pytorch':
    import torch
    models_impl = importlib.import_module('models.pytorch_impl')


environment = args.env
# Hyperparameters
rollout = args.time_horizon
batch_size = args.batch_size
sequence_length = args.seq_len
learning_rate = args.learning_rate
no_logging = args.no_logging

field_hitpoint = -2
field_ammo = -14
field_fuel = -13


class Learner:

    def __init__(self):
        self.session_id = generate_id()

        self._env = gym.make(environment, no_graphics=args.no_graphics, worker_id=args.worker_id, override_path=os.path.join('/', 'Users', 'rapsealk', 'Desktop', 'NavOps-v2'))
        self._buffer = ReplayBuffer(args.buffer_size)
        self._target_group = models_impl.COMAAgentGroup(
            self._env.observation_space.shape[0] * sequence_length,
            self._env.action_space.nvec,
            n=args.n
        )
        self._id = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}-{environment}'
        if not no_logging:
            self._writer = SummaryWriter(f'runs/{self._id}')
            self._plotly = WinRateBoard()

        if not args.no_logging:
            with open(os.path.join(os.path.dirname(__file__), f'{self._id}.log'), 'w') as f:
                experiment_settings = {
                    "session": self.session_id,
                    "id": self._id,
                    "framework": args.framework,
                    "environment": environment,
                    "time_horizon": args.time_horizon,
                    "batch_size": args.batch_size,
                    "sequence_length": args.seq_len,
                    "learning_rate": args.learning_rate
                }
                f.write(json.dumps(experiment_settings))

    def run(self):
        observation_shape = self._env.observation_space.shape[0]

        result_wins_dq = deque(maxlen=10)
        result_draws_dq = deque(maxlen=10)
        result_loses_dq = deque(maxlen=10)
        result_episodes_dq = deque(maxlen=10)
        result_wins = []
        result_draws = []
        result_loses = []

        ratings = (1200, 1200)
        training_step = 0
        for episode in count(1):
            rewards = []
            new_observations = self._env.reset()
            # new_obs1, new_obs2 = self._env.reset()

            observations = [
                np.concatenate([new_observation] * sequence_length)
                for new_observation in new_observations
            ]
            observations = np.asarray(observations)
            print('observations.shape:', observations.shape)

            hidden_states = self._target_group.reset_hidden_states()

            done = False

            while not done:
                batch = []
                for t in range(rollout):
                    if args.framework == 'tensorflow':
                        actions = self._target_group.get_action(observations, hidden_states)
                    elif args.framework == 'pytorch':
                        #h_in = h_out.copy()
                        # [(action, logit, h_out)]
                        actions = self._target_group.get_action(observations, hidden_states)
                        hidden_states = (h_out for _, _, h_out in actions)
                        # actions, logits, h_outs = 
                    # action = np.array([[[action1_m, action1_a]] * args.n]).squeeze()
                    print('[main] len(actions):', len(actions))
                    # FIXME: MultiHead
                    action = np.asarray([[[a//5, (a-5)%4]] for a, _, _ in actions])
                    print('[main] action.shape:', action.shape)
                    action = np.expand_dims(action.squeeze(), axis=0)

                    next_obs, reward, done, info = self._env.step(action)

                    rewards.append(reward[0])

                    next_observations = [
                        np.concatenate((observation[observation_shape:], next_observation))
                        for observation, next_observation in zip(observations, next_obs)
                    ]
                    next_observations = np.asarray(next_observations)
                    print('next_observations.shape:', next_observations.shape)

                    # if args.framework == 'tensorflow':
                    #     batch.append((observations[0], action[0], reward[0], next_observations[0], (prob1_m, prob1_a), not done))
                    # elif args.framework == 'pytorch':
                    #     batch.append((observations[0], action[0], reward[0], next_observations[0], (prob1_m, prob1_a), h_in[0], h_out[0], not done))

                    if done:
                        print(f'[{datetime.now().isoformat()}] Done! ({",".join(list(map(lambda x: str(x[field_hitpoint]), observations)))}) -> {info.get("win", None)}')

                        result_wins.append(info.get('win', -1) == 0)
                        result_loses.append(info.get('win', -1) == 1)
                        result_draws.append(info.get('win', -1) == -1)

                        ratings = EloRating.calc(ratings[0], ratings[1], info.get('win', -1) == 0)
                        if not no_logging:
                            self._writer.add_scalar('r/rewards', np.sum(rewards), episode)
                            self._writer.add_scalar('r/rating', ratings[0], episode)
                            self._writer.add_scalar('logging/hitpoint', observations[0][field_hitpoint], episode)
                            # self._writer.add_scalar('logging/hitpoint_gap', obs1[field_hitpoint] - obs2[field_hitpoint], episode)
                            # self._writer.add_scalar('logging/damage', 1 - obs2[field_hitpoint], episode)
                            self._writer.add_scalar('logging/ammo_usage', 1 - observations[0][field_ammo], episode)
                            self._writer.add_scalar('logging/fuel_usage', 1 - observations[0][field_fuel], episode)
                            if episode % 100 == 0:
                                result_wins_dq.append(np.sum(result_wins))
                                result_draws_dq.append(np.sum(result_draws))
                                result_loses_dq.append(np.sum(result_loses))
                                result_episodes_dq.append(str(episode))
                                result_wins = []
                                result_draws = []
                                result_loses = []
                                data = [tuple(result_wins_dq), tuple(result_draws_dq), tuple(result_loses_dq)]
                                self._plotly.plot(tuple(result_episodes_dq), data)
                                self._plotly.plot_scatter(data)
                                # self._writer.add_scalar('r/wins', np.mean(result_wins), episode)
                                # self._writer.add_scalar('r/loses', np.mean(result_loses), episode)
                                # self._writer.add_scalar('r/draws', np.mean(result_draws), episode)

                                self._target_agent.save(os.path.join(os.path.dirname(__file__), 'checkpoints', f'{environment}-acer-{episode}.ckpt'), episode=episode)
                        break

                    # obs1, obs2 = next_obs1, next_obs2
                    observations = next_observations

                if not no_logging:
                    self._buffer.push(batch)
                    if len(self._buffer) > 5:#00:
                        training_step += 1
                        loss = self._target_agent.train(batch_size, on_policy=True)
                        loss += self._target_agent.train(batch_size)
                        print(f'[{datetime.now().isoformat()}] Loss: {loss} (batch: {len(self._buffer)})')
                        self._writer.add_scalar('loss', loss, training_step)


# @SlackNotification(SLACK_API_TOKEN)
def main():
    Learner().run()


if __name__ == "__main__":
    main()
