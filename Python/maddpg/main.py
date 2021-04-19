#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import random
from collections import deque
from itertools import count
from datetime import datetime

import numpy as np
from torch.utils.tensorboard import SummaryWriter

import gym
import gym_navops   # noqa: F401
from agent import Agent
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import generate_id
from plotboard import WinRateBoard


parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='NavOpsMultiDiscrete-v2')
parser.add_argument('--no-graphics', action='store_true', default=False)
parser.add_argument('--batch-size', type=int, default=1024)
parser.add_argument('--n', type=int, default=3)
# parser.add_argument('--worker-id', type=int, default=0)
# parser.add_argument('--time-horizon', type=int, default=2048)
parser.add_argument('--sequence-length', type=int, default=64)
# parser.add_argument('--learning-rate', type=float, default=1e-3)
# parser.add_argument('--no-logging', action='store_true', default=False)
args = parser.parse_args()


class ReplayBuffer:

    def __init__(self, capacity=1_000_000, seed=0):
        self.capacity = capacity
        self._buffer = deque(maxlen=capacity)
        random.seed(seed)

    def push(self, *args):
        self._buffer.append(args)

    def sample(self, batch_size, on_policy=False):
        batch_size = min(batch_size, len(self._buffer))
        if on_policy:
            return [self._buffer[-1]]
        return random.sample(self._buffer, batch_size)

    def __len__(self):
        return len(self._buffer)


def main():
    build_path = os.path.join('C:\\', 'Users', 'rapsealk', 'Desktop', 'NavOps')
    # build_path = os.path.join(os.path.dirname(__file__), '..', 'NavOps')
    env = gym.make(args.env, no_graphics=args.no_graphics, worker_id=0, override_path=build_path)
    print(f'[navops_league] obs: {env.observation_space.shape[0]}, action: {np.sum(env.action_space.nvec)}')

    agents = [
        Agent(
            env.observation_space.shape[0] * args.sequence_length,
            np.sum(env.action_space.nvec),
            agent_id=i,
            n=args.n
        )
        for i in range(args.n)
    ]
    buffer = ReplayBuffer()
    episode_wins = []
    episode_loses = []
    episode_draws = []

    exprmt_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    # generate_id()
    writer = SummaryWriter(f'runs/{exprmt_id}')
    plotboard = WinRateBoard(dirpath=os.path.join(os.path.dirname(__file__), 'plots', exprmt_id))

    for episode in count(1):
        observations = env.reset()
        observations = [
            np.concatenate([observation] * args.sequence_length, axis=0)
            for observation in observations
        ]
        h_ins = [agent.reset_hidden_states(batch_size=1)[0] for agent in agents]
        for step in count(1):
            actions = []
            h_outs = []
            for agent, observation, h_in in zip(agents, observations, h_ins):
                action, h_out = agent.select_action(observation, h_in)
                actions.append(action)
                h_outs.append(h_out)
            actions = np.array(actions)

            # actions_mh = np.asarray([actions, actions]).T
            actions_mh = [[], []]
            for action in actions:
                actions_mh[0].append(action if action < env.action_space.nvec[0] else 0)
                actions_mh[1].append(action - env.action_space.nvec[0] if action >= env.action_space.nvec[0] else 0)
            actions_mh = np.array(actions_mh)
            actions_mh = np.transpose(actions_mh)

            next_observations, rewards, done, info = env.step(actions_mh)

            next_observationss = []
            for i in range(len(observations)):
                new_obs = np.concatenate((observations[i][env.observation_space.shape[0]:], next_observations[i]))
                next_observationss.append(new_obs)
            """
            next_observations = [
                np.concatenate((observations[env.observation_space.shape[0]:], next_observation), axis=0)
                # np.stack([observation[1:], next_observation], axis=0)
                for observation, next_observation in zip(observations, next_observations)
            ]
            print(f'[main] n_obs_: {next_observations[0].shape}')
            """

            buffer.push(observations, actions, next_observationss, rewards, h_ins, h_outs, done)

            # observations = next_observations
            observations = next_observationss
            h_ins = h_outs

            if done:
                episode_wins.append(info.get('win', -1) == 0)
                episode_loses.append(info.get('win', -1) == 1)
                episode_draws.append(info.get('win', -1) == -1)
                break

        print(f'Episode #{episode} (buffer={len(buffer)}/{args.batch_size})')

        if len(buffer) > args.batch_size:
            train_losses = []
            for agent in agents:
                other_agents = agents.copy()
                other_agents.remove(agent)
                batch = buffer.sample(args.batch_size)
                loss = agent.learn(batch, other_agents)
                train_losses.append(loss)
            print(f'Episode #{episode}: Loss={np.mean(train_losses)}')
            # Tensorboard
            try:
                writer.add_scalar('loss/sum', np.sum(train_losses), episode)
                writer.add_scalar('loss/mean', np.mean(train_losses), episode)
                hps = [next_obs[0] for next_obs in next_observations]
                writer.add_scalar('performance/hp', np.mean(hps), episode)
            except:
                sys.stderr.write(f'[{datetime.now().isoformat()}] [MAIN/TENSORBOARD] FAILED TO LOG LOSS!\n')

        if episode % 100 == 0:
            print(f'Episode #{episode} :: WinRate={np.mean(episode_wins)}')
            # plotly
            ep_wins = [np.sum(episode_wins[i*100:(i+1)*100]) for i in range(episode//100)]
            ep_draws = [np.sum(episode_draws[i*100:(i+1)*100]) for i in range(episode//100)]
            ep_loses = [np.sum(episode_loses[i*100:(i+1)*100]) for i in range(episode//100)]
            data = [ep_wins, ep_draws, ep_loses]
            try:
                plotboard.plot(tuple(map(str, range(1, episode//100+1))), data)
                plotboard.plot(data)
            except:
                sys.stderr.write(f'[{datetime.now().isoformat()}] [MAIN/PLOTLY] FAILED TO PLOT!\n')
            # Tensorboard
            try:
                writer.add_scalar('r/wins', np.mean(episode_wins), episode // 100)
                writer.add_scalar('r/draws', np.mean(episode_draws), episode // 100)
                writer.add_scalar('r/loses', np.mean(episode_loses), episode // 100)
            except:
                sys.stderr.write(f'[{datetime.now().isoformat()}] [MAIN/TENSORBOARD] FAILED TO WRITE TENSORBOARD!\n')

    env.close()


if __name__ == "__main__":
    main()
