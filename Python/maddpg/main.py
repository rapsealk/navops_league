#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import pathlib
from itertools import count
from datetime import datetime

import numpy as np
from torch.utils.tensorboard import SummaryWriter

import gym
import gym_navops   # noqa: F401
from agent import Agent
from memory import ReplayBuffer
from database import MongoDatabase
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import generate_id
from plotboard import WinRateBoard

REWARD_IDX = 3


parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='NavOpsMultiDiscrete-v2')
parser.add_argument('--no-graphics', action='store_true', default=False)
parser.add_argument('--batch-size', type=int, default=1024)
parser.add_argument('--n', type=int, default=3)
parser.add_argument('--worker-id', type=int, default=0)
# parser.add_argument('--time-horizon', type=int, default=2048)
parser.add_argument('--sequence-length', type=int, default=32)
# parser.add_argument('--learning-rate', type=float, default=1e-3)
# parser.add_argument('--no-logging', action='store_true', default=False)
args = parser.parse_args()


def discount_with_dones(rewards, dones, gamma=0.998):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r * (1.0 - done)
        # r = r * (1.0 - done)
        discounted.append(r)
    return discounted[::-1]


def main():
    build_path = os.path.join('C:\\', 'Users', 'rapsealk', 'Desktop', 'NavOps')
    # build_path = os.path.join(os.path.dirname(__file__), '..', 'NavOps')
    env = gym.make(args.env, no_graphics=args.no_graphics, worker_id=args.worker_id, override_path=build_path)
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
    # buffer = ReplayBuffer(capacity=50000)
    episode_buffer = ReplayBuffer(capacity=50000)
    episode_wins = []
    episode_loses = []
    episode_draws = []

    exprmt_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    # generate_id()
    writer = SummaryWriter(f'runs/{exprmt_id}')
    plotboard = WinRateBoard(dirpath=os.path.join(os.path.dirname(__file__), 'plots', exprmt_id))

    session_id = generate_id()
    database = MongoDatabase()
    result_db = database.ref("result")
    session_db = database.ref(session_id)
    initial_episode = len(session_db) + 1

    for episode in count(initial_episode):
        observations = env.reset()
        observations = [
            np.concatenate([observation] * args.sequence_length, axis=0)
            for observation in observations
        ]
        h_ins = [agent.reset_hidden_states(batch_size=1)[0] for agent in agents]

        episode_rewards = []

        # TODO
        episode_id = ('0' * 10 + str(episode))[-10:]
        ref = session_db.ref(episode_id)

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
            episode_rewards.append(np.mean(rewards))

            episode_buffer.push(observations, actions, next_observationss, rewards, h_ins, done)

            # TODO: Logging
            position_idx = -env.observation_space.shape[0] + 3  # [3, 4]
            rotation_idx = -env.observation_space.shape[0] + 5  # [5, 6]
            warship_obs_shape = 8
            value = {
                "hp": [obs[-warship_obs_shape] for obs in observations],
                "position": [obs[position_idx:position_idx+2] for obs in observations],
                "rotation": [obs[rotation_idx:rotation_idx+2] for obs in observations],
                "opponent": {
                    "hp": [observations[0][(i+3)*warship_obs_shape] for i in range(len(observations))],
                    "position": [observations[0][(i+3)*warship_obs_shape+3:(i+3)*warship_obs_shape+5]
                                 for i in range(len(observations))],
                    "rotation": [observations[0][(i+3)*warship_obs_shape+5:(i+3)*warship_obs_shape+7]
                                 for i in range(len(observations))]
                },
                "action": actions,
                "reward": rewards
            }
            # print(f'[main] value: {value}')
            _ = ref.put(**value)

            # observations = next_observations
            observations = next_observationss
            h_ins = h_outs

            if done:
                # TODO: MongoDB Logging
                episode_wins.append(info.get('win', -1) == 0)
                episode_loses.append(info.get('win', -1) == 1)
                episode_draws.append(info.get('win', -1) == -1)

                result_db.put(**{
                    "session": session_id,
                    "episode": episode_id,
                    "result": info.get('win', -1)
                })

                # TODO: reward
                experiences = episode_buffer.items
                episode_buffer.clear()

                discounted_rewards = np.array([exp[3] for exp in experiences]).transpose()
                dones = [exp[-1] for exp in experiences]
                for i in range(discounted_rewards.shape[0]):
                    # discounted_rewards[i][:-1] = 0
                    discounted_rewards[i] = discount_with_dones(discounted_rewards[i], dones, gamma=0.996)  # 172
                discounted_rewards = np.transpose(discounted_rewards)
                for i in range(discounted_rewards.shape[0]):
                    experiences[i][REWARD_IDX][:] = discounted_rewards[i]
                # buffer.extend(experiences)

                train_losses = []
                actor_losses = []
                critic_losses = []

                for agent in agents:
                    other_agents = agents.copy()
                    other_agents.remove(agent)
                    total_loss, actor_loss, critic_loss = agent.learn(experiences, other_agents)
                    train_losses.append(total_loss)
                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)

                # Tensorboard
                try:
                    writer.add_scalar('loss/sum', np.sum(train_losses), episode)
                    writer.add_scalar('loss/mean', np.mean(train_losses), episode)
                    writer.add_scalar('loss/actor', np.mean(actor_losses), episode)
                    writer.add_scalar('loss/critic', np.mean(critic_losses), episode)

                    hps = [next_obs[0] for next_obs in next_observations]
                    writer.add_scalar('performance/hp', np.mean(hps), episode)
                    writer.add_scalar('performance/reward', np.sum(episode_rewards), episode)
                except:
                    sys.stderr.write(f'[{datetime.now().isoformat()}] [MAIN/TENSORBOARD] FAILED TO LOG TENSORBOARD!\n')

                print(f'[{datetime.now().isoformat()}] Episode #{episode} Loss={np.sum(train_losses)}')

                break

        """
        total_loss = None
        if len(buffer) > args.batch_size:
            train_losses = []
            actor_losses = []
            critic_losses = []
            for agent in agents:
                other_agents = agents.copy()
                other_agents.remove(agent)
                batch = buffer.sample(args.batch_size)
                total_loss, actor_loss, critic_loss = agent.learn(batch, other_agents)
                train_losses.append(total_loss)
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
            #print(f'Episode #{episode}: Loss={np.mean(train_losses)}')
            total_loss = np.mean(train_losses)
            # Tensorboard
            try:
                writer.add_scalar('loss/sum', np.sum(train_losses), episode)
                writer.add_scalar('loss/mean', np.mean(train_losses), episode)
                writer.add_scalar('loss/actor', np.mean(actor_losses), episode)
                writer.add_scalar('loss/critic', np.mean(critic_losses), episode)
            except:
                sys.stderr.write(f'[{datetime.now().isoformat()}] [MAIN/TENSORBOARD] FAILED TO LOG LOSS!\n')

        print(f'[{datetime.now().isoformat()}] Episode #{episode} Loss={total_loss} (buffer={len(buffer)}/{args.batch_size})')
        """

        if episode % 100 == 0:
            print(f'Episode #{episode} :: WinRate={np.mean(episode_wins)}')
            # plotly
            ep_wins = [np.sum(episode_wins[i*100:(i+1)*100]) for i in range(episode//100)][-10:]
            ep_draws = [np.sum(episode_draws[i*100:(i+1)*100]) for i in range(episode//100)][-10:]
            ep_loses = [np.sum(episode_loses[i*100:(i+1)*100]) for i in range(episode//100)][-10:]
            data = [ep_wins, ep_draws, ep_loses]
            try:
                plotboard.plot(tuple(map(str, range(1, episode//100+1)[-10:])), data)
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

            # save model
            checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoints', exprmt_id)
            if not os.path.exists(checkpoint_path):
                pathlib.Path(os.path.abspath(checkpoint_path)).mkdir(parents=True, exist_ok=True)
            for i, agent in enumerate(agents):
                agent.save(os.path.join(checkpoint_path, f'maddpg{i}-ep{episode}.ckpt'))

    env.close()


if __name__ == "__main__":
    main()
