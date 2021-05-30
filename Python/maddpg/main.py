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
# from plotboard import WinRateBoard

REWARD_IDX = 3


parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='NavOpsMultiDiscrete-v2')
parser.add_argument('--no-graphics', action='store_true', default=False)
parser.add_argument('--batch-size', type=int, default=1024)
parser.add_argument('--buffer-size', type=int, default=40000)
parser.add_argument('--n', type=int, default=3)
parser.add_argument('--worker-id', type=int, default=0)
# parser.add_argument('--time-horizon', type=int, default=2048)
parser.add_argument('--sequence-length', type=int, default=32)
parser.add_argument('--actor-learning-rate', type=float, default=3e-4)
parser.add_argument('--critic-learning-rate', type=float, default=1e-3)
parser.add_argument('--no-logging', action='store_true', default=False)
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
            env.action_space.nvec,
            agent_id=i,
            n=args.n,
            actor_learning_rate=args.actor_learning_rate,
            critic_learning_rate=args.critic_learning_rate
        )
        for i in range(args.n)
    ]
    # buffer = ReplayBuffer(capacity=50000)
    episode_buffer = ReplayBuffer(capacity=args.buffer_size)
    episode_wins = []
    episode_loses = []
    episode_draws = []

    initial_episode = 1

    if not args.no_logging:
        exprmt_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    # generate_id()
        writer = SummaryWriter(f'runs/{exprmt_id}')
        # plotboard = WinRateBoard(dirpath=os.path.join(os.path.dirname(__file__), 'plots', exprmt_id))

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

        if not args.no_logging:
            episode_id = ('0' * 10 + str(episode))[-10:]
            ref = session_db.ref(episode_id)

        for step in count(1):
            actions = []
            h_outs = []
            for agent, observation, h_in in zip(agents, observations, h_ins):
                action_m, action_a, h_out = agent.select_action(observation, h_in)
                actions.append([action_m, action_a])
                h_outs.append(h_out)
            actions = np.array(actions)

            next_observations, rewards, done, info = env.step(actions)

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

            if not args.no_logging:
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
                episode_wins.append(info.get('win', -1) == 0)
                episode_loses.append(info.get('win', -1) == 1)
                episode_draws.append(info.get('win', -1) == -1)

                # TODO: reward
                experiences = episode_buffer.items
                episode_buffer.clear()

                discounted_rewards = np.array([exp[3] for exp in experiences]).transpose()
                print(f'[{datetime.now().isoformat()}] Episode #{episode}')
                np.set_printoptions(suppress=True)
                print(discounted_rewards)
                # env.close()
                # sys.exit(0)

                """
                [2021-05-09T20:01:41.594224] NavOpsEnv.Reset() => behavior_names: ['RimpacBehavior?team=1']
                [gym-navops] win: 1 (Episode Rewards: [-1. -1. -1.])
                [2021-05-09T20:02:50.336122] Episode #1
                [[-0.00017672 -0.00017377 -0.00017086 -0.00016819 -0.00016581 -0.0001633
                -0.00016068 -0.00015809 -0.00015551 -0.00015296 -0.00014994 -0.00014653
                -0.00014242 -0.00013783 -0.0001332  -0.0001286  -0.00012399 -0.00011949
                -0.00011502 -0.00011066 -0.00010635 -0.0001022  -0.00009809 -0.00009408
                -0.00009019 -0.00008649 -0.00008289 -0.00007946 -0.00007619 -0.00007308
                -0.00007017 -0.00006745 -0.00006514 -0.00006314 -0.0000612  -0.00005937
                -0.00005762 -0.00005598 -0.00005444 -0.00005297 -0.0000516  -0.00005032
                -0.00004911 -0.00004799 -0.00004693 -0.00004596 -0.00004507 -0.00004429
                -0.00004362 -0.00004303 -0.00004245 -0.0000419  -0.00004145 -0.00004109
                -0.00004084 -0.00004068 -0.00004053 -0.00004037 -0.00004029 -0.00004026
                -0.00004013 -0.0000399  -0.00003955 -0.0000391  -0.00003854 -0.00003787
                -0.00003712 -0.00003627 -0.00003534 -0.00003437 -0.00003339 -0.00003241
                0.00131809 -0.00003036 -0.00002933 -0.00002827 -0.00002717 -0.00002607
                -0.00002512 -0.00002431 -0.00002358 -0.00002276 -0.00002196  0.01773297
                -0.00002065 -0.00002026 -0.00001978 -0.00001923 -0.00001863 -0.00001795
                -0.00001726 -0.0000166  -0.00001601 -0.00001554 -0.00001523 -0.00001505
                -0.00001483 -0.00001462 -0.0000144  -0.00001416 -0.00001392 -0.00001368
                -0.00001347 -0.00001323 -0.00001304 -0.0000128  -0.00001257 -0.00001233
                -0.00001203 -0.00001166 -0.00001128 -0.00001088 -0.00001048 -0.0000101
                -0.0000097  -0.00000933 -0.00000896  0.          0.          0.
                0.          0.          0.          0.          0.          0.
                0.          0.          0.          0.          0.          0.
                0.          0.          0.          0.          0.          0.
                0.          0.          0.          0.          0.          0.
                0.          0.          0.          0.          0.          0.
                0.          0.          0.          0.          0.          0.
                0.          0.          0.          0.          0.          0.
                0.          0.          0.          0.          0.          0.
                0.          0.          0.         -1.        ]
                [-0.00012676 -0.00012421 -0.00012104 -0.00011705 -0.00011258 -0.00010814
                -0.0001038  -0.00009958 -0.00009541 -0.00009135 -0.00008739 -0.00008352
                -0.00007972 -0.00007602 -0.0000724  -0.00006887 -0.00006541 -0.00006207
                -0.0000588  -0.00005564 -0.00005255 -0.00004959 -0.00004666 -0.00004382
                -0.00004107 -0.00003843 -0.00003585 -0.00003338 -0.00003098 -0.00002867
                -0.00002647 -0.00002436 -0.00002232 -0.00002037 -0.00001851 -0.00001678
                -0.00001513 -0.0000136  -0.00001217 -0.0000108  -0.00000953 -0.00000835
                -0.00000725 -0.00000624 -0.00000533 -0.0000045  -0.00000376 -0.00000313
                -0.00000261 -0.00000218 -0.00000181 -0.00000152 -0.00000132 -0.00000122
                -0.00000121 -0.00000128 -0.00000142 -0.00000163 -0.00000191 -0.00000227
                -0.00000268 -0.00000295 -0.00000245 -0.00000201 -0.00000162 -0.00000129
                -0.00000102 -0.00000078  0.          0.          0.          0.
                0.          0.          0.          0.          0.          0.
                0.          0.          0.          0.          0.          0.
                0.          0.          0.          0.          0.          0.
                0.          0.          0.          0.          0.          0.
                0.          0.          0.          0.          0.          0.
                0.          0.          0.          0.          0.          0.
                0.          0.          0.          0.          0.          0.
                0.          0.          0.          0.          0.          0.
                0.          0.          0.          0.          0.          0.
                0.          0.          0.          0.          0.          0.
                0.          0.          0.          0.          0.          0.
                0.          0.          0.          0.          0.          0.
                0.          0.          0.          0.          0.          0.
                0.          0.          0.          0.          0.          0.
                0.          0.          0.          0.          0.          0.
                0.          0.          0.          0.          0.          0.
                0.          0.          0.         -1.        ]
                [-0.00017672 -0.00017419 -0.00017197 -0.00016945 -0.0001668  -0.00016345
                -0.00016041 -0.00015783 -0.00015551 -0.00015341 -0.00015135 -0.0001493
                -0.00012402 -0.00014449 -0.00014203 -0.00013961 -0.00013718 -0.00013481
                -0.00013244 -0.00013011 -0.00012779 -0.00012552 -0.00012324 -0.00012097
                -0.00011872 -0.00011652 -0.00011432 -0.00011215 -0.00010999 -0.00010785
                -0.00010576 -0.00010369 -0.00010162 -0.0000996  -0.00009765 -0.00009582
                -0.00009407 -0.00009246 -0.00009092 -0.00008938 -0.00008786 -0.0000864
                -0.00008494 -0.00008354 -0.00008215 -0.00008083 -0.00007959 -0.0000785
                -0.00007756 -0.0000767  -0.0000758  -0.00007491 -0.00007417 -0.00007358
                -0.00007316 -0.0000729  -0.00007265 -0.00007239 -0.00007229 -0.00007228
                -0.00007217 -0.00007194 -0.00007159 -0.00007112 -0.00007053 -0.00006982
                -0.00006901 -0.0000681  -0.00006709 -0.00006606 -0.00006512 -0.00006428
                -0.00006342 -0.00006267 -0.00006195 -0.00006127 -0.00006032 -0.00005915
                -0.00005795 -0.00005673 -0.00005551 -0.00005428 -0.00005305 -0.0000518
                -0.00005055 -0.00004933 -0.00004808 -0.00004686 -0.00004565 -0.00004446
                -0.00004331 -0.00004217 -0.00004107 -0.00003999 -0.00003893 -0.00003792
                -0.00003691 -0.00003586 -0.00003479 -0.00003368 -0.0000326  -0.00003156
                -0.00003057 -0.00002961 -0.00002869 -0.00002781 -0.00002699 -0.0000262
                -0.00002544 -0.00002474 -0.00002407 -0.00002345 -0.00002287 -0.00002233
                -0.0000218  -0.00002129 -0.00002081 -0.00002034 -0.00001984 -0.00001929
                -0.00001871 -0.00001809 -0.00001746 -0.00001685 -0.00001626 -0.00001567
                -0.00001508 -0.00001451 -0.00001398 -0.00001343 -0.00001289 -0.00001237
                -0.00001185 -0.00001131 -0.00001075 -0.00001018 -0.00000961 -0.00000903
                -0.00000845 -0.00000788 -0.00000732 -0.00000677 -0.00000623 -0.00000572
                -0.00000523 -0.00000476 -0.00000433 -0.00000392 -0.00000356 -0.00000323
                -0.00000294 -0.0000027  -0.0000025  -0.00000234 -0.00000223 -0.00000217
                -0.00000213 -0.00000216 -0.00000225 -0.00000237 -0.00000254 -0.00000276
                -0.00000301 -0.00000331 -0.00000365 -0.00000402 -0.00000441 -0.00000482
                -0.00000525 -0.00000571 -0.0000062  -1.        ]]
                """

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
                if not args.no_logging:
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

                    result_db.put(**{
                        "session": session_id,
                        "episode": episode_id,
                        "result": info.get('win', -1),
                        "loss": np.sum(train_losses),
                        "hp": np.mean(hps),
                        "reward": np.sum(episode_rewards)
                    })

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
            """
            try:
                plotboard.plot(tuple(map(str, range(1, episode//100+1)[-10:])), data)
                plotboard.plot(data)
            except:
                sys.stderr.write(f'[{datetime.now().isoformat()}] [MAIN/PLOTLY] FAILED TO PLOT!\n')
            """
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
