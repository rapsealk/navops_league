#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import sys
import argparse
from datetime import datetime
from itertools import count

import numpy as np
from torch.utils.tensorboard import SummaryWriter

import gym
import gym_navops
from algorithms import AttentionSoftActorCritic
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import generate_id
from plotboard import WinRateBoard

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='NavOpsMultiDiscrete-v2')
parser.add_argument('--no-graphics', action='store_true', type=bool, default=False)
parser.add_argument('--batch-size', type=int, default=1024)
parser.add_argument('--actor-hidden-dim', type=int, default=128)
parser.add_argument('--critic-hidden-dim', type=int, default=128)
parser.add_argument('--n', type=int, default=3)
parser.add_argument('--worker-id', type=int, default=0)
parser.add_argument('--tau', type=float, default=1e-3)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--reward-scale', type=float, default=1.0)
parser.add_argument('--actor-learning-rate', type=float, default=1e-3)
parser.add_argument('--critic-learning-rate', type=float, default=1e-3)
args = parser.parse_args()

"""
def make_parallel_env(env, n_rollout_threads, seed):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=True)
"""


def main():
    build_path = os.path.join('C:\\', 'Users', 'rapsealk', 'Desktop', 'NavOps')
    env = gym.make(args.env, no_graphics=args.no_graphics, worker_id=args.worker_id, override_path=build_path)
    print(f'[navops_league] obs: {env.observation_space.shape}, action: {sum(env.action_space.nvec)}')

    agent = AttentionSoftActorCritic.init_from_env(
        env,
        tau=args.tau,
        gamma=args.gamma,
        pi_lr=args.actor_learning_rate,
        q_lr=args.critic_learning_rate,
        reward_scale=args.reward_scale,
        pol_hidden_dim=args.actor_hidden_dim,
        critic_hidden_dim=args.critic_hidden_dim,
        attend_heads=args.attend_heads)


if __name__ == "__main__":
    main()
