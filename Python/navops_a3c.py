#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import argparse
import threading
import queue
import time
from collections import deque
from datetime import datetime
from itertools import count

import numpy as np
import torch.multiprocessing as mp
import gym
import gym_navops   # noqa: F401
from gym_navops import EnvironmentConfig
from torch.utils.tensorboard import SummaryWriter

from models.pytorch_impl import MultiHeadLSTMActorCriticAgent
from utils import get_free_port, generate_id
from utils.board import ReportingBoard
from utils.database import MongoDatabase

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='NavOpsMultiDiscrete-v0')
parser.add_argument('--n', type=int, default=4)
parser.add_argument('-p', '--port', type=int, default=9090)
parser.add_argument('--gamma', type=float, default=0.98)
parser.add_argument('--target-update-period', type=int, default=4)
parser.add_argument('--time-horizon', type=int, default=32)
parser.add_argument('--no-logging', type=bool, default=False)
args = parser.parse_args()

ENVPATH = os.path.join('C:\\', 'Users', 'rapsealk', 'Desktop', 'NavOps', 'NavOps.exe')
# GLOBAL_LOCK = mp.Lock()
# GLOBAL_LOCK = threading.Lock()
GLOBAL_UPDATE_INTERVAL = 4
GLOBAL_EPISODE = 0

PROCESS_CPU_COUNT = args.n or mp.cpu_count()


def discount_rewards(rewards, gamma=0.98):
    rewards_ = np.zeros_like(rewards)
    rewards_[0] = rewards[-1]
    for i in range(1, len(rewards)):
        rewards_[i] = rewards_[i-1] * gamma + rewards[-i-1]
    return rewards_[::-1]


def td_discount_rewards(rewards, gamma=0.98, time_horizon=args.time_horizon):
    rewards_ = np.zeros_like(rewards)
    i = 0
    while i < len(rewards):
        tmp = list(reversed(rewards[i:i+time_horizon]))
        for j in range(1, len(tmp)):
            tmp[j] += gamma * tmp[j-1]
        rewards_[i:i+len(tmp)] = list(reversed(tmp))
        i += len(tmp)
    return rewards_


def gae():
    pass


class Trainer:

    def __init__(self):
        self.supported_ports = [get_free_port(args.port+i) for i in range(PROCESS_CPU_COUNT)]
        input_size = EnvironmentConfig["NavOpsMultiDiscrete"]["observation_space"]["shape"][0]
        output_sizes = EnvironmentConfig["NavOpsMultiDiscrete"]["action_space"]["nvec"]

        self.agent = MultiHeadLSTMActorCriticAgent(input_size,
                                                output_sizes,
                                                hidden_size=512,
                                                rnn_hidden_size=64,
                                                rnn_num_layers=1,
                                                learning_rate=3e-5,
                                                cuda=True)
        self.agent.share_memory()
        # self.event_queue = mp.Queue()
        self.event_queue = queue.Queue()
        self.lock = threading.Lock()

        self._id = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}-{args.env}'
        self.session_id = generate_id()
        if not args.no_logging:
            self._writer = SummaryWriter(f'runs/{self._id}')
            self._plotly = ReportingBoard()
            # database = MongoDatabase()
            # self._result_db = database.ref("result")
            # self._session_db = database.ref(self.session_id)
            # self._loss_db = self._session_db.ref("loss")

    def start(self, n=PROCESS_CPU_COUNT):
        processes = [Worker(self.agent, self.event_queue, self.lock, f'Worker-{i}', self.supported_ports[i])
                     for i in range(n)]
        for process in processes:
            process.start()

        queue_thread = threading.Thread(target=self.digest_queue, daemon=True)
        queue_thread.start()

        for process in processes:
            process.join()
        # for process in processes:
        #     process.close()

    def digest_queue(self):
        episode = 0
        results = [deque(maxlen=10) for _ in range(3)]
        results_total = [[] for _ in range(3)]
        tmp_results = [[] for _ in range(3)]
        time_ = time.time()
        while True:
            try:
                dict_ = self.event_queue.get_nowait()
            except queue.Empty:
                time.sleep(1.0)
                continue
            episode += 1
            t_ = time.time() - time_
            flag = 'W' if (dict_["Win"] > 0) else 'L' if (dict_["Win"] < 0) else 'D'
            print(f'[{datetime.now().isoformat()}] ({int(t_//3600):02d}h {int(t_%3600//60):02d}m {(t_%3600)%60:02.2f}s) Episode:{episode} -> {flag} Reward: {dict_["Reward"]["Mean"]:.3f} Loss={dict_["Loss"]:.3f}')
            tmp_results[0].append(dict_["Win"] > 0)
            tmp_results[1].append(dict_["Win"] == 0)
            tmp_results[2].append(dict_["Win"] < 0)

            if not args.no_logging:
                self._writer.add_scalar('loss', dict_["Loss"], episode)
                self._writer.add_scalar('performance/prob/maneuver', dict_["Performance"]["Prob"]["Maneuver"], episode)
                self._writer.add_scalar('performance/prob/attack', dict_["Performance"]["Prob"]["Attack"], episode)
                self._writer.add_scalar('rewards/sum', dict_["Reward"]["Sum"], episode)
                self._writer.add_scalar('rewards/mean', dict_["Reward"]["Mean"], episode)

            if episode % 100 == 0:
                for _ in range(3):
                    results[_].append(np.sum(tmp_results[_][-100:]))
                    results_total[_].append(int(np.sum(tmp_results[_]) / episode * 100))
                if not args.no_logging:
                    self._plotly.plot_winning_rate(*results)
                    self._plotly.plot_winning_rate(*results_total)

    """
    def train(self, n=4, base_port=9090):
        ports = [get_free_port(base_port+i) for i in range(n)]
        processes = [mp.Process(target=self._spawn_env, args=(i, ports[i])) for i in range(n)]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
        for process in processes:
            process.close()

    def _spawn_env(self, worker_id=0, port=9090):
        try:
            env = gym.make(args.env, build_path=ENVPATH, port=port)
        except Exception as e:
            print('Exception:', e)
            return
        done = False

        while not done:
            s, r, done, info = env.step(env.action_space.sample())

        print(f'Worker Id: {worker_id} finished!')
    """


# class Worker(mp.Process):
class Worker(threading.Thread):

    def __init__(self, global_agent, queue, lock, name, port):
        super(Worker, self).__init__()
        self.name = name

        self.global_agent = global_agent
        self.env = gym.make(args.env, build_path=ENVPATH, port=port)
        self.agent = MultiHeadLSTMActorCriticAgent(self.env.observation_space.shape[0],
                                                   self.env.action_space.nvec,
                                                   hidden_size=512,
                                                   rnn_hidden_size=64,
                                                   rnn_num_layers=1,
                                                   learning_rate=3e-5,
                                                   cuda=False)
        self.queue = queue
        self.lock = lock

    def run(self):
        for episode in count(1):
            observations = []
            # hidden_ins = []
            rewards = []
            actions = []
            probs = {"maneuver": [], "attack": []}

            done = False
            obs = self.env.reset()
            h_in = self.agent.reset_hidden_state(batch_size=1)

            for t in count(1):
                (action_m, prob_m), (action_a, prob_a), h_out = self.agent.get_action(obs, h_in)
                probs["maneuver"].append(np.max(prob_m.detach().cpu().squeeze().numpy(), axis=-1))
                probs["attack"].append(np.max(prob_a.detach().cpu().squeeze().numpy(), axis=-1))
                # action = self.env.action_space.sample()
                action = [action_m, action_a]
                obs, reward, done, info = self.env.step(action)
                # if abs(reward) == 1.0 and done:
                #     reward *= 10.0

                observations.append(obs)
                actions.append(action)
                rewards.append(reward)
                # hidden_ins.append(h_in)

                h_in = h_out

                if done:
                    rewards_ = td_discount_rewards(rewards, gamma=args.gamma)
                    # print(f'[Reward] {self.name}\nRaw: {rewards}\nDiscounted: {rewards_}\nSum: {np.sum(rewards)} -> {np.sum(rewards_)}\nMean: {np.mean(rewards)} -> {np.mean(rewards_)}')
                    observations = np.array(observations)
                    actions = np.array(actions)
                    rewards = np.array(rewards_)
                    # hidden_ins = np.array(hidden_ins)

                    # with GLOBAL_LOCK:
                    with self.lock:
                        device_ = self.agent.device
                        self.agent = self.agent.to(self.global_agent.device)
                        print(f'[Main] {self.name}: Device is {device_} -> {self.agent.device}')

                        loss = self.agent.loss(observations, actions, rewards)
                        self.global_agent.apply_gradients(self.agent, loss)
                        self.global_agent.soft_update(tau=1e-2)

                        if episode % GLOBAL_UPDATE_INTERVAL == 0:
                            # self.agent.set_state_dict(self.global_agent.state_dict())
                            self.agent.model.load_state_dict(self.global_agent.model.state_dict())
                            self.agent.target_model.load_state_dict(self.global_agent.target_model.state_dict())

                        self.agent = self.agent.to(device_)

                        global GLOBAL_EPISODE
                        if (GLOBAL_EPISODE := GLOBAL_EPISODE + 1) % 100 == 0:
                            self.global_agent.save(
                                path=os.path.join(os.path.dirname(__file__), 'checkpoints', f'{args.env}-a3c-{GLOBAL_EPISODE}.ckpt'),
                                episode=GLOBAL_EPISODE)

                    self.queue.put({
                        # "Episode": episode,
                        "Loss": loss.item(),
                        "Reward": {"Sum": np.sum(rewards), "Mean": np.mean(rewards)},
                        "Performance": {"Prob": {"Maneuver": np.mean(probs["maneuver"]), "Attack": np.mean(probs["attack"])}},
                        "Win": reward
                    })

                    # self.agent.soft_update(tau=1e-2)
                    # if episode % args.target_update_period == 0:
                    #     self.agent.hard_update()
                    break


class Tester:
    pass


def main():
    Trainer().start()


if __name__ == "__main__":
    main()
    # trainer = Trainer()
    # trainer.train(args.n, base_port=args.port)
