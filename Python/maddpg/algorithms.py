#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.distributions import Categorical

from models import Actor, Critic


def convert_to_tensor(device, *args):
    # return map(lambda tensor: tensor.to(device), map(torch.FloatTensor, args))
    return map(lambda tensor: tensor.float().to(device), map(torch.tensor,
        map(lambda arr: np.asarray(arr, dtype=np.float32), args)
    ))


class MADDPG:
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=64,
        actor_learning_rate=3e-4,
        critic_learning_rate=1e-3,
        gamma=0.95,
        tau=0.01,
        agent_id=0,
        n=3
    ):
        self._gamma = gamma
        self._tau = tau
        self._agent_id = agent_id

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # create the network
        self.actor_network = Actor(input_size, output_size, hidden_size=hidden_size).to(self._device)
        self.critic_network = Critic((input_size+output_size)*n, output_size, hidden_size=hidden_size).to(self._device)

        # build up the target network
        self.actor_target_network = Actor(input_size, output_size, hidden_size=hidden_size).to(self._device)
        self.critic_target_network = Critic((input_size+output_size)*n, output_size, hidden_size=hidden_size).to(self._device)

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=actor_learning_rate)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=critic_learning_rate)

        """
        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = self.model_path + '/' + 'agent_%d' % agent_id
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        # 加载模型
        if os.path.exists(self.model_path + '/actor_params.pkl'):
            self.actor_network.load_state_dict(torch.load(self.model_path + '/actor_params.pkl'))
            self.critic_network.load_state_dict(torch.load(self.model_path + '/critic_params.pkl'))
            print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,
                                                                          self.model_path + '/actor_params.pkl'))
            print('Agent {} successfully loaded critic_network: {}'.format(self.agent_id,
                                                                           self.model_path + '/critic_params.pkl'))
        """

        self.train_step = 0

    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)

    # update the network
    def train(self, transitions, other_agents):
        states, actions, next_states, rewards, dones = [], [], [], [], []
        for s, a, s_, r, d in transitions:
            states.append(s)
            actions.append(a)
            next_states.append(s_)
            rewards.append(r)
            dones.append(d)

        # print(f'[MADDPG] np.states: {states[0].shape} {states[-1].shape}')
        print(f'[MADDPG] np.actions: {actions[0].shape} {actions[-1].shape}')
        # print(f'[MADDPG] np.next_states: {next_states[0].shape} {next_states[-1].shape}')
        # print(f'[MADDPG] np.rewards: {rewards[0].shape} {rewards[-1].shape}')
        # print(f'[MADDPG] np.dones: {dones[0]} {dones[-1]}')

        states, actions, next_states, rewards, dones = convert_to_tensor(self._device, states, actions, next_states, rewards, dones)

        # print(f'[MADDPG] train.states: {states.shape}')
        print(f'[MADDPG] train.actions: {actions.shape}')
        # print(f'[MADDPG] train.next_states: {next_states.shape}')
        # print(f'[MADDPG] train.rewards: {rewards.shape}')
        # print(f'[MADDPG] train.dones: {dones.shape}')

        """
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)
        r = transitions[f'r_{self.agent_id}']
        o, u, o_next = [], [], []  # 用来装每个agent经验中的各项
        for agent_id in range(self.args.n_agents):
            o.append(transitions[f'o_{agent_id}'])
            u.append(transitions[f'u_{agent_id}'])
            o_next.append(transitions[f'o_next_{agent_id}'])
        """

        r = rewards[:, self.agent_id]
        o = states#[:, self.agent_id]
        u = actions#[:, self.agent_id]
        o_next = next_states#[:, self.agent_id]

        # calculate the target Q value function
        u_next = []
        with torch.no_grad():
            index = 0
            for agent_id in range(len(other_agents)+1):
                if agent_id == self.agent_id:
                    # u_next.append(self.actor_target_network(o_next[agent_id]))
                    action_ = self.actor_target_network(o_next[:, agent_id])
                    print(f'[MADDPG] action_: {action_.shape}')
                    action_ = torch.argmax(action_, dim=-1)
                    u_next.append(action_.numpy())
                else:
                    # u_next.append(other_agents[index].policy.actor_target_network(o_next[agent_id]))
                    action_ = other_agents[index].policy.actor_target_network(o_next[:, agent_id])
                    print(f'[MADDPG] action_: {action_.shape}')
                    action_ = torch.argmax(action_, dim=-1)
                    u_next.append(action_.numpy())
                    index += 1
            # print(f'[MADDPG] o_next: {o_next}')
            print(f'[MADDPG] u_next: {u_next}')
            u_next = np.asarray(u_next, dtype=np.uint8)
            u_next = torch.from_numpy(u_next)
            q_next = self.critic_target_network(o_next, u_next).detach()

            target_q = (r.unsqueeze(1) + self.gamma * q_next).detach()

        # the q loss
        # q_value = self.critic_network(o, u)
        q_value = self.critic_network(o, torch.transpose(u, dim0=0, dim1=1))
        critic_loss = (target_q - q_value).pow(2).mean()

        # the actor loss
        # 重新选择联合动作中当前agent的动作，其他agent的动作不变
        print(f'[MADDPG] u.shape: {u.shape}')
        ap = self.actor_network(o[self.agent_id])
        print(f'[MADDPG] ap.shape: {ap.shape}')
        ap = Categorical(ap).sample()
        print(f'[MADDPG] a.shape: {ap.shape}')
        u[self.agent_id] = ap
        # u[self.agent_id] = self.actor_network(o[self.agent_id])
        # actor_loss = - self.critic_network(o, u).mean()
        actor_loss = -self.critic_network(o, torch.transpose(u, dim0=0, dim1=1)).mean()
        # if self.agent_id == 0:
        #     print('critic_loss is {}, actor_loss is {}'.format(critic_loss, actor_loss))
        # update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self._soft_update_target_network()
        """
        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step)
        """
        self.train_step += 1

    def save_model(self, train_step):
        """
        num = str(train_step // self.args.save_rate)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, f'agent_{self.agent_id}')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.actor_network.state_dict(), model_path + '/' + num + '_actor_params.pkl')
        torch.save(self.critic_network.state_dict(),  model_path + '/' + num + '_critic_params.pkl')
        """
        return

    @property
    def gamma(self):
        return self._gamma

    @property
    def tau(self):
        return self._tau

    @property
    def agent_id(self):
        return self._agent_id


def main():
    pass


if __name__ == "__main__":
    main()
