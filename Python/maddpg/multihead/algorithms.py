#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.distributions import Categorical

from models import MultiHeadActorCritic
from utils import convert_to_tensor


class MADDPG:
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=512,
        rnn_hidden_size=512,
        learning_rate=3e-4,
        gamma=0.998,
        tau=0.01,
        agent_id=0,
        n=3
    ):
        self._gamma = gamma
        self._tau = tau
        self._agent_id = agent_id
        self._n = n

        self._device = torch.device("cpu")
        # self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = MultiHeadActorCritic(input_size, output_size, hidden_size=hidden_size, rnn_hidden_size=rnn_hidden_size).to(self.device)
        self.target = MultiHeadActorCritic(input_size, output_size, hidden_size=hidden_size, rnn_hidden_size=rnn_hidden_size).to(self.device)

        self.target.load_state_dict(self.model.state_dict())

        self.optim = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

        self.train_step = 0

    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.target.parameters(), self.model.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)

    def train(self, transitions, other_agents):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device) # in-place

        for agent in other_agents:
            agent.policy.to(device)

        batch_size = len(transitions)

        states, actions, next_states, rewards, h_ins, dones = [], [], [], [], [], []

        for s, a, s_, r, h_in, d in transitions:
            states.append(s)
            actions.append(a)
            next_states.append(s_)
            rewards.append(r)
            h_ins.append(h_in)
            dones.append(d)

        states = convert_to_tensor(device, states).squeeze()
        actions = convert_to_tensor(device, actions).squeeze()
        next_states = convert_to_tensor(device, next_states).squeeze()
        rewards = convert_to_tensor(device, rewards).squeeze()
        h_ins = [
            torch.stack([h_in[0] for h_in in h_ins]).squeeze().to(device)
            for _ in range(self.n)
        ]
        dones = convert_to_tensor(device, dones)

        r = rewards[:, self.agent_id]

        # calculate the target Q value function
        u_next = []
        with torch.no_grad():
            index = 0

        """#"""

        # calculate the target Q value function
        u_next = []
        with torch.no_grad():
            index = 0
            for agent_id in range(len(other_agents)+1):
                if agent_id == self.agent_id:
                    _, h_out_ = self.actor_target_network(states[:, agent_id], h_ins[agent_id])
                    next_h_ins = torch.cat((h_ins[agent_id][1:], h_out_[-1:]))
                    action_, _ = self.actor_target_network(next_states[:, agent_id], next_h_ins)
                    action_ = torch.argmax(action_, dim=-1)
                    u_next.append(action_.cpu().numpy())
                else:
                    # FIXME: other h_in
                    _, h_out_ = other_agents[index].policy.actor_target_network(states[:, agent_id], h_ins[agent_id])
                    next_h_ins_ = torch.cat((h_ins[agent_id][1:], h_out_[-1:]))
                    action_, _ = other_agents[index].policy.actor_target_network(next_states[:, agent_id], next_h_ins_)
                    action_ = torch.argmax(action_, dim=-1)
                    u_next.append(action_.cpu().numpy())
                    index += 1
            u_next = np.array(u_next, dtype=np.uint8)
            u_next = torch.from_numpy(u_next).to(device)
            q_next, _ = self.critic_target_network(next_states, u_next, next_h_ins)
            q_next = q_next.detach()

            target_q = (r.unsqueeze(1) + self.gamma * q_next).detach()

        # the q loss
        critic_h_in = self.critic_network.reset_hidden_state(batch_size=batch_size)
        q_value, _ = self.critic_network(states, torch.transpose(actions, dim0=0, dim1=1), critic_h_in)
        critic_loss = (target_q - q_value).pow(2).mean()

        # the actor loss
        probs, _ = self.actor_network(states[:, self.agent_id], h_ins[self.agent_id])
        asmpl = Categorical(probs).sample()
        actions = torch.transpose(actions, dim0=0, dim1=1)
        actions[self.agent_id] = asmpl
        # actions[: self.agent_id] = asmpl
        # actions[self.agent_id] = Categorical(probs).sample()
        #actor_loss = -self.critic_network(states, actions, critic_h_in).mean()
        actor_loss, _ = self.critic_network(states, actions, critic_h_in)
        actor_loss = -actor_loss.mean()
        #actor_loss = -self.critic_network(states, torch.transpose(actions, dim0=0, dim1=1)).mean()
        # if self.agent_id == 0:
        #     print('critic_loss is {}, actor_loss is {}'.format(critic_loss, actor_loss))
        # update the network
        actor_loss_ = actor_loss.cpu().item()
        critic_loss_ = critic_loss.cpu().item()
        total_loss = actor_loss_ + critic_loss_

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

        self.to(self.device)
        for agent in other_agents:
            agent.policy.to(agent.policy.device)

        return total_loss, actor_loss_, critic_loss_

    def save(self, path):
        torch.save({
            "state_dict": self.model.state_dict()
        }, path)

    def to(self, device):
        self.model = self.model.to(device)
        self.target_network = self.target_network.to(device)
        self._device = device

    def reset_hidden_state(self, batch_size=8):
        return self.model.reset_hidden_state(batch_size=batch_size)

    @property
    def device(self):
        return self._device

    @property
    def gamma(self):
        return self._gamma

    @property
    def tau(self):
        return self._tau

    @property
    def agent_id(self):
        return self._agent_id

    @property
    def n(self):
        return self._n


def main():
    pass


if __name__ == "__main__":
    main()
