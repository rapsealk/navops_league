#!/usr/bin/python3
# -*- coding: utf-8 -*-
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def onehot_from_logits(logits, eps=0.0, dim=1):
    argmax_actions = (logits == logits.max(dim, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_actions
    random_actions = Variable(torch.eye(logits.shape[1])[[np.random.choice(
        range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
    return torch.stack([argmax_actions[i] if r > eps else random_actions[i]
                        for i, r in enumerate(torch.rand(logits.shape[0]))])


def categorical_sample(probs, use_cuda=False):
    action_indices = torch.multinomial(probs, num_samples=1)
    if use_cuda:
        tensor_type = torch.cuda.FloatTensor
    else:
        tensor_type = torch.FloatTensor
    actions = Variable(tensor_type(*probs.shape).fill_(0)).scatter_(1, action_indices, 1)
    return action_indices, actions


class BasePolicy(nn.Module):

    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=256,
        nonlin=F.leaky_relu,
        norm_in=True,
        onehot_dim=0
    ):
        super(BasePolicy, self).__init__()

        if norm_in:
            self.in_fn = nn.BatchNorm1d(input_size, affine=False)
        else:
            self.in_fn = lambda x: x

        self.linear1 = nn.Linear(input_size + onehot_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.nonlin = nonlin

    def forward(self, x):
        onehot = None
        if type(x) is tuple:
            x, onehot = x
        x = self.in_fn(x)
        if onehot is not None:
            x = torch.cat((onehot, x), dim=1)
        x = self.nonlin(self.linear1(x))
        x = self.nonlin(self.linear2(x))
        x = self.linear3(x)
        return x


class DiscretePolicy(BasePolicy):

    def __init__(self, *args, **kwargs):
        super(DiscretePolicy, self).__init__(*args, **kwargs)

    def forward(
        self,
        observation,
        sample=True,
        return_all_probs=False,
        return_log_pi=False,
        regularize=False,
        return_entropy=False
    ):
        logits = super(DiscretePolicy, self).forward(observation)
        probs = F.softmax(logits, dim=1)
        on_gpu = next(self.parameters()).is_cuda
        if sample:
            action_indices, actions = categorical_sample(probs, use_cuda=on_gpu)
        else:
            actions = onehot_from_logits(probs)

        returns = [actions]
        if return_log_pi or return_entropy:
            log_probs = F.log_softmax(logits, dim=1)
        if return_all_probs:
            returns.append(probs)
        if return_log_pi:
            returns.append(log_probs.gather(1, action_indices))
        if regularize:
            returns.append([(logits**2).mean()])
        if return_entropy:
            returns.append(-(log_probs * probs).sum(1).mean())
        if len(returns) == 1:
            return returns[0]
        return returns


class AttentionCritic(nn.Module):

    def __init__(
        self,
        state_size,
        action_size,
        hidden_size=128,
        norm_in=True,
        attend_heads=1,
        n=3
    ):
        super(AttentionCritic, self).__init__()
        assert hidden_size % attend_heads == 0

        self.state_action_size = (state_size, action_size)
        self._n = n
        self.attend_heads = attend_heads

        self.critic_encoders = nn.ModuleList()
        self.critics = nn.ModuleList()

        self.state_encoders = nn.ModuleList()

        input_size = state_size + action_size
        output_size = action_size

        for _ in range(n):
            encoder = nn.Sequential()
            if norm_in:
                encoder.add_module('enc_bn', nn.BatchNorm1d(input_size, affine=False))
            encoder.add_module('enc_fc1', nn.Linear(input_size, hidden_size))
            encoder.add_module('enc_nl', nn.LeakyReLU())
            self.critic_encoders.append(encoder)

            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(2 * hidden_size, hidden_size))
            critic.add_module('critic_nl', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(hidden_size, output_size))
            self.critics.append(critic)

            state_encoder = nn.Sequential()
            if norm_in:
                state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(state_size, affine=False))
            state_encoder.add_module('s_enc_fc1', nn.Linear(state_size, hidden_size))
            state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
            self.state_encoders.append(state_encoder)

        attend_dim = hidden_size // attend_heads
        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()  # Query
        self.value_extractors = nn.ModuleList()

        for _ in range(attend_dim):
            self.key_extractors.append(nn.Linear(hidden_size, attend_dim, bias=False))
            self.selector_extractors.append(nn.Linear(hidden_size, attend_dim, bias=False))
            self.value_extractors.append(nn.Sequential(
                nn.Linear(hidden_size, attend_dim),
                nn.LeakyReLU()
            ))

        self.shared_modules = [
            self.key_extractors,
            self.selector_extractors,
            self.value_extractors,
            self.critic_encoders
        ]

    def forward(
        self,
        inputs,
        agents=None,
        return_q=True,
        return_all_q=False,
        regularize=False,
        return_attend=False,
        logger=None,
        niter=0
    ):
        """
        Args:
            inputs (list of PyTorch Matrices): Inputs to each agents' encoder
                                               (batch of obs + ac)
            agents (int): indices of agents to return Q for
            return_q (bool): return Q-value
        """
        if agents is None:
            agents = range(len(self.critic_encoders))

        states = [s for s, a in inputs]
        actions = [a for s, a in inputs]
        inputs = [torch.cat((s, a), dim=1) for s, a in inputs]
        # extract state-action encoding for each agent
        sa_encodings = [encoder(input_) for encoder, input_ in zip(self.critic_encoders, inputs)]
        # extract state encoding for each agent that we're return Q for
        s_encodings = [self.state_encoders[a_i](states[a_i]) for a_i in agents]
        # extract keys for each head for each agent
        all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors]
        # extract state-action values for each head for each agent
        all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]
        # extract selectors for each head for each agent that we're returning Q for
        all_head_selectors = [[sel_ext(enc) for i, enc in enumerate(s_encodings) if i in agents]
                              for sel_ext in self.selector_extractors]

        other_all_values = [[] for _ in range(len(agents))]
        all_attend_logits = [[] for _ in range(len(agents))]
        all_attend_probs = [[] for _ in range(len(agents))]

        # calculate attention per head
        for curr_head_keys, curr_head_values, curr_head_selectors in zip(
                all_head_keys, all_head_values, all_head_selectors):
            # iterate over agents
            for i, a_i, selector in zip(range(len(agents)), agents, curr_head_selectors):
                keys = [k for j, k in enumerate(curr_head_keys) if j != a_i]
                values = [v for j, v in enumerate(curr_head_values) if j != a_i]
                # calculate attention across agents
                attend_logits = torch.matmul(selector.view(selector.shape[0], 1, -1),
                                             torch.stack(keys).permute(1, 2, 0))
                # scale dot-products by size of key (from Attention is All You Need)
                scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])
                attend_weights = F.softmax(scaled_attend_logits, dim=2)
                other_values = (torch.stack(values).permute(1, 2, 0) * attend_weights).sum(dim=2)
                other_all_values[i].append(other_values)
                all_attend_logits[i].append(attend_logits)
                all_attend_probs[i].append(attend_weights)

        # calculate Q per agent
        all_returns = []
        for i, a_i in enumerate(agents):
            head_entropies = [(-((probs + 1e-8).log() * probs).squeeze().sum(dim=1).mean())
                              for probs in all_attend_probs[i]]
            agent_returns = []
            critic_in = torch.cat((s_encodings[i], *other_all_values[i]), dim=1)
            all_q = self.critics[a_i](critic_in)
            action_indices = actions[a_i].max(dim=1, keepdim=True)[1]
            q = all_q.gather(1, action_indices)
            if return_q:
                agent_returns.append(q)
            if return_all_q:
                agent_returns.append(all_q)
            if regularize:
                # regularize magnitude of attention logits
                attend_mag_reg = 1e-3 * sum((logits ** 2).mean() for logit in all_attend_logits[i])
                regs = (attend_mag_reg,)
                agent_returns.append(regs)
            if return_attend:
                agent_returns.append(np.array(all_attend_probs[i]))
            if logger is not None:
                logger.add_scalar(f'agent{a_i}/attention',
                                  dict((f'head{h_i}_entropy', ent) for h_i, ent in enumerate(head_entropies)), niter)

            if len(agent_returns) == 1:
                all_returns.append(agent_returns[0])
            else:
                all_returns.append(agent_returns)

        if len(all_returns) == 1:
            return all_returns[0]
        return all_returns

    def shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        """
        return chain(*[m.parameters() for m in self.shared_modules])

    def scale_shared_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.shared_parameters():
            p.grad.data.mul_(1.0 / self.n)

    @property
    def n(self):
        return self._n


if __name__ == "__main__":
    pass
