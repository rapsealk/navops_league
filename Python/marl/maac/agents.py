#!/usr/bin/python3
# -*- coding: utf-8 -*-
import torch.optim as optim

from models import DiscretePolicy
from utils import hard_update


class AttentionAgent:

    def __init__(
        self,
        input_size_policy,
        output_size_policy,
        hidden_size=128,
        learning_rate=1e-2,
        onehot_dim=0
    ):
        self.policy = DiscretePolicy(
            input_size_policy,
            output_size_policy,
            hidden_size=hidden_size,
            onehot_dim=onehot_dim
        )
        self.target_policy = DiscretePolicy(
            input_size_policy,
            output_size_policy,
            hidden_size=hidden_size,
            onehot_dim=onehot_dim
        )

        hard_update(self.target_policy, self.policy)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

    def step(self, observation, explore=False):
        return self.policy(observation, sample=explore)

    def get_state_dicts(self):
        return {
            "policy": self.policy.state_dict(),
            "target_policy": self.target_policy.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict()
        }

    def set_state_dicts(self, state_dicts):
        self.policy.load_state_dict(state_dicts["policy"])
        self.target_policy.load_state_dict(state_dicts["target_policy"])
        self.policy_optimizer.load_state_dict(state_dicts["policy_optimizer"])


if __name__ == "__main__":
    pass
