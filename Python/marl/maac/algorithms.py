#!/usr/bin/python3
# -*- coding: utf-8 -*-
import torch
import torch.optim as optim

from agents import AttentionAgent
from models import AttentionCritic
from utils import soft_update, hard_update, enable_gradients, disable_gradients

MSELoss = torch.nn.MSELoss()


class AttentionSoftActorCritic:

    def __init__(
        self,
        input_size,
        output_size,
        actor_hidden_size=256,
        critic_hidden_size=256,
        attend_heads=4,
        n=3,
        params={},
        gamma=0.95,
        tau=0.01,
        actor_learning_rate=1e-2,
        critic_learning_rate=1e-2,
        reward_scale=10.0,
        **kwargs
    ):
        super(AttentionSoftActorCritic, self).__init__()

        self._n = n
        self.agents = [
            AttentionAgent(
                input_size,
                output_size,
                hidden_size=actor_hidden_size,
                learning_rate=actor_learning_rate)
        ]
        self.critic = AttentionCritic(
            input_size,
            output_size,
            hidden_size=critic_hidden_size,
            attend_heads=attend_heads)
        self.target_critic = AttentionCritic(
            input_size,
            output_size,
            hidden_size=critic_hidden_size,
            attend_heads=attend_heads)

        hard_update(self.target_critic, self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate, weight_decay=1e-3)

        # self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.reward_scale = reward_scale
        self.actor_device = 'cpu'
        self.critic_device = 'cpu'
        self.target_actor_device = 'cpu'
        self.target_critic_device = 'cpu'
        self.niter = 0

    def step(self, observations, explore=False):
        return [a.step(observation, explore=explore)
                for a, observation in zip(self.agents, observations)]

    def update_critic(self, sample, soft=True, logger=None, **kwargs):
        """
        Update central critic for all agents
        """
        observationss, actions, rewards, next_observations, dones = sample

        # Q Loss
        next_actions = []
        next_log_pis = []
        for pi, obs in zip(self.target_policies, next_observations):
            current_next_action, current_next_log_pi = pi(obs, return_log_pi=True)
            next_actions.append(current_next_action)
            next_log_pis.append(current_next_log_pi)
        target_critic_in = list(zip(next_observations, next_actions))
        critic_in = list(zip(observationss, actions))
        next_q_values = self.target_critic(target_critic_in)
        critic_returns = self.critic(critic_in, regularize=True, logger=logger, niter=self.niter)
        q_loss = 0
        for a_i, nq, log_pi, (pq, regs) in zip(range(self.n), next_q_values, next_log_pis, critic_returns):
            target_q = (rewards[a_i].view(-1, 1) + self.gamma * nq * (1 - dones[a_i].view(-1, 1)))
            if soft:
                target_q -= log_pi / self.reward_scale
            q_loss += MSELoss(pq, target_q.detach())
            for reg in regs:
                q_loss += reg
        q_loss.backward()
        self.critic.scale_shared_grads()
        grad_norm = torch.nn.utils.clip_grad_norm(
            self.critic.parameters(), 10 * self.n)
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()

        if logger is not None:
            logger.add_scalar('losses/q_loss', q_loss, self.niter)
            logger.add_scalar('grad_norms/q', grad_norm, self.niter)
        self.niter += 1

    def update_policies(self, sample, soft=True, logger=None, **kwargs):
        observations, actions, rewards, next_observations, dones = sample
        sample_actions = []
        all_probs = []
        all_log_pis = []
        all_policy_regs = []

        for a_i, pi, obs in zip(range(self.n), self.policies, observations):
            current_action, probs, log_pi, pol_regs, entropy = pi(
                obs, return_all_probs=True, return_log_pi=True,
                regularize=True, return_entropy=True)

            logger.add_scalar(f'agent{a_i}/policy_entropy', ent, self.niter)
            sample_actions.append(current_action)
            all_probs.append(probs)
            all_log_pis.append(log_pi)
            all_policy_regs.append(pol_regs)

        critic_in = list(zip(observations, sample_actions))
        critic_returns = self.critic(critic_in, return_all_q=True)

        for a_i, probs, log_pi, pol_regs, (q, all_q) in zip(range(self.n), all_probs, all_log_pis, all_policy_regs, critic_returns):
            current_agent = self.agents[a_i]
            value = (all_q * probs).sum(dim=1, keepdim=True)
            pol_target = q - value
            if soft:
                pol_loss = (log_pi * (log_pi / self.reward_scale - pol_target).detach()).mean()
            else:
                pol_loss = (log_pi * (-pol_target).detach()).mean()
            for reg in pol_regs:
                pol_loss += 1e-3 * reg

            # don't want critic to accumulate gradients from policy loss
            disable_gradients(self.critic)
            pol_loss.backward()
            enable_gradients(self.critic)

            grad_norm = torch.nn.utils.clip_grad_norm(
                current_agent.policy.parameters(), 0.5)
            current_agent.policy_optimizer.step()
            current_agent.policy_optimizer.zero_grad()

            if logger is not None:
                logger.add_scalar(f'agent{a_i}/losses/pol_loss', pol_loss, self.niter)
                logger.add_scalar(f'agent{a_i}/grad_norms/pi', grad_norm, self.niter)

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been performed for each agent)
        """
        soft_update(self.target_critic, self.critic, tau=self.tau)
        for a in self.agents:
            soft_update(a.target_policy, a.policy, tau=self.tau)

    def prep_training(self, device='gpu'):
        self.critic.train()
        self.target_critic.train()
        for a in self.agents:
            a.policy.train()
            a.target_policy.train()
        if device == 'gpu':
            def fn(x): return x.cuda()
        else:
            def fn(x): return x.cpu()
        if not self.actor_device == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.actor_device = device
        if not self.critic_device == device:
            self.critic = fn(self.critic)
            self.critic_device = device
        if not self.target_actor_device == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.target_actor_device = device
        if not self.target_critic_device == device:
            self.target_critic = fn(self.target_critic)
            self.target_critic_device = device

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.policy.eval()
        if device == 'gpu':
            def fn(x): return x.cuda()
        else:
            def fn(x): return x.cpu()
        # only need main policy for rollouts
        if not self.actor_device == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.acotr_device = device

    def save(self, path):
        self.prep_training(device='cpu')
        state_dicts = {
            "init_dict": self.init_dict,
            "agent_params": [a.get_params() for a in self.agents],
            "critic_params": {
                "critic": self.critic.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict()
            }
        }
        torch.save(state_dicts, path)

    @classmethod
    def init_from_env(
        cls,
        env,
        gamma=0.95,
        tau=0.01,
        pi_lr=1e-2,
        q_lr=1e-2,
        reward_scale=1.0,
        pol_hidden_dim=128,
        critic_hidden_dim=128,
        attend_heads=4,
        **kwargs
    ):
        agent_init_params = []
        sa_size = []

        for action_space, observation_space in zip(env.action_space, env.observation_space):
            agent_init_params.append({
                "num_in_pol": observation_space.shape[0],
                "num_out_pol": action_space.n
            })
            sa_size.append((observation_space.shape[0], action_space.n))

        init_dict = {
            "gamma": gamma,
            "tau": tau,
            "pi_lr": pi_lr,
            "q_lr": q_lr,
            "reward_scale": reward_scale,
            "pol_hidden_dim": pol_hidden_dim,
            "critic_hidden_dim": critic_hidden_dim,
            "attend_heads": attend_heads,
            "agent_init_params": agent_init_params,
            "sa_size": sa_size
        }
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename, load_critic=False):
        save_dict = torch.load(filename)
        instance = cls(**save_dict["init_dict"])
        instance.init_dict = save_dict["init_dict"]
        for a, params in zip(instance.agents, save_dict["agent_params"]):
            a.load_params(params)

        if load_critic:
            critic_params = save_dict["critic_params"]
            instance.critic.load_state_dict(critic_params["critic"])
            instance.target_critic.load_state_dict(critic_params["target_critic"])
            instance.critic_optimizer.load_state_dict(critic_params["critic_optimizer"])

        return instance

    @property
    def n(self):
        return self._n

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]


if __name__ == "__main__":
    pass
