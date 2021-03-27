#!/usr/bin/python3
# -*- coding: utf-8 -*-
from itertools import count

import gym
import numpy as np
import torch
import torch.nn.functional as F


def model(x, *args):
    l1, b1, l2, b2, l3, b3 = args
    y = F.relu(F.linear(x, l1, b1))
    y = F.relu(F.linear(y, l2, b2))
    y = F.linear(y, l3, b3)
    y = torch.log_softmax(y, dim=0)
    return y


def unpack_params(params, layers):
    # TODO: Layer-by-Layer
    unpacked_params = []
    e = 0
    for i, l in enumerate(layers):
        s, e = e, e + np.prod(l)
        weights = params[s:e].view(l)
        s, e = e, e + l[0]
        bias = params[s:e]
        unpacked_params.extend([weights, bias])
    return unpacked_params


def _get_param_size(layers):
    param_size = 0
    for outputs, inputs in layers:
        param_size += outputs * inputs + outputs
    return param_size


def spawn_population(n=50, size=407):
    return [
        {"params": torch.randn(size) / 2.0, "fitness": 0.0}
        for _ in range(n)
    ]


def recombine(x1, x2):
    x1, x2 = x1['params'], x2['params']
    shape = x1.shape[0]
    split_pt = np.random.randint(shape)
    child1, child2 = torch.zeros((2, shape))
    child1[0:split_pt] = x1[0:split_pt]
    child1[split_pt:] = x2[split_pt:]
    child2[0:split_pt] = x2[0:split_pt]
    child2[split_pt:] = x1[split_pt:]
    c1 = {"params": child1, "fitness": 0.0}
    c2 = {"params": child2, "fitness": 0.0}
    return c1, c2


def mutate(x, rate=0.01):
    x_ = x['params']
    num_to_change = int(rate * x_.shape[0])
    idx = np.random.randint(low=0, high=x_.shape[0], size=(num_to_change,))
    x_[idx] = torch.randn(num_to_change) / 10.0
    x['params'] = x_
    return x


def test_model(env, agent, layers):
    done = False
    state = torch.from_numpy(env.reset()).float()
    score = 0
    while not done:
        params = unpack_params(agent['params'], layers)
        probs = model(state, *params)
        action = torch.distributions.Categorical(probs).sample()
        state_, reward, done, info = env.step(action.item())
        state = torch.from_numpy(state_).float()
        score += 1
    return score


def evaluate_population(env, population, layers):
    total_fitness = 0
    for agent in population:
        agent['fitness'] = test_model(env, agent, layers)
        total_fitness += agent['fitness']
    average_fitness = total_fitness / len(population)
    return population, average_fitness


def next_generation(population, rate=0.001, tournament_size=0.2):
    next_population = []
    while len(next_population) < len(population):
        rids = np.random.randint(low=0, high=len(population), size=int(tournament_size * len(population)))
        batch = np.array([[i, x["fitness"]]
                          for i, x in enumerate(population) if i in rids])
        scores = batch[batch[:, 1].argsort()]
        i0, i1 = int(scores[-1][0]), int(scores[-2][0])
        parent0, parent1 = population[i0], population[i1]
        offspring = recombine(parent0, parent1)
        child1 = mutate(offspring[0], rate=rate)
        child2 = mutate(offspring[1], rate=rate)
        offspring = [child1, child2]
        next_population.extend(offspring)
    return next_population


def main():
    population_size = 500
    mutation_rate = 0.01
    population_fitness = []
    layers = [(25, 4), (10, 25), (2, 10)]
    population = spawn_population(n=population_size, size=_get_param_size(layers))

    env = gym.make('CartPole-v0')

    for generation in count(1):
        population, avg_fit = evaluate_population(env, population, layers)
        population_fitness.append(avg_fit)
        print(f'Generation #{generation}: {avg_fit}')
        population = next_generation(population, rate=mutation_rate, tournament_size=0.2)

        if generation == 200:
            break


if __name__ == "__main__":
    main()
