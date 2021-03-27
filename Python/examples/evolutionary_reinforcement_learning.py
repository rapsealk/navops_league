#!/usr/bin/python3
# -*- coding: utf-8 -*-
import random
from difflib import SequenceMatcher
from itertools import count

alphabet = (alphabet := "abcdefghijklmnop") + alphabet.upper() + ",.! "
target = "Hello, World!"


class Individual:

    def __init__(self, string, fitness=0):
        self.string = string
        self.fitness = fitness


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def spawn_population(length=26, size=100):
    pop = []
    for i in range(size):
        string = ''.join(random.choices(alphabet, k=length))
        individual = Individual(string)
        pop.append(individual)
    return pop


def recombine(p1: Individual, p2: Individual):
    child1, child2 = [], []
    cross_pt = random.randint(0, len(p1.string))
    child1.extend(p1.string[0:cross_pt])
    child1.extend(p2.string[cross_pt:])
    child2.extend(p2.string[0:cross_pt])
    child2.extend(p1.string[cross_pt:])
    return Individual(''.join(child1)), Individual(''.join(child2))


def mutate(x: Individual, rate=0.01):
    new_x = []
    for char in x.string:
        if random.random() < rate:
            new_x.extend(random.choices(alphabet, k=1))
        else:
            new_x.extend(char)
    return Individual(''.join(new_x))


def evaluate_population(population, target):
    avg_fit = 0
    for individual in population:
        fitness = similarity(individual.string, target)
        individual.fitness = fitness
        avg_fit += fitness
    avg_fit /= len(population)
    return population, avg_fit


def next_generation(population, size=100, length=26, rate=0.01):
    next_population = []
    while len(next_population) < size:
        parents = random.choices(population, k=2, weights=[x.fitness for x in population])
        offspring = recombine(parents[0], parents[1])
        child1 = mutate(offspring[0], rate=rate)
        child2 = mutate(offspring[1], rate=rate)
        offspring = [child1, child2]
        next_population.extend(offspring)
    return next_population


def main():
    population_size = 900
    str_len = len(target)
    mutation_rate = 0.0001

    pop_fit = []
    pop = spawn_population(size=population_size, length=str_len)
    for generation in count(1):
        pop, avg_fit = evaluate_population(pop, target)
        pop_fit.append(avg_fit)
        if generation % 100 == 0:
            print(f'Generation #{generation}: {pop[0].string} ({pop[0].fitness})')
        pop = next_generation(pop, size=population_size, length=str_len, rate=mutation_rate)
        pop = sorted(pop, key=lambda x: x.fitness, reverse=True)

        if pop[0] == target:
            break

    pop.sort(key=lambda x: x.fitness, reverse=True)
    print(f'Generation #{generation}: {pop[0].string} ({pop[0].fitness})')


if __name__ == "__main__":
    main()
