from random import randint
from typing import List

import pandas as pd
from matplotlib import pyplot as plt
from pyeasyga.pyeasyga import GeneticAlgorithm as GA

from src.model import Model


class GeneticAlgorithm:
    def __init__(
        self,
        X: pd.DataFrame,
        model: Model,
        population_size: int = 100,
        mutation_probability: float = 0.01,
        crossover_probability: float = 0.7,
        elitism_rate: float = 0.1,
        max_generations: int = 1000,
        max_stagnation: int = 100,
        verbose: bool = False,
    ):

        self.X = X
        self.model = model
        self.population_size = population_size
        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
        self.elitism_rate = elitism_rate
        self.verbose = verbose
        self.max_generations = max_generations
        self.max_stagnation = max_stagnation

        self.genetic_algorithm = GA(
            self.X.sample(1).reset_index(drop=True),
            self.population_size,
            generations=self.max_generations,
            crossover_probability=self.crossover_probability,
            mutation_probability=self.mutation_probability,
            elitism=self.elitism_rate,
            maximise_fitness=False,
        )

        self.fitnesses: List[float] = []

    def fitness(self, individual, data=None):
        y_pred = self.model.predict(individual)
        if len(self.fitnesses) == 0:
            self.fitnesses.append(y_pred)
        else:
            self.fitnesses.append((y_pred if y_pred < self.fitnesses[-1] else self.fitnesses[-1]))
        return self.model.predict(individual)

    def create_individual(self, data=None):
        return self.X.sample(1).reset_index(drop=True)

    def crossover(self, parent1: pd.DataFrame, parent2: pd.DataFrame):
        child1 = parent1.copy()
        child2 = parent2.copy()
        index = randint(0, len(parent1.columns) - 1)
        child1.iloc[:, index:] = parent2.iloc[:, index:]
        child2.iloc[:, index:] = parent1.iloc[:, index:]
        return child1, child2

    def mutate(self, individual):
        new_individual = self.create_individual()
        mutation_index = randint(0, len(individual.columns) - 1)
        individual.iloc[:, mutation_index] = new_individual.iloc[:, mutation_index]
        return individual

    def fit(self):
        self.genetic_algorithm.create_individual = self.create_individual
        self.genetic_algorithm.crossover_function = self.crossover
        self.genetic_algorithm.mutate_function = self.mutate
        self.genetic_algorithm.fitness_function = self.fitness
        self.genetic_algorithm.run()

        fitness, best_individual = self.genetic_algorithm.best_individual()
        return fitness, best_individual

    def show_history(self):
        fitnesses = self._best_individuals_per_generation(self.fitnesses, self.genetic_algorithm.population_size)
        plt.plot(fitnesses)
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.show()

    def _best_individuals_per_generation(self, fitnesses, population_size):
        return [min(fitnesses[i : i + population_size]) for i in range(0, len(fitnesses), population_size)]

    def report_best_individual(self):
        fitness, best_individual = self.genetic_algorithm.best_individual()
        print(f"Best individual:\n{best_individual.iloc[0].to_markdown()}")
        print(f"Fitness: {fitness}")
