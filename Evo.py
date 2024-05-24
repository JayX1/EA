import random
import numpy as np
from typing import List, Callable  # dodać dla selection
import pandas as pd


class Matrix:
    def __init__(self, size):
        self.size = size
        self.arr = []

    @property
    def max(self):
        return max(self.arr[0])


class Triangular(Matrix):
    def __init__(self, size):
        super().__init__(size)
        self.arr = np.array(
            [[0, 48, 16, 36, 35]
                , [0, 0, 26, 31, 49]
                , [0, 0, 0, 25, 44]
                , [0, 0, 0, 0, 22]
                , [0, 0, 0, 0, 0]])

    def create(self):
        matrix = np.zeros((self.size, self.size))
        for row in range(self.size):
            for col in range(row, self.size - 1):
                matrix[row, col + 1] = random.randint(15, 50)
        self.arr = matrix


class Jagged(Matrix):
    def __init__(self, size):
        super().__init__(size)
        self.arr = [[0, 48, 16, 36, 35]
            , [0, 26, 31, 49]
            , [0, 25, 44]
            , [0, 22]
            , [0]]

    def create(self):
        matrix = []
        for i in range(self.size):
            matrix.append(random.choices(list(map(lambda x: x + 10, range(41))), k=self.size - i - 1))
            matrix[i].insert(0, 0)
        self.arr = matrix


class Evo:
    def __init__(self, matrix: Matrix):
        self.weight_matrix = matrix
        self.population : List[List[int]] = []
        self.size = self.weight_matrix.size
        self.prime_specimen = self.weight_matrix.max * self.size

    @property
    def population_size(self):
        return len(self.population)

    def Create_genome(self) -> List[int]:
        return random.sample(list(range(self.size)), self.size)

    def Create_population(self) -> None:
        self.population = [self.Create_genome() for _ in range(5)]

    def Fitness(self):  # -> List[int]:
        weight_size = self.size - 1
        weights = []
        for i in range(self.population_size):
            weight = 0
            specimen = self.population[i]

            for i in range(weight_size):
                p, n = specimen[i], specimen[i + 1]
                if p < n:
                    weight += self.weight_matrix.arr[p][n - p]
                else:
                    weight += self.weight_matrix.arr[n][p - n]

            if weight < self.prime_specimen:
                self.prime_specimen = weight
            weights.append(weight)

        return weights

    def Selection(self,num :int = 2) -> List[List[int]]:
        return random.choices(
            population=self.population,
            weights=self.Fitness(),
            k=num
        )

    def Crossover(self):
        print("crossover")
        # parent1: List[int] = specimens[0]
        # parent2: List[int] = specimens[1]

        parent1: List[int] = self.Selection()[0]
        parent2: List[int] = self.Selection()[1]

        #slice_point = round(len(parent1) / 2)
        slice_point = random.randint(1,self.size-1)

        child1 = parent1[slice_point:]+parent2[:slice_point]
        child2 = parent2[slice_point:]+parent1[:slice_point]
        self.population.append(child1)
        self.population.append(child2)

    def Mutate(self):
        for specimen_index in range(self.population_size):
            specimen = self.population[specimen_index]
            mutation1_index,mutation2_index = random.sample(range(0,self.size),2)

            gene1 = specimen[mutation1_index]
            gene2 = specimen[mutation2_index]
            specimen[mutation2_index] = gene1
            specimen[mutation1_index] = gene2
        print("Mutate")

    def Drive_evolution(self,generations):
        for _ in range(generations):
            evo.Mutate()
            evo.Crossover()


m = Jagged(5)
evo = Evo(m)
evo.Create_population()
print(f"Prime {evo.prime_specimen}")
print(evo.population)
for _ in range(100):
    evo.Mutate()
    evo.Crossover()
    evo.Fitness()
print(evo.prime_specimen)
