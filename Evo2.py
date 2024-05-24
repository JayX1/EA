import random
from random import randint
from random import sample
from random import choices
import numpy as np
from typing import List, Callable  # dodać dla selection
import copy
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

class Specimen:
    def __init__(self,genome:List[int],fitness:int):
        self.genome = genome
        self.fitness = fitness
    def __str__(self):
        print(f"{self.genome} \n {self.fitness} \n")

class Evo:
    def __init__(self, matrix: Matrix):
        self.weight_matrix = matrix
        self.population : List[Specimen] = []
        self.size = self.weight_matrix.size

    @property
    def population_size(self):
        return len(self.population)

    @property
    def prime_specimen(self) -> int:
        print("Prime")
        return min(self.population , key=lambda specimen : specimen.fitness).fitness

    def Print_population(self):
        for specimen in self.population:
            specimen.__str__()

    def Create_population(self) -> None:
        #random.sample(list(range(self.size)), self.size)
        for _ in range(5):
            genome = sample(list(range(self.size)), self.size)
            fitness = self.Fitness(genome)
            self.population.append(Specimen(genome,fitness))


    def Fitness(self,genome:List[int]) -> int:
        weight_size = self.size - 1
        weight = 0

        for index in range(weight_size):
            prev, next = genome[index], genome[index + 1]
            if prev < next:
                weight += self.weight_matrix.arr[prev][next - prev]
            else:
                weight += self.weight_matrix.arr[next][prev - next]

        return weight

    def Selection(self,num :int = 2) -> List[Specimen]:
        return choices(
            population=self.population,
            weights=[specimen.fitness for specimen in self.population],
            k=num
        )

    def Selection_tournament(self):
        self.population.remove((max(self.Selection(),key=lambda specimen:specimen.fitness)))


    def Crossover(self):
        print("crossover")

        child_1: List[int] = copy.deepcopy(self.Selection()[0].genome)
        child_2: List[int] = copy.deepcopy(self.Selection()[1].genome)
        # slice_point = round(self.size / 2)
        slice_point = randint(1, self.size - 1)
        child_1[slice_point:],child_2[slice_point:] = \
        child_2[slice_point:],child_1[slice_point:]
        self.population.append(Specimen(child_1, self.Fitness(child_1)))
        self.population.append(Specimen(child_2, self.Fitness(child_2)))

    def Crossover_uniform(self, freq:float = 0.33):
        print("uniform crossover")
        child_1: List[int] = copy.deepcopy(self.Selection()[0].genome)
        child_2: List[int] = copy.deepcopy(self.Selection()[1].genome)
        indexes = choices(population=range(self.size),
                          k=round(self.size * freq))

        for i in range(self.size):
            if i in indexes:
                child_1[i],child_2[i] = child_2[i],child_1[i]
        self.population.append(child_1)
        self.population.append(child_2)

    def Mutate(self):
        print("Mutate")
        for i in range(self.population_size):
            specimen = self.population[i].genome
            mutation_1,mutation_2 = sample(range(0,self.size),2)
            specimen[mutation_1],specimen[mutation_2] = \
            specimen[mutation_2], specimen[mutation_1]


    def Mutate_shift(self,shift:int = 3):
        print("Mutate with shift")
        for i in range(self.population_size):
            specimen = self.population[i].genome
            print(f"before {specimen}")
            print("--------")
            print(list(map(lambda x:abs(x-self.size+1),specimen)))


    def Mutate_shuffle(self):
        print("Mutate with shuffle")
        [random.shuffle(specimen.genome) for specimen in self.population]

    def Mutate_inversion(self):


m = Jagged(5)
evo = Evo(m)
evo.Create_population()
# print(evo.prime_specimen)
# print(evo.population)
# for _ in range(3):
evo.Print_population()
evo.Mutate_shuffle()
evo.Print_population()
#     evo.Crossover()
#     evo.Selection_tournament()
#     evo.Selection_tournament()
#     evo.Print_population()
# print(evo.prime_specimen)
