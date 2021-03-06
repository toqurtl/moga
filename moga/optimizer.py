from .chromosome import BinaryChromosome
from .generation import Fronting
import numpy as np
import time
import pickle


class ParetoOptimizer(object):
    # generation_list - chromosome, objective value
    def __init__(self):
        self.num_chromosome_in_generation = 50
        self.max_generation = 10
        self.num_objective = 2
        self.generation_list = []
        self.time_measure_list = []
        Fronting.num_objective = self.num_objective

    def set_generation_generator(self, generation_generator):
        self.generation_generator = generation_generator
        self.generic_parameter_dict = generation_generator.generic_parameter_dict
        self.objective_function = generation_generator.fitness_func
        self.local_algorithm = generation_generator.local_algorithm_enum

    def setting(self, num_chromosome_in_generation=50, max_generation=10, num_objective=2):
        self.num_chromosome_in_generation = num_chromosome_in_generation
        self.max_generation = max_generation
        self.num_objective = num_objective

    def optimize(self, file_path=None, generation=None):

        self.initialization(generation=generation)
        for idx in range(0, self.max_generation):
            start_time = time.time()
            self.next_generation()
            self.time_measure_list.append(time.time()-start_time)
            if file_path is not None:
                with open(file_path, 'wb') as f:
                    pickle.dump(self.get_generation_dict(), f, pickle.HIGHEST_PROTOCOL)

    def initialization(self, generation=None):
        if generation is not None:
            self.generation_list.append(generation)
            return

        num = self.num_chromosome_in_generation
        chromosome_list = BinaryChromosome.get_random_chromosome_from_geno_shape(num)
        local_optimized_chromosome_list = []
        for chromosome in chromosome_list:
            chromosome, objective_values = self.local_algorithm(chromosome, self.objective_function)
            local_optimized_chromosome_list.append((chromosome, objective_values))
        self.generation_list.append(np.array(local_optimized_chromosome_list))

    def next_generation(self):
        self.generation_generator.set_generation(self.generation_list[-1])
        new_generation_info = self.generation_generator.get_new_generation(self.num_chromosome_in_generation * 2)
        Fronting.fronting(new_generation_info)
        new_generation_info = Fronting.get_survived_chromosome(self.num_chromosome_in_generation)

        self.generation_list.append(new_generation_info)
        return new_generation_info

    def get_generation_dict(self, milestone=0):
        generation_dict = {}
        generation_list = self.generation_list
        for idx, generation in enumerate(generation_list):
            if idx != 0:
                num = idx + milestone
                time = self.time_measure_list[idx-1]
                generation_dict[num] = (num, generation, time)
        return generation_dict

