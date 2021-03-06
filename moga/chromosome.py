import numpy as np
import random
from .exception import GenoTypeRangeException


def decimal_to_binary_with_integer(num, num_digit, offspring=0):
    return format(num + offspring, 'b').zfill(num_digit)


def decimal_to_binary(np_array, num_digit, offspring=0):
    binary_string = ""
    for idx in np_array:
        binary_string += decimal_to_binary_with_integer(idx, num_digit, offspring)
    return binary_string


def binary_to_decimal(binary_string, offspring=0):
    return int('0b' + binary_string, 2) - offspring


def binary_to_decimal_with_digit(binary_string, num_digit, offspring=0):
    num_element = int(len(binary_string) / num_digit)
    decimal_list = []
    for idx in range(0, num_element):
        string = binary_string[num_digit * idx: num_digit * (idx +1)]
        decimal_list.append(binary_to_decimal(string, offspring))
    return decimal_list


class GenoShapeEnum(object):
    NUM_VALUE = 'num'
    MIN_VALUE = 'min'
    MAX_VALUE = 'max'
    DIGIT = 'digit'


class BinaryChromosome(object):
    geno_shape = {}
    geno_position = {}
    len_str_chromosome = 0
    fitted_in_geno_func_list = []

    @classmethod
    def get_random_chromosome_from_geno_shape(cls, num_chromosome=1):
        chromosome_list = []
        for chromosome_idx in range(0, num_chromosome):
            while len(chromosome_list) < num_chromosome:
                chromosome_str = cls.get_random_chromosome()
                fitted = cls.chromosome_fitted_in_geno_space(chromosome_str)
                if chromosome_str not in chromosome_list and fitted:
                    chromosome_list.append(chromosome_str)

        return chromosome_list

    @classmethod
    def get_phenotype(cls, chromosome):
        # TODO - check chromosome genotype function
        pheno_info = {}
        for key, elements in cls.geno_position.items():
            value_list = []
            position_list = cls.geno_position[key]
            for min_pos, max_pos in position_list:
                value_str = chromosome[min_pos:max_pos]
                value_list.append(binary_to_decimal(value_str, cls.geno_shape[key]['offspring']))
            pheno_info[key] = np.array(value_list)
        return pheno_info

    @classmethod
    def get_phenotype_content(cls, chromosome, key):
        value_str = cls.get_chromosome_content(chromosome, key)
        return binary_to_decimal_with_digit(value_str, cls.geno_shape[key]['digit'],
                                            cls.geno_shape[key]['offspring'])

    @classmethod
    def get_chromosome_content(cls, chromosome, key):
        value_str = ''
        position_list = cls.geno_position[key]
        for min_pos, max_pos in position_list:
            value_str += chromosome[min_pos:max_pos]
        return value_str

    @classmethod
    def get_chromosome_element(cls, chromosome, key, idx):
        min_pos, max_pos = cls.geno_position[key][idx]
        return chromosome[min_pos:max_pos]

    @classmethod
    def get_phenotype_element(cls, chromosome, key, idx):
        min_pos, max_pos = cls.geno_position[key][idx]
        return binary_to_decimal(chromosome[min_pos:max_pos], cls.geno_shape[key]['offspring'])

    @classmethod
    def get_genotype(cls, **element_info):
        chromosome = ''
        for key, elements in element_info.items():
            chromosome += decimal_to_binary(elements, cls.geno_shape[key]['digit'],
                                            cls.geno_shape[key]['offspring'])
        return chromosome

    @classmethod
    def set_geno_shape(cls, **geno_shape):
        for key, geno_info in geno_shape.items():
            if cls.__check_bound_info(geno_info):
                cls.geno_shape[key] = geno_info
        cls.__set_geno_position()
        cls.__set_len_str_chromosome()

    # private class method
    @classmethod
    def __set_geno_position(cls):
        pos = 0
        for key, content in cls.geno_shape.items():
            interval = content['digit']
            num_content = content['num']
            position_list = [(pos + idx * interval, pos + (idx+1) * interval) for idx in range(0, num_content)]
            cls.geno_position[key] = position_list
            pos += content['digit'] * content['num']

    @classmethod
    def __check_bound_info(cls, geno_info):
        offspring = geno_info['offspring']
        lower_bound = 0 - offspring
        upper_bound = pow(2, geno_info['digit']) - offspring
        if geno_info['min'] < lower_bound:
            check = False
            raise GenoTypeRangeException(1, lower_bound, geno_info['min'])
        elif geno_info['max'] >= upper_bound:
            check = False
            raise GenoTypeRangeException(0, upper_bound, geno_info['max'])
        else:
            check = True
        return check

    @classmethod
    def chromosome_fitted_in_geno_space(cls, chromosome, new_generation=[]):
        fitted = True
        for key, elements in cls.geno_position.items():
            min_value = cls.geno_shape[key]['min'] + cls.geno_shape[key]['offspring']
            max_value = cls.geno_shape[key]['max'] + cls.geno_shape[key]['offspring']
            breaker = False
            for min_pos, max_pos in cls.geno_position[key]:
                value_str = chromosome[min_pos:max_pos]
                num = binary_to_decimal(value_str)
                if num < min_value or num > max_value:
                    fitted = False
                    breaker = True
                    break
            if breaker:
                break

        for func in cls.fitted_in_geno_func_list:
            if not func(chromosome, new_generation):
                fitted = False
                break
        return fitted

    @classmethod
    def __set_len_str_chromosome(cls):
        cls.len_str_chromosome = 0
        for content in cls.geno_shape.values():
            cls.len_str_chromosome += content['digit'] * content['num']

    @classmethod
    def get_random_chromosome(cls):
        chromosome_str = ''
        for key, elements in cls.geno_position.items():
            min_value = cls.geno_shape[key]['min'] + cls.geno_shape[key]['offspring']
            max_value = cls.geno_shape[key]['max'] + cls.geno_shape[key]['offspring']
            for idx in range(0, cls.geno_shape[key]['num']):
                rand_num = random.randint(min_value, max_value)
                # Offspring is already applied at upper lines
                chromosome_str += decimal_to_binary_with_integer(
                    rand_num, cls.geno_shape[key]['digit'])
        return chromosome_str
