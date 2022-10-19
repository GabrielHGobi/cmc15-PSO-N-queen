import numpy as np
import random
from math import inf
from collections import OrderedDict

class Sequence:
    """
    Represents the position of a particle in a Discrete PSO problem.
    """

    def __init__(self, n):
        """
        Creates a sequence from 1 to n with no repetitions.
        :param n: number of elements in the sequence.
        :type n: int.
        """
        self.list = list(range(1, n + 1))
        random.shuffle(self.list)

    def get_distance_to(self, other_seq):
        count = 0
        for i, val in enumerate(self.list):
            if val != other_seq[i]:
                count += 1
        return count

    def __copy__(self):
        new_sequence = Sequence(len(self.list))
        new_sequence.list = self.list.copy()
        return new_sequence

    def __sub__(self, other):
        v = PermutationList()
        other_list = other.list.copy()
        while self.list != other_list:
            i = random.randrange(0, len(self.list))
            if self.list[i] != other_list[i]:
                searching = self.list[i]
                for j in range(i+1, len(self.list)):
                    if other_list[j] == searching:
                        v.add((i, j))
                        other_list[j] = other_list[i]
                        other_list[i] = searching
                        break
        return v

    def __add__(self, permutation_list):
        if type(permutation_list) == PermutationList:
            sum_sequence = self.__copy__()
            for permutation in permutation_list.keys():
                i, j = permutation
                aux = sum_sequence.list[j]
                sum_sequence.list[j] = sum_sequence.list[i]
                sum_sequence.list[i] = aux
            return sum_sequence
        else:
            raise TypeError("can only add a PermutationList to Sequence object.")

    def __repr__(self):
        return self.list.__repr__()


class PermutationList:
    """
    Represents the velocity of a particle in a Discrete PSO problem.
    """

    def __init__(self):
        """
        Creates an empty permutation list.
        """
        self.p_list = OrderedDict()

    def add(self, permutation):
        self.p_list.update({permutation: None})

    def keys(self):
        return list(self.p_list.keys())

    def __rmul__(self, n):
        if type(n) == int:
            return self.keys()[0:n]
        elif type(n) == float:
            if n < 0.0 or n > 1.0:
                raise ValueError("can't multiplie PermutationList by %.2f" % n)
            else:
                return self.keys()[0:round(n*len(self.keys()))]
                pass
        else:
            raise TypeError("TODO")

    def __mul__(self, other):
        return self.__rmul__(other)

    def __repr__(self):
        return self.keys().__repr__()


class Particle:
    """
    Represents a particle of Discrete Particle Swarm Optimization algorithm.
    """

    def __init__(self, n):
        """
        Creates a particle of the Particle Swarm Optimization algorithm.

        :param n: number of possible values for the discrete problem.
        :type n: int.
        """
        self.x = Sequence(n)
        self.v = PermutationList()
        self.best = self.x
        self.best_value = inf

    def update(self, w, phip, rp, phig, rg, best_global):
        self.v = w * self.v + \
                 phip * rp * (self.best - self.x) + \
                 phig * rg * (best_global - self.x)
        self.x = self.x + self.v

class ParticleSwarmOptimization:
    """
    Represents the Discrete Particle Swarm Optimization algorithm.
    Hyperparameters:
        num_particles: number of particles used in PSO
        inertia_weight: inertia weight.
        cognitive_parameter: cognitive parameter.
        social_parameter: social parameter.

    :param hyperparams: hyperparameters used by Particle Swarm Optimization.
    :type hyperparams: Params.
    :param n: number of possible values for the discrete problem.
    :type n: int.
    """

    def __init__(self, hyperparams, n):
        # unwrapping hyperparameters
        self.w = hyperparams.inertia_weight
        self.phip = hyperparams.cognitive_parameter
        self.phig = hyperparams.social_parameter

        # for saving the best array of params of ALL iteration and its value
        self.best_global = None
        self.best_global_value = inf

        # for saving the best array of params of EACH iteration and its value
        self.best_iteration = None
        self.best_iteration_value = inf

        # number of discrete values to choose
        self.n = n

        # list of objects representing the particles of PCO
        self.particles = [Particle(n)
                          for i in range(hyperparams.num_particles)]

        # index of the position to be evaluated by the PCO
        self.i = 0

    def get_best_position(self):
        """
        Obtains the best position so far found by the algorithm.

        :return: the best position.
        :rtype: list of int.
        """
        return self.best_global

    def get_best_value(self):
        """
        Obtains the value of the best position so far found by the algorithm.

        :return: value of the best position.
        :rtype: int.
        """
        return self.best_global_value

    def get_position_to_evaluate(self):
        """
        Obtains a new position to evaluate.

        :return: position to evaluate.
        :rtype: list of int.
        """
        return self.particles[self.i].x

    def advance_generation(self):
        """
        Advances the generation of particles.
        Auxiliary method to be used by notify_evaluation().
        """
        for particle in self.particles:
            rp = random.uniform(0.0, 1.0)
            rg = random.uniform(0.0, 1.0)

            # TODO: change this for discrete problem

            particle.x = particle.x + particle.v

    def notify_evaluation(self, value):
        """
        Notifies the algorithm that a particle position evaluation was completed.

        :param value: quality of the particle position.
        :type value: int.
        """
        if value < self.particles[self.i].best_value:
            self.particles[self.i].best = self.particles[self.i].x
            self.particles[self.i].best_value = value
        if value < self.best_iteration_value:
            self.best_iteration = self.particles[self.i].x
            self.best_iteration_value = value

        if self.i == len(self.particles) - 1:
            self.i = 0
            if self.best_iteration_value < self.best_global_value:
                self.best_global = self.best_iteration
                self.best_global_value = self.best_iteration_value
            self.advance_generation()
        else:
            self.i += 1
