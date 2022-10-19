from dataclasses import dataclass
from particle_swarm_optimization import ParticleSwarmOptimization

@dataclass
class Param:
    num_particles: int = 0
    inertia_weight: float = 0.0
    cognitive_parameter: float = 0.0
    social_parameter: float = 0.0


def cost_function(x):
    num_invalid_restrictions = 0
    for i, Q_i in enumerate(x):
        for j, Q_j in enumerate(x):
            if i != j and abs(Q_i - Q_j) == abs(i - j):
                num_invalid_restrictions += 1
    return num_invalid_restrictions // 2


hyperparams = Param()
hyperparams.num_particles = 100
hyperparams.inertia_weight = 0.7
hyperparams.cognitive_parameter = 0.8
hyperparams.social_parameter = 0.9

pso = ParticleSwarmOptimization(hyperparams, 8)
position_history = []
quality_history = []

# Number of function evaluations will be 1000 times the number of particles,
# i.e. PSO will be executed by 1000 generations
num_evaluations = 1000 * hyperparams.num_particles
for i in range(num_evaluations):
    position = pso.get_position_to_evaluate()
    value = cost_function(position.list)
    pso.notify_evaluation(value)
    position_history.append(position.list)
    quality_history.append(value)
# Finally, print the best position found by the algorithm and its value
print('Best position:', pso.get_best_position())
print('Best value:', pso.get_best_value())