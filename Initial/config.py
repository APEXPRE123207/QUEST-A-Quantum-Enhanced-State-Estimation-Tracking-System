from dataclasses import dataclass

@dataclass
class QNEATOptions:
    """A dataclass to hold all hyperparameters for the QNEAT algorithm."""
    # Population
    population_size: int = 150
    
    # Mutation
    add_rot_prob: float = 0.1
    add_cnot_prob: float = 0.05
    weight_mutate_prob: float = 0.8
    new_weight_prob: float = 0.1
    weight_mutate_power: float = 0.5
    
    # Crossover & Speciation
    crossover_rate: float = 0.75
    compatibility_threshold: float = 3.0
    c1: float = 1.0 # Disjoint coefficient
    c2: float = 1.0 # Excess coefficient
    c3: float = 0.4 # Weight difference coefficient
    
    # Elitism & Survival
    species_elitism: int = 1
    survival_rate: float = 0.2

@dataclass
class SimOptions:
    """A dataclass for general simulation parameters."""
    dt: float = 0.1
    total_time: int = 20 #120
    num_particles: int =200  #2000

@dataclass
class QAOAOptions:
    """A dataclass for QAOA parameters."""
    reps: int = 2
    w1: float = 0.5
    w2: float = 0.5
    P: int = 24 # Penalty coefficient