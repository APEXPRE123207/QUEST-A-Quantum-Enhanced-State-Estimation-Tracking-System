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
    total_time: int = 60 #120
    num_particles: int = 700 #700  # optimized for precision/performance balance
    # Randomness and physics controls
    seed: int = 42
    missile_max_speed: float = 450.0 #350.0
    missile_max_accel: float = 80.0 #50.0
    # Particle filter and planner safeguards
    ess_resample_threshold_ratio: float = 0.5
    fallback_window: int = 3
    # Quantum planning controls
    planning_horizon: int = 4
    waypoint_step: float = 700.0 #700.0

@dataclass
class QAOAOptions:
    """A dataclass for QAOA parameters."""
    reps: int = 2
    shots: int = 64
    max_iterations: int = 25
    w1: float = 0.5
    w2: float = 0.5
    P: int = 24 # Penalty coefficient