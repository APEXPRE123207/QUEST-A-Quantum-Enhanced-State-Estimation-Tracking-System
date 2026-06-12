# train_qoder.py
# -------------------------------------------------------
# Code-2: QODER / Quantum NEAT training script
# High-quality training (Option B)
#
# Trains a quantum circuit to predict target acceleration
# and velocity change from sensor-like features that match
# what QCTPF will use at runtime.
# -------------------------------------------------------

import numpy as np
import pickle
import os
import random

from qiskit_aer import AerSimulator
from qiskit import transpile

from Quantum_Core.qneat import Population, QNEATOptions, Genome
from Quantum_Core.nqpf import build_circuit_from_genome
from Simulation.target_dynamics import Target
from Simulation.sensor_model import Sensor
from config import SimOptions

# ======================================================
# Global Simulator & Sensor (shared across evaluations)
# ======================================================
SIMULATOR = AerSimulator()

SENSOR = Sensor(
    radar_noise_std={'range': 50.0, 'velocity': 5.0, 'azimuth': 0.005},
    irst_noise_std=0.1
)

# ======================================================
# Scenario Generator (moderate 2–4g style turns)
# ======================================================
def generate_scenario() -> dict:
    """
    Generate a turning scenario roughly similar to the
    engagement geometry used in your guidance demos.
    """
    # Turn between about 1.5g and 3.5g
    g_force = np.random.uniform(1.5, 3.5)
    maneuver = ('turn', {'g_force': g_force})

    # Position near 10–12 km range and 9–11 km altitude
    pos = [
        np.random.uniform(8000, 13000),
        np.random.uniform(-1500, 1500),
        np.random.uniform(9000, 11000)
    ]

    # Forward velocity 200–300 m/s with some lateral variation
    vx_sign = random.choice([-1, 1])
    vel = [
        vx_sign * np.random.uniform(200, 300),
        np.random.uniform(-30, 30),
        np.random.uniform(-15, 15)
    ]

    return {'maneuver': maneuver, 'pos': pos, 'vel': vel}

# ======================================================
# Helper: build *training* features matching QCTPF
# ======================================================
def build_training_features(target: Target) -> np.ndarray:
    """
    Build the same kind of feature vector QCTPF uses:

      [closing_velocity, azimuth, 0.0, 0.0, thermal_proxy]

    Here we assume ownship at the origin for training.
    """
    own = np.zeros(3)
    pos = target.position
    vel = target.velocity

    rel_pos = pos - own
    r = np.linalg.norm(rel_pos) + 1e-9

    azimuth = np.arctan2(rel_pos[1], rel_pos[0])
    closing_vel = -np.dot(vel, rel_pos) / r

    thermal_proxy = 1.0 / (r * r + 1e-6)

    features = np.array([
        closing_vel,
        azimuth,
        0.0,           # dummy accel_y (QCTPF also uses 0 here)
        0.0,           # dummy g-load
        thermal_proxy
    ], dtype=float)

    # Normalize
    norm = np.linalg.norm(features)
    return features / (norm + 1e-9)

# ======================================================
# Fitness Evaluation (heavy, high-quality Option B)
# ======================================================
def evaluate_fitness(population: Population):
    """
    Fitness:
      - multiple long episodes
      - acceleration + Δv error
      - input features aligned with QCTPF
    """
    sim_opts = SimOptions()
    N_EPISODES = 10        # heavier: more scenarios per genome
    num_steps = 40         # longer horizon
    shots = 64             # more samples → lower measurement noise

    # deterministic seeding per generation for fairness
    np.random.seed(sim_opts.seed + population.generation)
    random.seed(sim_opts.seed + population.generation)

    print(f"\n--- Evaluating Fitness for Generation {population.generation} ---")

    for genome in population.population:
        total_error = 0.0

        try:
            # Build circuit template once per genome
            qc_template, params = build_circuit_from_genome(genome)

            if len(qc_template) == 0 or len(params) == 0:
                genome.fitness = 1e-6
                continue

            qc_template.measure_all()
            transpiled_template = transpile(qc_template, SIMULATOR)

            for _ in range(N_EPISODES):
                scenario = generate_scenario()
                target = Target(scenario['pos'], scenario['vel'])

                last_velocity = np.array(target.velocity)

                for _ in range(num_steps):
                    # Advance target in its maneuver
                    target.update(sim_opts.dt, scenario['maneuver'])

                    # Build features matching QCTPF._build_features_for_particle
                    input_state = build_training_features(target)

                    # Pad to match number of parameters
                    if len(input_state) < len(params):
                        input_state = np.pad(
                            input_state,
                            (0, len(params) - len(input_state)),
                            mode='constant'
                        )

                    # Bind parameters
                    param_map = {p: v for p, v in zip(params, input_state)}
                    bound_qc = transpiled_template.assign_parameters(param_map)

                    # Run circuit
                    result = SIMULATOR.run(bound_qc, shots=shots).result()
                    counts = result.get_counts()

                    # Decode measurement into outputs
                    avg_values = np.zeros(genome.num_qubits)
                    for bitstring, count in counts.items():
                        bitstring = bitstring[::-1]  # Qiskit is little-endian
                        for i, bit in enumerate(bitstring):
                            if bit == '1':
                                avg_values[i] += count
                    avg_values /= shots

                    # Map first 6 qubits:
                    #   0:3 → acceleration
                    #   3:6 → velocity change
                    predicted_output = (avg_values[:6] - 0.5) * 30.0

                    # True dynamics
                    true_accel = target.get_state()[2]
                    true_velocity_change = np.array(target.velocity) - last_velocity

                    predicted_accel = predicted_output[0:3]
                    predicted_velocity_change = predicted_output[3:6]

                    # Errors
                    accel_error = np.linalg.norm(predicted_accel - true_accel)
                    vel_error = np.linalg.norm(
                        predicted_velocity_change - true_velocity_change
                    )

                    # Weighted combination (prioritize accel)
                    total_error += (0.7 * accel_error) + (0.3 * vel_error)

                    last_velocity = np.array(target.velocity)

            mean_error = total_error / (N_EPISODES * num_steps)
            genome.fitness = float(1.0 / (1.0 + mean_error))

        except Exception as e:
            print(f"[Fitness Error] {e}")
            genome.fitness = 1e-6

# ======================================================
# Main Training Loop (Option B: best performance)
# ======================================================
if __name__ == '__main__':
    qneat_opts = QNEATOptions()

    # Option B tweaks: bigger population for richer search
    # (you can comment these out if QNEATOptions is already tuned)
    try:
        qneat_opts.population_size = 220
        qneat_opts.compatibility_threshold = 2.5
    except Exception:
        # If QNEATOptions is not mutable this way, it's fine.
        pass

    NUM_GENERATIONS = 600                 # long run
    CHECKPOINT_INTERVAL = 1
    CHECKPOINT_PATH = "checkpoints/qneat_checkpoint.pkl"

    # Resume if checkpoint exists
    if os.path.exists(CHECKPOINT_PATH):
        population = Population.load_checkpoint(CHECKPOINT_PATH)
        print(f"Checkpoint loaded at generation {population.generation}.")
    else:
        # 8-qubit models are enough for 5 input features
        population = Population(num_qubits=8, options=qneat_opts)
        print("New population created.")

    for gen in range(population.generation, NUM_GENERATIONS):
        evaluate_fitness(population)
        population.run_evolutionary_cycle()

        # Save checkpoint + dashboard state
        if gen % CHECKPOINT_INTERVAL == 0:
            os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
            population.save_checkpoint(CHECKPOINT_PATH)
            with open("population_state.pkl", "wb") as f:
                pickle.dump(population, f)

        if population.population:
            best_genome = max(population.population, key=lambda g: g.fitness)
            print(f"Gen {gen} | Best Fitness: {best_genome.fitness:.4f}")

    # Save final champion genome
    if population.population:
        champion_genome = max(population.population, key=lambda g: g.fitness)
        with open("champion_genome.pkl", 'wb') as f:
            pickle.dump(champion_genome, f)
        print("\n--- Champion genome saved to 'champion_genome.pkl' ---")

    population.close_writer()
