import numpy as np
import pickle
import os
import random
from typing import List

# Qiskit Imports
from qiskit_aer import AerSimulator
from qiskit import transpile

# Project Imports
from Quantum_Core.qneat import Population, QNEATOptions, Genome
from Quantum_Core.nqpf import build_circuit_from_genome
from Simulation.target_dynamics import Target
from Simulation.sensor_model import Sensor
from config import SimOptions

# =====================================================================================
# Global Simulator (re-used for all circuits)
# =====================================================================================
SIMULATOR = AerSimulator()

# Simplified sensor for training QPF model
SENSOR = Sensor(
    radar_noise_std={'range': 50.0, 'velocity': 5.0, 'azimuth': 0.005},
    irst_noise_std=0.1
)


# =====================================================================================
# Generate moderate-maneuver random scenarios
# =====================================================================================
def generate_scenario() -> dict:
    g_force = np.random.uniform(1.5, 3.0)  # Moderate turning
    maneuver = ('turn', {'g_force': g_force})

    pos = [
        np.random.uniform(8000, 12000),
        np.random.uniform(-1000, 1000),
        np.random.uniform(9000, 11000)
    ]

    vx_sign = random.choice([-1, 1])
    vel = [
        vx_sign * np.random.uniform(200, 300),
        np.random.uniform(-20, 20),
        np.random.uniform(-20, 20)
    ]

    return {'maneuver': maneuver, 'pos': pos, 'vel': vel}


# =====================================================================================
# Evaluate Fitness for the Entire Population
# =====================================================================================
# def evaluate_fitness(population: Population):
#     sim_opts = SimOptions()
#     N_EPISODES = 5
#     np.random.seed(sim_opts.seed)
#     random.seed(sim_opts.seed)

#     print("\n--- Evaluating Fitness for Generation", population.generation, "---")

#     for genome in population.population:
#         total_error = 0.0
#         num_steps = 30

#         try:
#             qc_template, params = build_circuit_from_genome(genome)

#             if len(qc_template) == 0:
#                 genome.fitness = 1e-6
#                 continue

#             qc_template.measure_all()
#             transpiled_template = transpile(qc_template, SIMULATOR)

#             # Repeat on multiple scenarios for stability
#             for _ in range(N_EPISODES):
#                 scenario = generate_scenario()
#                 target = Target(scenario['pos'], scenario['vel'])

#                 for step in range(num_steps):
#                     target.update(sim_opts.dt, scenario['maneuver'])
#                     observation = SENSOR.observe(target, ownship_position=np.zeros(3))

#                     input_state = np.array([
#                         observation['closing_velocity'],
#                         observation['azimuth'],
#                         target.get_state()[2][1],
#                         np.linalg.norm(target.get_state()[2]) / 9.81,
#                         observation['thermal_intensity']
#                     ])

#                     norm = np.linalg.norm(input_state)
#                     norm_state = input_state / (norm + 1e-9)

#                     if len(norm_state) < len(params):
#                         norm_state = np.pad(norm_state, (0, len(params) - len(norm_state)))

#                     param_map = {p: val for p, val in zip(params, norm_state)}
#                     bound_qc = transpiled_template.assign_parameters(param_map)

#                     result = SIMULATOR.run(bound_qc, shots=32).result()
#                     counts = result.get_counts()

#                     avg_values = np.zeros(genome.num_qubits)
#                     for bitstring, count in counts.items():
#                         bitstring = bitstring[::-1]
#                         for i, bit in enumerate(bitstring):
#                             if bit == '1':
#                                 avg_values[i] += count
#                     avg_values /= 32

#                     predicted_output = (avg_values[:6] - 0.5) * 30.0

#                     # Error: ONLY acceleration for initial training success
#                     true_accel = target.get_state()[2]
#                     predicted_accel = predicted_output[0:3]
#                     accel_error = np.linalg.norm(predicted_accel - true_accel)

#                     total_error += accel_error

#             mean_error = total_error / (N_EPISODES * num_steps)
#             genome.fitness = float(1.0 / (1.0 + mean_error))

#         except Exception as e:
#             print(f"[Fitness Error] {e}")
#             genome.fitness = 1e-6

def evaluate_fitness(population: Population):
    sim_opts = SimOptions()
    N_EPISODES = 8            # more episodes → more stable fitness
    np.random.seed(sim_opts.seed)
    random.seed(sim_opts.seed)

    print(f"\n--- Evaluating Fitness for Generation {population.generation} (Phase 2) ---")

    for genome in population.population:
        total_error = 0.0
        num_steps = 30

        try:
            qc_template, params = build_circuit_from_genome(genome)

            if len(qc_template) == 0:
                genome.fitness = 1e-6
                continue

            qc_template.measure_all()
            transpiled_template = transpile(qc_template, SIMULATOR)

            for _ in range(N_EPISODES):
                # slightly wider, more varied scenario than phase 1
                scenario = generate_scenario()
                target = Target(scenario['pos'], scenario['vel'])

                last_velocity = np.array(target.velocity)

                for _ in range(num_steps):
                    target.update(sim_opts.dt, scenario['maneuver'])
                    observation = SENSOR.observe(target, ownship_position=np.zeros(3))

                    input_state = np.array([
                        observation['closing_velocity'],
                        observation['azimuth'],
                        target.get_state()[2][1],
                        np.linalg.norm(target.get_state()[2]) / 9.81,
                        observation['thermal_intensity']
                    ])

                    norm = np.linalg.norm(input_state)
                    norm_state = input_state / (norm + 1e-9)
                    if len(norm_state) < len(params):
                        norm_state = np.pad(norm_state, (0, len(params) - len(norm_state)))

                    param_map = {p: v for p, v in zip(params, norm_state)}
                    bound_qc = transpiled_template.assign_parameters(param_map)

                    result = SIMULATOR.run(bound_qc, shots=32).result()
                    counts = result.get_counts()

                    avg_values = np.zeros(genome.num_qubits)
                    for bitstring, count in counts.items():
                        bitstring = bitstring[::-1]
                        for i, bit in enumerate(bitstring):
                            if bit == '1':
                                avg_values[i] += count
                    avg_values /= 32

                    # keep same scaling for continuity
                    predicted_output = (avg_values[:6] - 0.5) * 30.0

                    # --- NEW: include velocity-change error as well ---
                    true_velocity_change = np.array(target.velocity) - last_velocity
                    predicted_velocity_change = predicted_output[3:6]
                    velocity_error = np.linalg.norm(
                        predicted_velocity_change - true_velocity_change
                    )

                    true_accel = target.get_state()[2]
                    predicted_accel = predicted_output[0:3]
                    accel_error = np.linalg.norm(predicted_accel - true_accel)

                    # weighted combo: still prioritize accel
                    total_error += (0.7 * accel_error) + (0.3 * velocity_error)

                    last_velocity = np.array(target.velocity)

            mean_error = total_error / (N_EPISODES * num_steps)
            genome.fitness = float(1.0 / (1.0 + mean_error))

        except Exception as e:
            print(f"[Fitness Error] {e}")
            genome.fitness = 1e-6


# =====================================================================================
# Main Training Loop
# =====================================================================================
if __name__ == '__main__':
    qneat_opts = QNEATOptions()
    NUM_GENERATIONS = 449
    CHECKPOINT_INTERVAL = 1
    CHECKPOINT_PATH = "checkpoints/qneat_checkpoint.pkl"

    if os.path.exists(CHECKPOINT_PATH):
        population = Population.load_checkpoint(CHECKPOINT_PATH)
        print("Checkpoint Loaded.")
    else:
        population = Population(num_qubits=8, options=qneat_opts)
        print("New Population Created.")

    for gen in range(population.generation, NUM_GENERATIONS):
        evaluate_fitness(population)
        population.run_evolutionary_cycle()

        # Save best genome for dashboard visualization
        if gen % CHECKPOINT_INTERVAL == 0:
            with open("population_state.pkl", "wb") as f:
                pickle.dump(population, f)
            population.save_checkpoint(CHECKPOINT_PATH)

        best_genome = max(population.population, key=lambda g: g.fitness)
        print(f"Gen {gen} | Best Fitness: {best_genome.fitness:.4f}")

    # Save final champion
    champion_genome = max(population.population, key=lambda g: g.fitness)
    with open("champion_genome.pkl", 'wb') as f:
        pickle.dump(champion_genome, f)

    print("\n--- Champion Genome Saved to 'champion_genome.pkl' ---")
    population.close_writer()
