# train_qoder.py
# -------------------------------------------------------
# Code-2: QODER / Quantum NEAT training with curriculum.
#
# Curriculum:
#   - Early gens: moderate 1.5–3g turns (easier).
#   - Mid gens: mixed turns + climb/dive + jink.
#   - Late gens: high-g 4–9g + aggressive jinking.
#
# Input features match QCTPF:
#   [closing_velocity, azimuth, 0.0, 0.0, thermal_proxy]
#
# Output:
#   first 3 qubits -> acceleration
#   next 3 qubits  -> velocity change (Δv)
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
# Global config
# ======================================================
SIMULATOR = AerSimulator()
SENSOR = Sensor(
    radar_noise_std={'range': 50.0, 'velocity': 5.0, 'azimuth': 0.005},
    irst_noise_std=0.1
)

TRAIN_NUM_GENERATIONS = 600  # curriculum is based on this


# ======================================================
# Curriculum-based scenario generator
# ======================================================
def generate_scenario_for_gen(gen: int) -> dict:
    """
    Generate a scenario whose difficulty depends on the generation index.
    - 0%–30% of training: easier, moderate turning (1.5–3g).
    - 30%–70%: mixed turns + climb/dive + jink with 2–5g effective loads.
    - 70%–100%: high-g 4–9g turns and aggressive jink.
    """
    phase = gen / max(1, TRAIN_NUM_GENERATIONS)  # in [0,1]

    # Common position envelope: around 8–13 km away, 9–11 km altitude
    base_pos = [
        np.random.uniform(8000, 13000),
        np.random.uniform(-2000, 2000),
        np.random.uniform(9000, 11000)
    ]

    # Baseline forward speed: 200–300 m/s
    vx_sign = random.choice([-1, 1])
    base_vel = [
        vx_sign * np.random.uniform(200, 300),
        np.random.uniform(-40, 40),
        np.random.uniform(-20, 20)
    ]

    # ----- Phase 1: Easy / moderate turns -----
    if phase < 0.3:
        g_force = np.random.uniform(1.5, 3.0)
        maneuver = ('turn', {'g_force': g_force})

    # ----- Phase 2: Mixed moderate + vertical + jink -----
    elif phase < 0.7:
        mode = random.choice(['turn', 'climb_dive', 'jink'])

        if mode == 'turn':
            g_force = np.random.uniform(2.0, 5.0)
            maneuver = ('turn', {'g_force': g_force})

        elif mode == 'climb_dive':
            vertical_g = np.random.uniform(1.0, 4.0) * random.choice([-1, 1])
            maneuver = ('climb_dive', {'vertical_g': vertical_g})

        else:  # 'jink'
            frequency = np.random.uniform(0.5, 2.0)   # left-right per second
            amplitude = np.random.uniform(20.0, 40.0) # lateral accel ~2–4g
            maneuver = ('jink', {'frequency': frequency, 'amplitude': amplitude})

    # ----- Phase 3: Hard / high-g evasive -----
    else:
        mode = random.choice(['turn', 'turn', 'jink', 'climb_dive'])  # bias toward turns/jink

        if mode == 'turn':
            g_force = np.random.uniform(4.0, 9.0)   # 4–9g break turns
            maneuver = ('turn', {'g_force': g_force})

        elif mode == 'climb_dive':
            vertical_g = np.random.uniform(2.0, 6.0) * random.choice([-1, 1])
            maneuver = ('climb_dive', {'vertical_g': vertical_g})

        else:  # 'jink'
            frequency = np.random.uniform(1.0, 3.0)   # more aggressive jink
            amplitude = np.random.uniform(30.0, 60.0) # stronger lateral accel
            maneuver = ('jink', {'frequency': frequency, 'amplitude': amplitude})

    return {
        'maneuver': maneuver,
        'pos': base_pos,
        'vel': base_vel
    }


# ======================================================
# Features matching QCTPF
# ======================================================
def build_training_features(target: Target) -> np.ndarray:
    """
    Build the same kind of feature vector QCTPF uses:

      [closing_velocity, azimuth, 0.0, 0.0, thermal_proxy]

    Ownship is at origin for training.
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
        0.0,          # dummy accel_y
        0.0,          # dummy g-load
        thermal_proxy
    ], dtype=float)

    norm = np.linalg.norm(features)
    return features / (norm + 1e-9)


# ======================================================
# Fitness Evaluation (with curriculum)
# ======================================================
def evaluate_fitness(population: Population):
    """
    Heavy, high-quality fitness with curriculum:
    - More episodes, longer horizon.
    - Input features match QCTPF.
    - Error = 0.7 * accel_error + 0.3 * Δv_error.
    """
    sim_opts = SimOptions()

    N_EPISODES = 10
    num_steps = 40
    shots = 64

    np.random.seed(sim_opts.seed + population.generation)
    random.seed(sim_opts.seed + population.generation)

    print(f"\n--- Evaluating Fitness for Generation {population.generation} ---")

    for genome in population.population:
        total_error = 0.0

        try:
            qc_template, params = build_circuit_from_genome(genome)

            if len(qc_template) == 0 or len(params) == 0:
                genome.fitness = 1e-6
                continue

            qc_template.measure_all()
            transpiled_template = transpile(qc_template, SIMULATOR)

            for _ in range(N_EPISODES):
                scenario = generate_scenario_for_gen(population.generation)
                target = Target(scenario['pos'], scenario['vel'])

                last_velocity = np.array(target.velocity)

                for _ in range(num_steps):
                    # Advance target
                    target.update(sim_opts.dt, scenario['maneuver'])

                    # Features for this state
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

                    avg_values = np.zeros(genome.num_qubits)
                    for bitstring, count in counts.items():
                        bitstring = bitstring[::-1]
                        for i, bit in enumerate(bitstring):
                            if bit == '1':
                                avg_values[i] += count
                    avg_values /= shots

                    # Map outputs:
                    #   0:3 -> acceleration
                    #   3:6 -> velocity change
                    predicted_output = (avg_values[:6] - 0.5) * 30.0

                    true_accel = target.get_state()[2]
                    true_dv = np.array(target.velocity) - last_velocity

                    pred_accel = predicted_output[0:3]
                    pred_dv = predicted_output[3:6]

                    accel_error = np.linalg.norm(pred_accel - true_accel)
                    dv_error = np.linalg.norm(pred_dv - true_dv)

                    total_error += (0.7 * accel_error) + (0.3 * dv_error)

                    last_velocity = np.array(target.velocity)

            mean_error = total_error / (N_EPISODES * num_steps)
            genome.fitness = float(1.0 / (1.0 + mean_error))

        except Exception as e:
            print(f"[Fitness Error] {e}")
            genome.fitness = 1e-6


# ======================================================
# Main Training Loop
# ======================================================
if __name__ == '__main__':
    qneat_opts = QNEATOptions()

    # Optionally adjust QNEAT settings for heavier search
    try:
        qneat_opts.population_size = 220
        qneat_opts.compatibility_threshold = 2.5
    except Exception:
        pass

    NUM_GENERATIONS = TRAIN_NUM_GENERATIONS
    CHECKPOINT_INTERVAL = 1
    CHECKPOINT_PATH = "checkpoints/qneat_checkpoint.pkl"

    if os.path.exists(CHECKPOINT_PATH):
        population = Population.load_checkpoint(CHECKPOINT_PATH)
        print(f"Checkpoint loaded at generation {population.generation}.")
    else:
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

    if population.population:
        champion_genome = max(population.population, key=lambda g: g.fitness)
        with open("champion_genome.pkl", 'wb') as f:
            pickle.dump(champion_genome, f)
        print("\n--- Champion genome saved to 'champion_genome.pkl' ---")

    population.close_writer()
