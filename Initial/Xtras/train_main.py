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
from Analysis.plot_results import genome_to_image
from config import SimOptions

# --- Global objects for efficient fitness evaluation ---
def generate_random_maneuver() -> tuple:
    """
    Generates a random maneuver with realistic, randomized parameters.
    """
    maneuver_type = random.choice(['straight', 'turn', 'climb_dive', 'jink'])

    if maneuver_type == 'straight':
        return ('straight', {})
    
    elif maneuver_type == 'turn':
        # Generate a G-force between a gentle 2G turn and a hard 9G break.
        g_force = np.random.uniform(2.0, 9.0)
        return ('turn', {'g_force': g_force})
        
    elif maneuver_type == 'climb_dive':
        # Generate a vertical G-force between 1G and 4G, and randomly make it a climb or a dive.
        vertical_g = np.random.uniform(1.0, 4.0) * random.choice([-1, 1])
        return ('climb_dive', {'vertical_g': vertical_g})
        
    elif maneuver_type == 'jink':
        # Generate a realistic defensive jink.
        frequency = np.random.uniform(0.5, 2.0) # Oscillate between 0.5 and 2 times per second
        amplitude = np.random.uniform(20.0, 40.0) # Lateral acceleration (approx 2-4 Gs)
        return ('jink', {'frequency': frequency, 'amplitude': amplitude})

    return ('straight', {}) # Fallback
SIMULATOR = AerSimulator()
SENSOR = Sensor(
    radar_noise_std={'range': 50.0, 'velocity': 5.0, 'azimuth': 0.005},
    irst_noise_std=0.1
)

def evaluate_fitness(population: Population):
    """
    Evaluates fitness by running each genome against a short, RANDOMLY
    generated maneuver scenario using the correct, efficient Qiskit pattern.
    """
    print("--- Evaluating Fitness for Population ---")

    for genome in population.population:
        total_error = 0.0
        num_steps = 30
        
        try:
            # 1. Build the circuit template and get its parameters ONCE
            qc_template, params = build_circuit_from_genome(genome)
            # CORRECT
            if len(qc_template) == 0:
                genome.fitness = 1e-6; continue
            
            qc_template.measure_all()
            # 2. Transpile the template ONCE
            transpiled_template = transpile(qc_template, SIMULATOR)
            
            # --- Create a new, random test scenario ---
            maneuver_to_test = generate_random_maneuver()
            initial_pos = [np.random.uniform(5000, 15000), np.random.uniform(-2000, 2000), np.random.uniform(8000, 12000)]
            initial_vel = [np.random.uniform(-300, -200), 0, 0]
            target = Target(initial_position=initial_pos, initial_velocity=initial_vel)
            last_velocity = target.velocity.copy()

            # --- EFFICIENT SIMULATION LOOP ---
            for step in range(num_steps):
                target.update(0.1, maneuver_to_test)
                
                # Get sensor data
                observation = SENSOR.observe(target, ownship_position=np.zeros(3))
                input_state = np.array([
                    observation['closing_velocity'], observation['azimuth'],
                    target.get_state()[2][1], np.linalg.norm(target.get_state()[2]) / 9.81,
                    observation['thermal_intensity']
                ])

                # 3. FAST: Assign parameters to the transpiled template
                norm = np.linalg.norm(input_state)
                norm_state = input_state / (norm + 1e-9)
                norm = np.linalg.norm(input_state)
                norm_state = input_state / (norm + 1e-9)

                # --- THIS IS THE CRITICAL FIX ---
                # Pad the normalized state with zeros if it's shorter than the number of params.
                if len(norm_state) < len(params):
                    padding = np.zeros(len(params) - len(norm_state))
                    norm_state = np.concatenate([norm_state, padding])
                
                param_map = {p: val for p, val in zip(params, norm_state)}
                
                # This creates a new, fully bound circuit without re-transpiling
                bound_qc = transpiled_template.assign_parameters(param_map)
                
                # 4. Execute
                result = SIMULATOR.run(bound_qc, shots=100).result()
                counts = result.get_counts()

                # Decode
                avg_values = np.zeros(genome.num_qubits)
                for bitstring, count in counts.items():
                    bitstring = bitstring[::-1]
                    for i, bit in enumerate(bitstring):
                        if bit == '1': avg_values[i] += count
                avg_values /= 100
                predicted_output = (avg_values[:6] - 0.5) * 0.1

                # Calculate error
                true_velocity_change = target.velocity - last_velocity
                predicted_velocity_change = predicted_output[3:6]
                total_error += np.linalg.norm(predicted_velocity_change - true_velocity_change)
                last_velocity = target.velocity.copy()

            genome.fitness = float(1.0 / ((total_error / num_steps) + 1e-6))

        except Exception as e:
            print(f"Error evaluating genome: {e}")
            genome.fitness = 1e-6

if __name__ == '__main__':
    qneat_opts = QNEATOptions()
    NUM_GENERATIONS = 200
    CHECKPOINT_INTERVAL = 25
    CHECKPOINT_PATH = "checkpoints/qneat_checkpoint.pkl"

    if os.path.exists(CHECKPOINT_PATH):
        population = Population.load_checkpoint(CHECKPOINT_PATH)
    else:
        population = Population(num_qubits=16, options=qneat_opts)

    for gen in range(population.generation, NUM_GENERATIONS):
        print(f"\n--- Starting Generation {gen} ---")
        
        evaluate_fitness(population)
        population.run_evolutionary_cycle()
        
        if population.population:
            best_genome = max(population.population, key=lambda g: g.fitness)
            best_genome_image = genome_to_image(best_genome)
            population.writer.add_image(
                'Best_Genome_Architecture', 
                best_genome_image, 
                global_step=population.generation, 
                dataformats='HWC'
            )
        
        if gen % CHECKPOINT_INTERVAL == 0:
            population.save_checkpoint(CHECKPOINT_PATH)
            # CORRECTED: Save the dashboard state file ONLY with the checkpoint
            with open("population_state.pkl", "wb") as f:
                pickle.dump(population, f)

    if population.population:
        print("--- Training Complete. Saving champion genome. ---")
        champion_genome = max(population.population, key=lambda g: g.fitness)
        with open("champion_genome.pkl", 'wb') as f:
            pickle.dump(champion_genome, f)

    population.close_writer()
    print("--- Champion genome saved to champion_genome.pkl ---")