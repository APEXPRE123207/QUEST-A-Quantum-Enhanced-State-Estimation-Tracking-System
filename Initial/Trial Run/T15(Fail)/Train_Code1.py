import numpy as np
import pickle
import os
import random
import multiprocessing
from typing import List
import pickle
# Qiskit Imports
from qiskit_aer import AerSimulator
from qiskit import transpile

# Project Imports
from Quantum_Core.qneat import Population, QNEATOptions, Genome
from Quantum_Core.nqpf import build_circuit_from_genome
from Simulation.target_dynamics import Target
from Simulation.sensor_model import Sensor
from Analysis.plot_results import genome_to_image
from config import SimOptions, SensorOptions

with open("champion_genome.pkl", "rb") as f:
    pretrained_genome = pickle.load(f)
# --- Helper Functions ---
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

def generate_scenarios(num_episodes: int) -> List[dict]:
    """Generates a fixed list of scenarios for a generation."""
    scenarios = []
    for _ in range(num_episodes):
        maneuver = generate_random_maneuver()
        pos = [np.random.uniform(5000, 15000), np.random.uniform(-2000, 2000), np.random.uniform(8000, 12000)]
        vx_sign = random.choice([-1, 1])
        vel = [vx_sign * np.random.uniform(200, 300), np.random.uniform(-50, 50), np.random.uniform(-20, 20)]
        scenarios.append({'maneuver': maneuver, 'pos': pos, 'vel': vel})
    return scenarios

def evaluate_single_genome(args) -> float:
    """
    Evaluates a single genome against a fixed set of scenarios.
    Args:
        args: Tuple containing (genome, scenarios)
    """
    genome, scenarios = args
    
    # Re-initialize simulator and sensor per process
    simulator = AerSimulator()
    sim_opts = SimOptions()
    
    # Seed randomness for reproducibility
    random.seed(os.getpid() + int(id(genome) % 10000))
    np.random.seed(os.getpid() + int(id(genome) % 10000))

    num_steps = sim_opts.eval_steps
    cumulative_episode_error = 0.0
    
    try:
        # 1. Build circuit ONCE
        qc_template, params = build_circuit_from_genome(genome)
        if len(qc_template) == 0: return 1e-6
        qc_template.measure_all()
        transpiled_template = transpile(qc_template, simulator)
        
        for scenario in scenarios:
            # Use the fixed scenario parameters
            target = Target(initial_position=scenario['pos'], initial_velocity=scenario['vel'])
            last_velocity = target.velocity.copy()
            episode_error = 0.0
            
            # --- SIMULATION LOOP ---
            for step in range(num_steps):
                target.update(sim_opts.dt, scenario['maneuver'])
                
                # Input State (Position + Velocity)
                current_state_vector = np.concatenate([target.position, target.velocity])
                input_state = current_state_vector
    
                # Normalize
                norm = np.linalg.norm(input_state)
                norm_state = input_state / (norm + 1e-9)
    
                # Pad
                if len(norm_state) < len(params):
                    padding = np.zeros(len(params) - len(norm_state))
                    norm_state = np.concatenate([norm_state, padding])
                
                # Bind Parameters
                param_map = {}
                for p in transpiled_template.parameters:
                    try:
                        idx = int(p.name.split('_')[1])
                        if idx < len(norm_state):
                            param_map[p] = norm_state[idx]
                    except: pass
                
                bound_qc = transpiled_template.assign_parameters(param_map)
                
                # Execute
                result = simulator.run(bound_qc, shots=sim_opts.training_shots).result()
                counts = result.get_counts()
    
                # Decode
                avg_values = np.zeros(genome.num_qubits)
                for bitstring, count in counts.items():
                    bitstring = bitstring[::-1]
                    for i, bit in enumerate(bitstring):
                        if bit == '1': avg_values[i] += count
                avg_values /= sim_opts.training_shots
                predicted_output = (avg_values[:6] - 0.5) * 50.0
    
                # Calculate Error
                true_velocity_change = target.velocity - last_velocity
                predicted_velocity_change = predicted_output[3:6]
                velocity_error = np.linalg.norm(predicted_velocity_change - true_velocity_change)
    
                true_accel = target.get_state()[2]
                predicted_accel = predicted_output[0:3]
                accel_error = np.linalg.norm(predicted_accel - true_accel)
                
                episode_error += (0.4 * velocity_error) + (0.6 * accel_error)
                last_velocity = target.velocity.copy()
            
            cumulative_episode_error += (episode_error / num_steps)

        mean_error = cumulative_episode_error / len(scenarios)
        return float(1.0 / (1.0 + mean_error))

    except Exception as e:
        print(f"[Genome eval error] {e}") 
        return 1e-6

def evaluate_fitness(population: Population):
    """
    Evaluates fitness using fixed scenarios for fairness.
    """
    try:
        cpu_count = multiprocessing.cpu_count()
        num_processes = max(1, min(4, cpu_count // 2))
    except NotImplementedError:
        num_processes = 2
        
    print(f"--- Starting Parallel Evaluation with {num_processes} processes ---")
    
    # Generate fixed scenarios for this generation
    N_EPISODES = 5
    scenarios = generate_scenarios(N_EPISODES)
    
    # Prepare arguments: each genome gets the SAME list of scenarios
    args = [(genome, scenarios) for genome in population.population]
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        fitness_scores = pool.map(evaluate_single_genome, args)
    
    for genome, fitness in zip(population.population, fitness_scores):
        genome.fitness = fitness

if __name__ == '__main__':
    # Windows support for multiprocessing
    multiprocessing.freeze_support()
    
    qneat_opts = QNEATOptions()
    NUM_GENERATIONS = 400
    CHECKPOINT_INTERVAL = 1
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
            # Save the dashboard state file
            with open("population_state.pkl", "wb") as f:
                pickle.dump(population, f)

    if population.population:
        print("--- Training Complete. Saving champion genome. ---")
        champion_genome = max(population.population, key=lambda g: g.fitness)
        with open("champion_genome_updated_config.pkl", 'wb') as f:
            pickle.dump(champion_genome, f)

    population.close_writer()
    print("--- Champion genome saved to champion_genome_updated_config.pkl ---")