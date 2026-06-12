import numpy as np
import pickle
import os

from qiskit_aer import AerSimulator
from qiskit import transpile

from Quantum_Core.qneat import Population, QNEATOptions
from Quantum_Core.nqpf_placeholder import build_circuit_from_genome
from config import SimOptions
from Analysis.plot_results import genome_to_image
# --- Global objects for the fitness evaluation ---
SIMULATOR = AerSimulator()
SIM_OPTS = SimOptions()

def evaluate_fitness(population: Population):
    """
    Evaluates the fitness of every genome in the population.
    This is a simplified fitness function for demonstration.
    """
    print("--- Evaluating Fitness for Population ---")
    for genome in population.population:
        # Simple test: Can the circuit predict a simple state change?
        # We create a simple input state and a simple target output.
        input_state = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        target_output = np.array([0.6, 0.5, 0.4, 0.3, 0.2, 0.1]) # Example target

        try:
            # Build and execute the circuit for this genome
            qc = build_circuit_from_genome(genome)
            qc.measure_all()
            
            # This is a highly simplified encoding/decoding for fitness
            param_idx = 0
            for i, gate in enumerate(genome.genes):
                if gate.gate_type in ['rx', 'ry', 'rz'] and param_idx < len(input_state):
                    qc.data[i][0].params[0] = input_state[param_idx]
                    param_idx += 1

            transpiled_qc = transpile(qc, SIMULATOR)
            result = SIMULATOR.run(transpiled_qc, shots=100).result()
            counts = result.get_counts()
            
            # Simple decoding
            avg_values = np.zeros(genome.num_qubits)
            for bitstring, count in counts.items():
                for i, bit in enumerate(bitstring):
                    if bit == '1': avg_values[i] += count
            avg_values /= 100
            predicted_output = (avg_values[:6] - 0.5) * 0.1
            
            # Fitness is the inverse of the error (higher is better)
            error = np.linalg.norm(predicted_output - target_output)
            genome.fitness = float(1.0 / (error + 1e-6))

        except Exception as e:
            # Assign a very low fitness if the circuit fails to build or run
            print(f"Error evaluating genome: {e}")
            genome.fitness = 1e-6


if __name__ == '__main__':
    # --- Configuration ---
    qneat_opts = QNEATOptions()
    NUM_GENERATIONS = 200 # Set the total number of generations to run
    CHECKPOINT_INTERVAL = 25 # Save progress every 25 generations
    CHECKPOINT_PATH = "W:/CODE/Somehting/6_Missile/checkpoints/qneat_checkpoint.pkl"

    # --- Initialize or Load Population ---
    if os.path.exists(CHECKPOINT_PATH):
        population = Population.load_checkpoint(CHECKPOINT_PATH)
    else:
        population = Population(num_qubits=16, options=qneat_opts)

    # --- Main Training Loop ---
    for gen in range(population.generation, NUM_GENERATIONS):
        print(f"\n--- Starting Generation {gen} ---")
        
        # 1. Evaluate the fitness of each genome
        evaluate_fitness(population)
        
        # 2. Run the evolutionary cycle (speciation, crossover, mutation)
        population.run_evolutionary_cycle()
        best_genome = max(population.population, key=lambda g: g.fitness)
        best_genome_image = genome_to_image(best_genome)
        population.writer.add_image(
            'Best_Genome_Architecture', 
            best_genome_image, 
            global_step=population.generation, 
            dataformats='HWC'
        )
        # 3. Save a checkpoint periodically
        if gen % CHECKPOINT_INTERVAL == 0:
            population.save_checkpoint(CHECKPOINT_PATH)
        with open("population_state.pkl", "wb") as f:
            pickle.dump(population, f)

    # --- Save the final champion genome ---
    print("--- Training Complete. Saving champion genome. ---")
    champion_genome = max(population.population, key=lambda g: g.fitness)
    with open("champion_genome.pkl", 'wb') as f:
        pickle.dump(champion_genome, f)

    population.close_writer()
    print("--- Champion genome saved to champion_genome.pkl ---")