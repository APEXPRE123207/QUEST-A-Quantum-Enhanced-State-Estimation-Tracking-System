import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from typing import Dict, Any, List, Tuple

# Assuming imports from our project
from .qneat import Gene, Genome
from Simulation.target_dynamics import Target 

def build_circuit_from_genome(genome: Genome) -> QuantumCircuit:
    """
    Dynamically builds a Qiskit QuantumCircuit from a given Genome object.

    This function iterates through the sorted genes of a genome and appends the
    corresponding Qiskit gates to a new quantum circuit.

    Args:
        genome: The Genome object to be translated.

    Returns:
        An executable Qiskit QuantumCircuit object.
    """
    # Create a new quantum circuit with the number of qubits defined in the genome.
    qc = QuantumCircuit(genome.num_qubits)

    # Define a mapping from our string representation to actual Qiskit circuit methods.
    # This makes the code clean and easily extensible.
    gate_map = {
        'h': qc.h,
        'x': qc.x,
        'y': qc.y,
        'z': qc.z,
        'rx': qc.rx,
        'ry': qc.ry,
        'rz': qc.rz,
        'cnot': qc.cx, # Note: Qiskit's CNOT is called cx
        # Add other gates as needed, e.g., 'cz': qc.cz
    }

    # Iterate through the genes in the genome (they should be sorted by innovation number)
    for gene in genome.genes:
        gate_function = gate_map.get(gene.gate_type.lower())
        
        if not gate_function:
            raise ValueError(f"Unknown gate type '{gene.gate_type}' in genome.")
            
        # Apply the gate to the circuit
        if gene.parameters:
            # For parametric gates like RX, RY, RZ
            # Assumes a single parameter per gate for simplicity
            gate_function(gene.parameters[0], *gene.target_qubits)
        else:
            # For non-parametric gates like H, X, CNOT
            gate_function(*gene.target_qubits)
            
    return qc

class NQPF:
    """
    Implements the Neuroevolutionary Quantum Particle Filter (NQPF).

    This class manages a swarm of particles and uses a QNEAT-evolved VQC
    as a quantum motion model to predict the state of a dynamic system.
    """
    def __init__(self, num_particles: int, trained_genome: Genome):
        """
        Initializes the particle filter.

        Args:
            num_particles: The number of particles in the swarm.
            trained_genome: The "champion" Genome from the QNEAT training process.
        """
        self.num_particles = num_particles
        self.trained_genome = trained_genome
        
        # Particles are stored as an array of states [N, state_dim]
        # For our target, state_dim is 6 (x, y, z, vx, vy, vz)
        self.particles = np.zeros((num_particles, 6)) 
        self.weights = np.ones(num_particles) / num_particles

        # This will be our executable quantum model
        self._template_circuit = build_circuit_from_genome(trained_genome)
        self.backend = AerSimulator()

    def initialize_swarm(self, initial_state_estimate: np.ndarray, initial_uncertainty: float):
        """
        Initializes the particle swarm around an initial state estimate.

        Args:
            initial_state_estimate: An array representing the initial guess of the target's state.
            initial_uncertainty: The standard deviation of the initial particle distribution.
        """
        self.particles = np.random.normal(loc=initial_state_estimate, 
                                          scale=initial_uncertainty, 
                                          size=(self.num_particles, 6))

    def _encode_state_to_circuit(self, circuit_template: QuantumCircuit, particle_state: np.ndarray) -> QuantumCircuit:
        """
        Encodes a classical particle state into the parameters of the VQC.
        
        NOTE: This is a simple encoding scheme. A production system would use a
              more sophisticated feature map.
        """
        # Create a fresh copy of the circuit to parameterize
        parameterized_circuit = circuit_template.copy()
        
        # Normalize state for encoding as rotation angles (simple example)
        norm_state = particle_state / np.linalg.norm(particle_state)
        
        param_idx = 0
        for i, gate in enumerate(self.trained_genome.genes):
            if gate.gate_type in ['rx', 'ry', 'rz']:
                # Assign a normalized state value to each parametric gate
                if param_idx < len(norm_state):
                    # Re-bind the parameter of the gate in the circuit
                    parameterized_circuit.data[i][0].params[0] = norm_state[param_idx]
                    param_idx += 1
        
        return parameterized_circuit

    def _decode_measurements(self, counts: Dict[str, int], shots: int) -> np.ndarray:
        """
        Decodes the measurement results from the quantum circuit into a classical state update.
        
        NOTE: This is a simple decoding scheme. A production system would have a
              scheme trained to map bitstrings to physical state changes.
        """
        # Example: Calculate the average measured bit value for each qubit
        num_qubits = self._template_circuit.num_qubits
        avg_values = np.zeros(num_qubits)
        
        for bitstring, count in counts.items():
            for i, bit in enumerate(bitstring):
                if bit == '1':
                    avg_values[i] += count
        
        avg_values /= shots
        
        # Simple mapping: Use the first 6 average values as a state update vector
        # This assumes the VQC was trained to produce such an output.
        state_update = (avg_values[:6] - 0.5) * 0.1 # Scale and center the update
        return state_update


    def predict(self, dt: float, shots: int = 1024) -> None:
        """
        Predicts the next state of all particles using the quantum motion model.
        """
        new_particles = np.zeros_like(self.particles)
        
        # --- Prepare all circuits for execution ---
        circuits_to_run = []
        for i in range(self.num_particles):
            # 1. Encode
            particle_state = self.particles[i]
            parameterized_circuit = self._encode_state_to_circuit(self._template_circuit, particle_state)
            parameterized_circuit.measure_all()
            circuits_to_run.append(parameterized_circuit)
            
        # --- Execute all circuits in a single batch job ---
        transpiled_circuits = transpile(circuits_to_run, self.backend)
        job = self.backend.run(transpiled_circuits, shots=shots)
        result = job.result()
        
        # --- Decode results for each particle ---
        for i in range(self.num_particles):
            counts = result.get_counts(i)
            # 2. Decode
            state_update_vector = self._decode_measurements(counts, shots)
            
            # 3. Update particle state
            # New state is a combination of simple physics and the quantum model's prediction
            physics_update = self.particles[i, 3:6] * dt
            new_particles[i] = self.particles[i]
            new_particles[i, 0:3] += physics_update
            new_particles[i] += state_update_vector

        self.particles = new_particles


    def update(self, measurement: Dict[str, float]) -> None:
        """
        Updates the weights of the particles based on a new sensor measurement.
        
        Args:
            measurement: A dictionary of sensor readings (e.g., from our Sensor class).
        """
        # This function calculates the likelihood of each particle given the measurement.
        # For simplicity, we'll use a basic distance metric.
        # A real implementation would use a proper likelihood function.
        
        # Example using range (assuming measurement['range'] is available)
        # and ownship is at [0,0,0]
        particle_ranges = np.linalg.norm(self.particles[:, 0:3], axis=1)
        errors = np.abs(particle_ranges - measurement.get('range', 0))
        
        # Convert errors to likelihoods (smaller error = higher likelihood)
        likelihoods = np.exp(-0.5 * (errors ** 2) / (50.0 ** 2)) # Assume 50m sensor error std dev
        
        self.weights *= likelihoods
        self.weights += 1e-300 # Avoid division by zero
        self.weights /= np.sum(self.weights) # Normalize

    def resample(self) -> None:
        """
        Resamples the particle swarm based on their weights.
        Low-weight particles are discarded, high-weight particles are duplicated.
        """
        # Systematic resampling: an efficient, low-variance method
        positions = (np.arange(self.num_particles) + np.random.rand()) / self.num_particles
        indexes = np.zeros(self.num_particles, 'i')
        cumulative_sum = np.cumsum(self.weights)
        
        i, j = 0, 0
        while i < self.num_particles:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
                
        self.particles = self.particles[indexes]
        self.weights.fill(1.0 / self.num_particles)