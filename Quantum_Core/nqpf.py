import numpy as np
from typing import Tuple, Dict, Any, List

# Qiskit Imports
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit import transpile
from qiskit_aer import AerSimulator

# Project Imports
from .qneat import Genome

def build_circuit_from_genome(genome: Genome) -> Tuple[QuantumCircuit, List[Parameter]]:
    """
    Dynamically builds a Qiskit QuantumCircuit template from a Genome.
    """
    qc = QuantumCircuit(genome.num_qubits)
    params = []
    gate_map = {'h': qc.h, 'x': qc.x, 'cnot': qc.cx}

    for gene in genome.genes:
        if gene.gate_type in ['rx', 'ry', 'rz']:
            param = Parameter(f'p_{len(params)}')
            params.append(param)
            qubit_index = gene.target_qubits[0]
            if gene.gate_type == 'rx': qc.rx(param, qubit_index)
            elif gene.gate_type == 'ry': qc.ry(param, qubit_index)
            elif gene.gate_type == 'rz': qc.rz(param, qubit_index)
        else:
            gate_function = gate_map.get(gene.gate_type.lower())
            if not gate_function:
                raise ValueError(f"Unknown non-parametric gate type '{gene.gate_type}' in genome.")
            gate_function(*gene.target_qubits)
            
    return qc, params

class NQPF:
    """Implements the Neuroevolutionary Quantum Particle Filter (NQPF)."""
    def __init__(self, num_particles: int, trained_genome: Genome):
        self.num_particles = num_particles
        self.trained_genome = trained_genome
        self.particles = np.zeros((num_particles, 6))
        self.weights = np.ones(num_particles) / num_particles
        
        self._template_circuit, self._params = build_circuit_from_genome(trained_genome)
        self.backend = AerSimulator()
        
        # Use .ops for modern Qiskit to check if circuit is empty
        if len(self._template_circuit) > 0:
            self._template_circuit.measure_all()
            self._transpiled_template = transpile(self._template_circuit, self.backend)
        else:
            self._transpiled_template = self._template_circuit

    def effective_sample_size(self) -> float:
        return 1.0 / (np.sum(self.weights ** 2) + 1e-12)

    def initialize_swarm(self, initial_state_estimate: np.ndarray, initial_uncertainty: float):
        self.particles = np.random.normal(loc=initial_state_estimate, scale=initial_uncertainty, size=(self.num_particles, 6))

    # WRONG QUANTUM PREDICT FUNCTION
    # def predict(self, dt: float, shots: int = 1024) -> None:
    #     if not self._params:
    #         for i in range(self.num_particles):
    #             self.particles[i, 0:3] += self.particles[i, 3:6] * dt
    #         return

    #     param_maps = []
    #     for i in range(self.num_particles):
    #         particle_state = self.particles[i]
    #         norm = np.linalg.norm(particle_state)
    #         norm_state = particle_state / (norm + 1e-9)
    #         if len(norm_state) < len(self._params):
    #             padding = np.zeros(len(self._params) - len(norm_state))
    #             norm_state = np.concatenate([norm_state, padding])
    #         param_map = {p: val for p, val in zip(self._params, norm_state)}
    #         param_maps.append(param_map)
        
    #     bound_circuits = [self._transpiled_template.assign_parameters(pm) for pm in param_maps]
    #     result = self.backend.run(bound_circuits, shots=shots).result()
        
    #     new_particles = np.zeros_like(self.particles)
    #     for i in range(self.num_particles):
    #         counts = result.get_counts(i)
    #         mean_update, cov_update = self._decode_measurements_to_gaussian(counts, shots)
    #         quantum_predicted_update = np.random.multivariate_normal(mean_update, cov_update)
    #         physics_update = self.particles[i, 3:6] * dt
            
    #         new_particles[i] = self.particles[i]
    #         new_particles[i, 0:3] += physics_update
    #         new_particles[i] += quantum_predicted_update
    #     self.particles = new_particles

    #NORMAL PHYSICS PREDICT FUNCTION
    # def predict(self, dt: float, shots: int = 1024) -> None:
    # # Simple kinematic model with Gaussian process noise.
    # # Ignore quantum circuit for now, just to debug PF.
    #     pos_noise_std = 30.0   # meters per step
    #     vel_noise_std = 3.0    # m/s per step

    #     for i in range(self.num_particles):
    #         # position update: x += v*dt + noise
    #         self.particles[i, 0:3] += self.particles[i, 3:6] * dt \
    #             + np.random.normal(0.0, pos_noise_std, 3)

    #         # velocity random walk
    #         self.particles[i, 3:6] += np.random.normal(0.0, vel_noise_std, 3)

    # def predict(self, dt: float, shots: int = 64) -> None:
    #     """
    #     Classical PF predict step with process noise:
    #     x_{k+1} = x_k + v_k * dt + position_noise
    #     v_{k+1} = v_k + velocity_noise

    #     This lets particles bend and follow turning motion.
    #     """
    #     # Tune these if needed
    #     pos_noise_std = 2.0   # meters per step
    #     vel_noise_std = 1.5   # m/s per step

    #     for i in range(self.num_particles):
    #         # Position update: x += v*dt + noise
    #         self.particles[i, 0:3] += (
    #             self.particles[i, 3:6] * dt
    #             + np.random.normal(0.0, pos_noise_std, 3)
    #         )

    #         # Velocity random walk (lets us approximate turning)
    #         self.particles[i, 3:6] += np.random.normal(0.0, vel_noise_std, 3)

    def predict(self, dt: float) -> None:
        """
        PF predict step with heading adjustment:
        Steers particles slightly toward sensor direction.
        """

        pos_noise_std = 3.0   # keep small
        vel_noise_std = 0.5   # small random walk

        for i in range(self.num_particles):
            # Update position
            self.particles[i, 0:3] += (
                self.particles[i, 3:6] * dt +
                np.random.normal(0.0, pos_noise_std, 3)
            )
            
            # Small velocity random walk
            self.particles[i, 3:6] += np.random.normal(0.0, vel_noise_std, 3)

        # 🧠 NEW: Steer particles using azimuth information (weakly)
        # Estimate current azimuth of each particle
        # Note: ownship assumed at origin for now
        rel_pos = self.particles[:, 0:3]
        az_pred = np.arctan2(rel_pos[:,1], rel_pos[:,0])

        # Innovation = predicted - weighted-mean azimuth
        mean_az = np.mean(az_pred)
        innovation = mean_az - az_pred

        # Apply small turn = adjust velocity direction
        turn_gain = 0.05  # 5% correction toward correct heading
        for i in range(self.num_particles):
            v = self.particles[i,3:6]
            speed = np.linalg.norm(v)
            if speed > 1e-3:
                heading = np.arctan2(v[1], v[0])
                heading += turn_gain * innovation[i]   # steer a little
                self.particles[i,3] = speed * np.cos(heading)
                self.particles[i,4] = speed * np.sin(heading)

                
    def _decode_measurements_to_gaussian(self, counts: Dict[str, int], shots: int) -> Tuple[np.ndarray, np.ndarray]:
        """Decodes measurement results into parameters for a Gaussian distribution."""
        num_qubits = self._template_circuit.num_qubits
        avg_values = np.zeros(num_qubits)
        
        for bitstring, count in counts.items():
            # Correct for Qiskit's little-endian bit order
            bitstring = bitstring[::-1] 
            for i, bit in enumerate(bitstring):
                if bit == '1':
                    avg_values[i] += count
        avg_values /= shots
        
        # Map the first 6 qubits to the mean update for the 6D state vector
        mean_update = (avg_values[0:6] - 0.5) * 0.1 
        
        # Map the next 6 qubits to the variance update. Ensure non-negativity.
        # Use min() to prevent IndexError if there are fewer than 12 qubits.
        variance_end_index = min(12, num_qubits)
        variance_values = np.abs(avg_values[6:variance_end_index]) * 0.01 + 1e-6
        
        # Pad with default variance if the circuit is smaller than 12 qubits
        if len(variance_values) < 6:
            padding = np.full(6 - len(variance_values), 1e-6)
            variance_values = np.concatenate([variance_values, padding])

        return mean_update, np.diag(variance_values)

    def update(self, measurement: Dict[str, float], ownship_position: np.ndarray = None) -> None:
        # Use richer radar likelihood combining range, closing_velocity, azimuth
        if ownship_position is None:
            ownship_position = np.zeros(3)
        rel_pos = self.particles[:, 0:3] - ownship_position
        ranges = np.linalg.norm(rel_pos, axis=1)
        azimuths = np.arctan2(rel_pos[:, 1], rel_pos[:, 0])
        # closing velocity approximation from state velocities
        rel_vel = self.particles[:, 3:6]
        closing_vel = -np.sum(rel_vel * rel_pos, axis=1) / (ranges + 1e-9)

        # Measurement values and assumed noise stds
        z_range = measurement.get('range', 0.0)
        z_az = measurement.get('azimuth', 0.0)
        z_cv = measurement.get('closing_velocity', 0.0)
        sigma_r = 20.0
        sigma_az = 0.003
        sigma_cv = 2.0

        # Wrap azimuth residual to [-pi, pi]
        def wrap_angle(a):
            return (a + np.pi) % (2 * np.pi) - np.pi

        dr = (ranges - z_range) / sigma_r
        da = wrap_angle(azimuths - z_az) / sigma_az
        dv = (closing_vel - z_cv) / sigma_cv

        mahal_sq = dr * dr + da * da + dv * dv
        likelihoods = np.exp(-0.5 * mahal_sq)
        self.weights *= likelihoods
        self.weights += 1e-300
        self.weights /= np.sum(self.weights)

    def resample(self) -> None:
        positions = (np.arange(self.num_particles) + np.random.rand()) / self.num_particles
        indexes = np.zeros(self.num_particles, 'i')
        cumulative_sum = np.cumsum(self.weights)
        i, j = 0, 0
        while i < self.num_particles:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j; i += 1
            else:
                j += 1
        self.particles = self.particles[indexes]
        self.weights.fill(1.0 / self.num_particles)
    
    def estimate_state(self) -> np.ndarray:
        """
        Returns the current best estimate of the 6D state [x, y, z, vx, vy, vz]
        as the weighted mean of the particles.
        """
        return np.average(self.particles, axis=0, weights=self.weights)
