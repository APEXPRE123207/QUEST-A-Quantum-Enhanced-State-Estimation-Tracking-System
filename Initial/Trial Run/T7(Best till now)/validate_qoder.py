import pickle
import numpy as np
import random

from qiskit_aer import AerSimulator
from qiskit import transpile

from Quantum_Core.nqpf import build_circuit_from_genome
from Simulation.target_dynamics import Target
from Simulation.sensor_model import Sensor
from config import SimOptions

SIMULATOR = AerSimulator()
SENSOR = Sensor(
    radar_noise_std={'range': 50.0, 'velocity': 5.0, 'azimuth': 0.005},
    irst_noise_std=0.1
)

def generate_scenario():
    g_force = np.random.uniform(1.5, 3.0)
    maneuver = ('turn', {'g_force': g_force})
    pos = [np.random.uniform(8000, 12000),
           np.random.uniform(-1000, 1000),
           np.random.uniform(9000, 11000)]
    vx_sign = random.choice([-1, 1])
    vel = [vx_sign * np.random.uniform(200, 300),
           np.random.uniform(-20, 20),
           np.random.uniform(-20, 20)]
    return {'maneuver': maneuver, 'pos': pos, 'vel': vel}

def main():
    with open("champion_genome.pkl", "rb") as f:
        genome = pickle.load(f)

    sim_opts = SimOptions()
    np.random.seed(sim_opts.seed)
    random.seed(sim_opts.seed)

    qc_template, params = build_circuit_from_genome(genome)
    qc_template.measure_all()
    transpiled = transpile(qc_template, SIMULATOR)

    N_EPISODES = 10
    num_steps = 30
    total_error = 0.0

    for _ in range(N_EPISODES):
        scenario = generate_scenario()
        target = Target(scenario['pos'], scenario['vel'])

        for _ in range(num_steps):
            target.update(sim_opts.dt, scenario['maneuver'])
            obs = SENSOR.observe(target, ownship_position=np.zeros(3))

            input_state = np.array([
                obs['closing_velocity'],
                obs['azimuth'],
                target.get_state()[2][1],
                np.linalg.norm(target.get_state()[2]) / 9.81,
                obs['thermal_intensity']
            ])
            norm = np.linalg.norm(input_state)
            norm_state = input_state / (norm + 1e-9)
            if len(norm_state) < len(params):
                norm_state = np.pad(norm_state, (0, len(params) - len(norm_state)))

            param_map = {p: v for p, v in zip(params, norm_state)}
            bound = transpiled.assign_parameters(param_map)

            result = SIMULATOR.run(bound, shots=32).result()
            counts = result.get_counts()

            avg_values = np.zeros(genome.num_qubits)
            for bitstring, count in counts.items():
                bitstring = bitstring[::-1]
                for i, bit in enumerate(bitstring):
                    if bit == '1':
                        avg_values[i] += count
            avg_values /= 32

            predicted_output = (avg_values[:6] - 0.5) * 30.0
            true_accel = target.get_state()[2]
            predicted_accel = predicted_output[0:3]

            total_error += np.linalg.norm(predicted_accel - true_accel)

    mean_error = total_error / (N_EPISODES * num_steps)
    fitness = 1.0 / (1.0 + mean_error)
    print(f"Validation: mean accel error = {mean_error:.3f}, fitness ≈ {fitness:.3f}")

if __name__ == "__main__":
    main()
