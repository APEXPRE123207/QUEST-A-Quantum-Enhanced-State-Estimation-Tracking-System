import numpy as np
from qiskit_aer import AerSimulator
from qiskit import transpile
from Quantum_Core.nqpf import build_circuit_from_genome
from Simulation.sensor_model import Sensor
from Simulation.target_dynamics import Target
import os
import pickle
MANEUVERS = [
    ("straight", {}),

    # Coordinated turns (difficulty via g-force)
    ("turn", {"g_force": 2.0}),
    ("turn", {"g_force": 4.0}),

    # Evasive lateral motion
    ("jink", {"amplitude": 50.0, "frequency": 0.2}),

    # Vertical maneuver
    ("climb_dive", {"vertical_g": 3.0}),
]

CHECKPOINT_DIR = "checkpoints_qneat_dz"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BEST_GENOME_DIR = "best_genomes_qneat_dz"
os.makedirs(BEST_GENOME_DIR, exist_ok=True)


def compute_dz_true(z_seq, k, H=3):
    dz = []
    for t in range(1, H + 1):
        dz.append(z_seq[k + t] - z_seq[k + t - 1])
    return np.concatenate(dz)

def build_features(particle_state, ownship_position, azimuth, azimuth_rate):
    pos = particle_state[0:3]
    vel = particle_state[3:6]
    omega = particle_state[6]

    rel = pos - ownship_position
    r = np.linalg.norm(rel) + 1e-9

    closing_vel = -np.dot(vel, rel) / r
    thermal = 1.0 / (r*r + 1e-6)

    f = np.array([closing_vel, azimuth, azimuth_rate, omega, thermal])
    return f / (np.linalg.norm(f) + 1e-9)


def quantum_predict_dz(tqc_template, params, features, backend, shots=128):
    if len(params) == 0:
        return np.zeros(9)

    if len(features) < len(params):
        features = np.pad(features, (0, len(params)-len(features)))
    else:
        features = features[:len(params)]

    bound = tqc_template.assign_parameters(
        {p: v for p, v in zip(params, features)}
    )

    result = backend.run(bound, shots=shots).result()
    counts = result.get_counts()

    num_qubits = tqc_template.num_qubits
    avg = np.zeros(num_qubits)

    for bits, c in counts.items():
        bits = bits[::-1]
        for i, b in enumerate(bits):
            if b == '1':
                avg[i] += c

    avg /= shots

    dz = np.zeros(9)
    usable = min(9, len(avg))
    dz[:usable] = (avg[:usable] - 0.5) * 50.0
    return dz


def dz_error(dz_pred, dz_true):
    return np.mean(np.abs(dz_pred - dz_true))

backend = AerSimulator()
sensor = Sensor(radar_noise_std={'range': 50.0, 'velocity': 5.0, 'azimuth': 0.005}, irst_noise_std=0.1)
from Quantum_Core.qneat import Population, QNEATOptions

from Quantum_Core.qneat import Population, QNEATOptions

CHECKPOINT_PATH = "checkpoints_qneat_dz/population_checkpoint.pkl"

if os.path.exists(CHECKPOINT_PATH):
    population = Population.load_checkpoint(CHECKPOINT_PATH)
    print(f"Resumed from generation {population.generation}")
else:
    options = QNEATOptions(population_size=30)
    population = Population(num_qubits=9, options=options)



dt = 0.1
H = 3
EPISODES = 5
STEPS = 150
NUM_GENERATIONS = 100
def select_maneuvers_for_generation(gen):
        if gen < 5:
            return MANEUVERS[:1]        # straight only
        elif gen < 15:
            return MANEUVERS[:3]        # gentle turns
        elif gen < 30:
            return MANEUVERS[:4]        # sinusoidal
        else:
            return MANEUVERS[:]         # full difficulty
for gen in range(NUM_GENERATIONS):
    for genome in population.population:
        qc, params = build_circuit_from_genome(genome)
        qc.measure_all()
        tqc_template = transpile(qc, backend)
        total_error = 0.0
        count = 0

        for ep in range(EPISODES):
            target = Target(
            initial_position=[0.0, 0.0, 5000.0],
            initial_velocity=[300.0, 0.0, 0.0]
        )
            active_maneuvers = select_maneuvers_for_generation(gen)

            maneuver_type, maneuver_params = active_maneuvers[np.random.randint(len(active_maneuvers))]

            maneuver = (maneuver_type, maneuver_params)
            ownship_pos = np.zeros(3)

            z_seq = []
            state_seq = []
            for t in range(STEPS):
                target.update(dt, maneuver)
                pos, vel, acc = target.get_state()

                state_seq.append(np.hstack([pos, vel, np.linalg.norm(acc)]))
                z_seq.append(pos.copy())
            az_seq = [
                np.arctan2(z[1], z[0])
                for z in z_seq
            ]

            for k in range(STEPS - H - 1):
                az = az_seq[k]
                if k == 0:
                    az_rate = 0.0
                else:
                    da = az_seq[k] - az_seq[k-1]
                    da = (da + np.pi) % (2*np.pi) - np.pi
                    az_rate = da / dt

                features = build_features(
                    state_seq[k],
                    ownship_pos,
                    az,
                    az_rate
                )

                dz_true = compute_dz_true(z_seq, k, H)
                dz_pred = quantum_predict_dz(tqc_template, params, features, backend)
                if gen == 0 and ep == 0 and k == 0:
                    print("Sample dz_true:", dz_true)
                    print("Sample dz_pred:", dz_pred)

                total_error += dz_error(dz_pred, dz_true)
                count += 1

        if count == 0:
            genome.fitness = 0.0
        else:
            genome.fitness = 1.0 / (1.0 + total_error / count)


    population.run_evolutionary_cycle()
    if gen % 1 == 0:   # every 5 generations
        ckpt_path = os.path.join(
            CHECKPOINT_DIR,
            f"population_checkpoint.pkl"
        )
        population.save_checkpoint(ckpt_path)
    if gen % 10 == 0:
        best_genome = max(population.population, key=lambda g: g.fitness)
        path = os.path.join(BEST_GENOME_DIR, f"best_genome_gen_{gen}.pkl")
        with open(path, "wb") as f:
            pickle.dump(best_genome, f)   

    mean_fitness = np.mean([g.fitness for g in population.population])
    print(f"[GEN {gen}] mean fitness = {mean_fitness:.4f}")

    print(f"[GEN {gen}] best fitness = {max(population.population, key=lambda g: g.fitness).fitness:.4f}")



best_genome = max(population.population, key=lambda g: g.fitness)

with open("best_qneat_dz_genome.pkl", "wb") as f:
    pickle.dump(best_genome, f)

print("Saved best genome to best_qneat_dz_genome.pkl")

