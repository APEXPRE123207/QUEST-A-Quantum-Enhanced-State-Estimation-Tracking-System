import numpy as np
from typing import Dict

class CTPF:
    """
    Coordinated-Turn Particle Filter (classical, no quantum).

    State per particle:
      [x, y, z, vx, vy, vz, omega]
        x,y,z  : position in meters (inertial)
        vx,vy,vz: velocity in m/s
        omega  : turn rate about +Z (rad/s), for level-ish coordinated turns
    """

    def __init__(self, num_particles: int):
        self.num_particles = num_particles
        self.particles = np.zeros((num_particles, 7))
        self.weights = np.ones(num_particles) / num_particles

    # ---------- Basics ----------

    def initialize_swarm(self, initial_state: np.ndarray,
                         pos_sigma: float = 100.0,
                         vel_sigma: float = 20.0,
                         omega_sigma: float = 0.02):
        """
        initial_state: [x,y,z,vx,vy,vz] truth-ish guess
        spreads particles around that, with small random omega.
        """
        if initial_state.shape[0] != 6:
            raise ValueError("initial_state must be 6D [x,y,z,vx,vy,vz].")

        base = np.tile(initial_state, (self.num_particles, 1))
        noise_pos = np.random.normal(0.0, pos_sigma, size=(self.num_particles, 3))
        noise_vel = np.random.normal(0.0, vel_sigma, size=(self.num_particles, 3))
        omega = np.random.normal(0.0, omega_sigma, size=(self.num_particles, 1))

        self.particles[:, 0:3] = base[:, 0:3] + noise_pos
        self.particles[:, 3:6] = base[:, 3:6] + noise_vel
        self.particles[:, 6:7] = omega
        self.weights.fill(1.0 / self.num_particles)

    def effective_sample_size(self) -> float:
        return 1.0 / (np.sum(self.weights ** 2) + 1e-12)

    def estimate_state(self) -> np.ndarray:
        """
        Returns weighted mean as [x,y,z,vx,vy,vz,omega].
        """
        return np.average(self.particles, axis=0, weights=self.weights)

    # ---------- Motion Model: Coordinated Turn ----------

    def predict(self, dt: float):
        """
        Coordinated-turn predict:

          heading_k   = atan2(vy, vx)
          heading_{k+1} = heading_k + omega*dt
          v_mag stays about constant (small noise)
          z updated via vz

        With small process noise on v and omega.
        """
        # Process noise stds
        accel_std = 5.0        # m/s^2 equivalent
        omega_std = 0.01       # rad/s per step
        vz_std = 1.0           # m/s vertical noise

        # Unpack
        x = self.particles[:, 0]
        y = self.particles[:, 1]
        z = self.particles[:, 2]
        vx = self.particles[:, 3]
        vy = self.particles[:, 4]
        vz = self.particles[:, 5]
        omega = self.particles[:, 6]

        # Speeds and headings
        speed_xy = np.sqrt(vx**2 + vy**2) + 1e-6
        heading = np.arctan2(vy, vx)

        # Random walk for omega
        omega = omega + np.random.normal(0.0, omega_std, size=omega.shape)

        # New heading
        new_heading = heading + omega * dt

        # Add some tangential accel noise in XY-plane
        tangential_noise = np.random.normal(0.0, accel_std, size=speed_xy.shape) * dt
        speed_xy = np.clip(speed_xy + tangential_noise, 50.0, 450.0)  # [50,450] m/s

        # New horizontal velocity
        vx = speed_xy * np.cos(new_heading)
        vy = speed_xy * np.sin(new_heading)

        # Vertical velocity random walk
        vz = vz + np.random.normal(0.0, vz_std, size=vz.shape)
        vz = np.clip(vz, -200.0, 200.0)

        # Position update
        x = x + vx * dt
        y = y + vy * dt
        z = z + vz * dt

        # Clamp altitude (sane bounds)
        z = np.clip(z, 0.0, 20000.0)

        # Store back
        self.particles[:, 0] = x
        self.particles[:, 1] = y
        self.particles[:, 2] = z
        self.particles[:, 3] = vx
        self.particles[:, 4] = vy
        self.particles[:, 5] = vz
        self.particles[:, 6] = omega

    # ---------- Measurement Update (Radar-like) ----------

    def update(self, measurement: Dict[str, float], ownship_position: np.ndarray):
        """
        Measurement model uses:
          - range         (m)
          - azimuth       (rad)
          - closing_velocity (m/s)
          - thermal_intensity (relative, optional)

        ownship_position: missile position in same frame as particles.
        """
        if ownship_position is None:
            ownship_position = np.zeros(3)

        rel_pos = self.particles[:, 0:3] - ownship_position
        ranges = np.linalg.norm(rel_pos, axis=1) + 1e-9
        azimuths = np.arctan2(rel_pos[:, 1], rel_pos[:, 0])

        # Closing velocity along line of sight
        rel_vel = self.particles[:, 3:6]
        closing_vel = -np.sum(rel_vel * rel_pos, axis=1) / ranges

        # Sensors
        z_range = measurement.get('range', 0.0)
        z_az = measurement.get('azimuth', 0.0)
        z_cv = measurement.get('closing_velocity', 0.0)

        # "Thermal" is roughly 1/r^2, we can match that logic
        z_thermal = measurement.get('thermal_intensity', None)
        if z_thermal is not None:
            thermal_pred = 1.0 / (ranges**2 + 1e-6)
        else:
            thermal_pred = None

        # Noise stds (tuneable)
        sigma_r = 40.0    # m
        sigma_az = 0.01   # rad
        sigma_cv = 5.0    # m/s
        sigma_th = 0.2 * (z_thermal if z_thermal is not None and z_thermal > 0 else 1.0)

        # Angle wrapping
        def wrap_angle(a):
            return (a + np.pi) % (2 * np.pi) - np.pi

        dr = (ranges - z_range) / sigma_r
        da = wrap_angle(azimuths - z_az) / sigma_az
        dv = (closing_vel - z_cv) / sigma_cv

        mahal_sq = dr*dr + da*da + dv*dv

        if thermal_pred is not None:
            dt = (thermal_pred - z_thermal) / (sigma_th + 1e-9)
            mahal_sq += dt * dt

        # Convert to likelihoods
        # Clip to avoid underflow / overflow
        mahal_sq = np.clip(mahal_sq, 0.0, 100.0)
        likelihoods = np.exp(-0.5 * mahal_sq)

        # Update weights
        self.weights *= likelihoods
        self.weights += 1e-300
        self.weights /= np.sum(self.weights)

    # ---------- Resampling ----------

    def resample(self):
        """
        Systematic resampling.
        """
        N = self.num_particles
        positions = (np.arange(N) + np.random.rand()) / N
        indexes = np.zeros(N, dtype=int)
        cumulative_sum = np.cumsum(self.weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        self.particles = self.particles[indexes]
        self.weights.fill(1.0 / N)


# ==========================================================
#  Quantum-assisted Coordinated-Turn PF (QCTPF)
# ==========================================================
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit.circuit import Parameter
from .qneat import Genome
from .nqpf import build_circuit_from_genome  # you already have this

class QCTPF(CTPF):
    """
    QCTPF = CTPF + quantum acceleration prediction from a trained genome.

    - Uses the same particle state as CTPF: [x,y,z,vx,vy,vz,omega]
    - Classical part: super().predict(dt)
    - Quantum part: small velocity correction v += alpha_q * a_q * dt
    """

    def __init__(self, num_particles: int, trained_genome: Genome):
        super().__init__(num_particles)
        self.trained_genome = trained_genome
        self._prev_azimuth = np.full(self.num_particles, np.nan)
        # Quantum circuit setup
        self.backend = AerSimulator()
        self.qc_template, self.params = build_circuit_from_genome(trained_genome)

        if len(self.qc_template) > 0:
            self.qc_template.measure_all()
            self.transpiled_template = transpile(self.qc_template, self.backend)
        else:
            self.transpiled_template = self.qc_template

    def _build_features_for_particle(self, particle_state: np.ndarray,
                                     ownship_position: np.ndarray,idx) -> np.ndarray:
        """
        Build the same kind of input feature vector we used during QODER training.
        Here we mimic:
          [closing_velocity, azimuth, 0, 0, thermal_proxy]
        """
        pos = particle_state[0:3]
        vel = particle_state[3:6]

        rel_pos = pos - ownship_position
        r = np.linalg.norm(rel_pos) + 1e-9

        azimuth = np.arctan2(rel_pos[1], rel_pos[0])
        if np.isnan(self._prev_azimuth[idx]):
            azimuth_rate = 0.0
        else:
            d_az = azimuth - self._prev_azimuth[idx]
            d_az = (d_az + np.pi) % (2 * np.pi) - np.pi
            azimuth_rate = d_az / self._dt

        self._prev_azimuth[idx] = azimuth


        closing_vel = -np.dot(vel, rel_pos) / r

        thermal_proxy = 1.0 / (r * r + 1e-6)

        features = np.array([
            closing_vel,
            azimuth,
            azimuth_rate,           #  turn-rate proxy (NEW, Phase 6.1)
            0.0,           # dummy g-load
            thermal_proxy
        ], dtype=float)

        # Normalize as during training
        norm = np.linalg.norm(features)
        return features / (norm + 1e-9)

    def predict(self, dt: float,
                ownship_position: np.ndarray,
                alpha_q: float = 0.1,
                shots: int = 64):
        """
        Classical CTPF predict + quantum correction.

        1) Call super().predict(dt) for coordinated-turn model.
        2) For each particle, build features and run the quantum circuit batch.
        3) Decode first 3 qubits to get acceleration a_q (m/s^2).
        4) Apply v += alpha_q * a_q * dt and a small position correction.
        """

        # --- Classical coordinated-turn step ---
        self._dt = dt
        super().predict(dt)

        # If no quantum circuit, just stay classical
        if not self.params or self.transpiled_template is None or len(self.transpiled_template) == 0:
            return

        # --- Build param maps for all particles ---
        feature_list = []
        for i in range(self.num_particles):
            f = self._build_features_for_particle(self.particles[i, 0:6], ownship_position, i)
            feature_list.append(f)

        param_maps = []
        for feat in feature_list:
            if len(feat) < len(self.params):
                padding = np.zeros(len(self.params) - len(feat))
                feat_padded = np.concatenate([feat, padding])
            else:
                feat_padded = feat
            pm = {p: v for p, v in zip(self.params, feat_padded)}
            param_maps.append(pm)

        # --- Run all circuits in a batch ---
        bound_circuits = [
            self.transpiled_template.assign_parameters(pm)
            for pm in param_maps
        ]
        result = self.backend.run(bound_circuits, shots=shots).result()

        max_accel = 40.0  # m/s^2, clip to ~4g

        # --- Decode & apply corrections ---
        for i in range(self.num_particles):
            counts = result.get_counts(i)
            num_qubits = self.qc_template.num_qubits
            avg_values = np.zeros(num_qubits)

            for bitstring, count in counts.items():
                bitstring = bitstring[::-1]  # little-endian fix
                for q, bit in enumerate(bitstring):
                    if bit == '1':
                        avg_values[q] += count
            avg_values /= shots

            # Map first 3 outputs to acceleration (similar scale to training)
            a_q = (avg_values[0:3] - 0.5) * 30.0  # m/s^2

            # Clip
            a_norm = np.linalg.norm(a_q)
            if a_norm > max_accel:
                a_q = a_q / (a_norm + 1e-9) * max_accel

            # Apply small quantum correction
            dv = alpha_q * a_q * dt
            self.particles[i, 3:6] += dv
            # small position correction too
            self.particles[i, 0:3] += 0.5 * alpha_q * a_q * dt * dt
