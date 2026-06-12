import numpy as np

from Simulation.target_dynamics import Target
from Simulation.sensor_model import Sensor
from Quantum_Core.ctpf import CTPF
from config import SimOptions

def main():
    sim_opts = SimOptions()
    dt = sim_opts.dt
    num_steps = 300

    # Same target as in your viewers
    target = Target(
        initial_position=[12000.0, 0.0, 10000.0],
        initial_velocity=[-250.0, 40.0, 0.0]
    )

    sensor = Sensor(
        radar_noise_std={'range': 50.0, 'velocity': 5.0, 'azimuth': 0.005},
        irst_noise_std=0.1
    )

    num_particles = 700
    pf = CTPF(num_particles=num_particles)

    # Initialize around the TRUE initial state (best-case for PF)
    init_state = np.concatenate([target.position, target.velocity])
    pf.initialize_swarm(init_state,
                        pos_sigma=50.0,
                        vel_sigma=20.0,
                        omega_sigma=0.02)

    pf_pos_errors = []

    for step in range(num_steps):
        # Target does the same 4g turn
        target.update(dt, ('turn', {'g_force': 4.0}))

        obs = sensor.observe(target, np.zeros(3))  # ownship at origin for debug

        pf.predict(dt)
        pf.update(obs, np.zeros(3))

        if pf.effective_sample_size() < 0.5 * pf.num_particles:
            pf.resample()

        est_state = pf.estimate_state()
        est_pos = est_state[:3]
        pf_err = np.linalg.norm(est_pos - target.position)
        pf_pos_errors.append(pf_err)

    mean_pf_err = float(np.mean(pf_pos_errors))
    final_pf_err = float(pf_pos_errors[-1])

    print(f"[CTPF DEBUG] Mean PF position error: {mean_pf_err:.2f} m")
    print(f"[CTPF DEBUG] Final PF position error: {final_pf_err:.2f} m")

if __name__ == "__main__":
    main()
