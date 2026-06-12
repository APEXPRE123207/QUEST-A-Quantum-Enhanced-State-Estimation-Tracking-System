import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

from Simulation.target_dynamics import Target
from Simulation.missile_dynamics import Missile
from Simulation.sensor_model import Sensor
from Quantum_Core.nqpf import NQPF
from config import SimOptions

def run_pf_guidance_viewer():
    sim_opts = SimOptions()
    dt = sim_opts.dt
    num_steps = 300  # keep same as truth demo

    # --- Initial conditions (same as your good truth demo) ---
    target = Target(
        initial_position=[12000.0, 0.0, 10000.0],
        initial_velocity=[-250.0, 40.0, 0.0]
    )

    missile = Missile(
        initial_position=[0.0, 0.0, 9000.0],
        initial_velocity=[300.0, 0.0, 0.0]
    )

    sensor = Sensor(
        radar_noise_std={'range': 50.0, 'velocity': 5.0, 'azimuth': 0.005},
        irst_noise_std=0.1
    )

    # --- Load any genome (PF will still work in classical mode if predict is classical) ---
    with open("champion_genome.pkl", "rb") as f:
        pretrained_genome = pickle.load(f)

    num_particles = 500  # viewer-friendly
    nqpf = NQPF(num_particles=num_particles, trained_genome=pretrained_genome)

    # Initialize swarm around true initial target state (for debugging)
    init_state = np.concatenate([target.position, target.velocity])
    nqpf.initialize_swarm(init_state, initial_uncertainty=50.0)

    # --- Histories for plotting ---
    target_hist = []
    missile_hist = []
    est_hist = []
    pf_pos_errors = []

    # --- Set up interactive plot ---
    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    target_line, = ax.plot([], [], [], 'r-', label='Target (truth)')
    missile_line, = ax.plot([], [], [], 'b-', label='Missile')
    est_line,    = ax.plot([], [], [], 'g--', label='PF Estimate')

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("PF-Based Guidance Viewer")
    ax.legend()

    # Set a reasonable view box (based on your initial conditions)
    ax.set_xlim(-2000, 13000)
    ax.set_ylim(-4000, 4000)
    ax.set_zlim(8000, 12000)

    for step in range(num_steps):
        # --- True target evolution ---
        target.update(dt, ('turn', {'g_force': 4.0}))

        # Noisy observation from missile's frame
        obs = sensor.observe(target, missile.position)

        # --- PF update ---
        nqpf.predict(dt=dt)            # whatever predict you currently have
        nqpf.update(obs, missile.position)

        if nqpf.effective_sample_size() < 0.5 * nqpf.num_particles:
            nqpf.resample()

        est_state = nqpf.estimate_state()
        est_pos = est_state[:3]
        est_vel = est_state[3:6]

        # PF error vs truth
        pf_err = np.linalg.norm(est_pos - target.position)
        pf_pos_errors.append(pf_err)

        # --- Guidance uses PF estimate ---
        missile.update(dt, est_pos, est_vel)

        target_hist.append(target.position.copy())
        missile_hist.append(missile.position.copy())
        est_hist.append(est_pos.copy())

        # --- Update plot every N steps to keep it fast ---
        if step % 5 == 0 or step == num_steps - 1:
            th = np.array(target_hist)
            mh = np.array(missile_hist)
            eh = np.array(est_hist)

            target_line.set_data(th[:, 0], th[:, 1])
            target_line.set_3d_properties(th[:, 2])

            missile_line.set_data(mh[:, 0], mh[:, 1])
            missile_line.set_3d_properties(mh[:, 2])

            est_line.set_data(eh[:, 0], eh[:, 1])
            est_line.set_3d_properties(eh[:, 2])

            # Update title with live stats
            miss_now = np.linalg.norm(target.position - missile.position)
            ax.set_title(
                f"PF Guidance | step {step}/{num_steps}  "
                f"Miss: {miss_now:7.1f} m  "
                f"PF err: {pf_err:7.1f} m"
            )

            plt.draw()
            plt.pause(0.001)

    # --- Final stats ---
    target_hist = np.array(target_hist)
    missile_hist = np.array(missile_hist)
    est_hist = np.array(est_hist)

    final_miss = np.linalg.norm(target_hist[-1] - missile_hist[-1])
    mean_pf_err = float(np.mean(pf_pos_errors))
    final_pf_err = float(pf_pos_errors[-1])

    print(f"\n[Viewer] Final miss distance: {final_miss:.2f} meters")
    print(f"[Viewer] Mean PF position error: {mean_pf_err:.2f} meters")
    print(f"[Viewer] Final PF position error: {final_pf_err:.2f} meters")

    # Keep plot open at the end
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_pf_guidance_viewer()
