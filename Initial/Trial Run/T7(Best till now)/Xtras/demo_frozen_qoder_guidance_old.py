import numpy as np
import pickle
import matplotlib.pyplot as plt

from Simulation.target_dynamics import Target
from Simulation.missile_dynamics import Missile
from Simulation.sensor_model import Sensor
from config import SimOptions
from Quantum_Core.nqpf import NQPF

def main():
    # Load trained qODER genome
    with open("champion_genome.pkl", "rb") as f:
        pretrained_genome = pickle.load(f)

    # Setup simulation + sensor
    sim_opts = SimOptions()
    sensor = Sensor(
    radar_noise_std={
        'range': 50.0,      # meters
        'velocity': 5.0,    # m/s
        'azimuth': 0.005    # radians (~0.3 deg)
    },
    irst_noise_std=0.1      # thermal noise scalar
)


    nqpf = NQPF(
        num_particles=200,#sim_opts.num_particles,
        trained_genome=pretrained_genome
    )

    # Scenario: Target ahead, maneuvering
    target = Target(
        initial_position=[12000.0, 0.0, 10000.0],
        initial_velocity=[-250.0, 40.0, 0.0]
    )

    # Missile launch position
    missile = Missile(
        initial_position=[0.0, 0.0, 9000.0],
        initial_velocity=[300.0, 0.0, 0.0]
    )

    # Initialize particles around a rough estimate
    initial_est = np.array([
        target.position[0], target.position[1], target.position[2],
        target.velocity[0], target.velocity[1], target.velocity[2]
    ])
    true_state = np.concatenate([target.position, target.velocity])
    # nqpf.initialize_swarm(initial_est, 200.0)
    nqpf.initialize_swarm(true_state, initial_uncertainty=50.0)

    num_steps = 300
    dt = sim_opts.dt

    target_hist = []
    missile_hist = []

    for step in range(num_steps):
        target.update(dt, ('turn', {'g_force': 4.0}))

        # obs = sensor.observe(target, missile.position)

        # # Quantum PF step (use smaller shots for speed)
        # nqpf.predict(dt=dt, shots=64)
        # nqpf.update(obs, missile.position)

        # # (Optional) resample when ESS drops too low
        # ess = nqpf.effective_sample_size()
        # if ess < sim_opts.ess_resample_threshold_ratio * nqpf.num_particles:
        #     nqpf.resample()

        # # ---- NEW: guide using PF estimate instead of truth ----
        # est_state = nqpf.estimate_state()   # [x,y,z,vx,vy,vz]
        # est_pos = est_state[:3]
        # est_vel = est_state[3:6]

        # missile.update(dt, est_pos, est_vel)
        obs = sensor.observe(target, missile.position)
        nqpf.predict(dt=dt)
        nqpf.update(obs, missile.position)

        est_state = nqpf.estimate_state()
        est_pos = est_state[:3]
        est_vel = est_state[3:6]

        missile.update(dt, est_pos, est_vel)


        target_hist.append(target.position.copy())
        missile_hist.append(missile.position.copy())


    target_hist = np.array(target_hist)
    missile_hist = np.array(missile_hist)

    final_miss = np.linalg.norm(target_hist[-1] - missile_hist[-1])
    print(f"\nFinal miss distance: {final_miss:.2f} meters\n")

    # Plot trajectories
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(target_hist[:, 0], target_hist[:, 1], target_hist[:, 2],
            'r-', label='Target')
    ax.plot(missile_hist[:, 0], missile_hist[:, 1], missile_hist[:, 2],
            'b-', label='Missile')

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Missile vs Target with Frozen Trained qODER")
    ax.legend()
    ax.view_init(elev=25, azim=135)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


#======================================================================================
# import numpy as np
# import matplotlib.pyplot as plt

# from Simulation.target_dynamics import Target
# from Simulation.missile_dynamics import Missile
# from Simulation.sensor_model import Sensor
# from config import SimOptions

# def main():
#     sim_opts = SimOptions()
#     dt = sim_opts.dt
#     num_steps = 300  # 30 seconds

#     # --- Initial conditions (same style as before) ---
#     target = Target(
#         initial_position=[12000.0, 0.0, 10000.0],
#         initial_velocity=[-250.0, 40.0, 0.0]
#     )

#     missile = Missile(
#         initial_position=[0.0, 0.0, 9000.0],
#         initial_velocity=[300.0, 0.0, 0.0]
#     )

#     sensor = Sensor(
#         radar_noise_std={'range': 50.0, 'velocity': 5.0, 'azimuth': 0.005},
#         irst_noise_std=0.1
#     )

#     target_hist = []
#     missile_hist = []

#     for step in range(num_steps):
#         # Target executes a 4G level turn
#         target.update(dt, ('turn', {'g_force': 4.0}))

#         # We *can* get a noisy observation (not used for guidance here)
#         _obs = sensor.observe(target, missile.position)

#         # ❗ SIMPLE BASELINE GUIDANCE:
#         # use TRUE target state (no PF, no quantum, nothing fancy)
#         missile.update(dt, target.position, target.velocity)

#         target_hist.append(target.position.copy())
#         missile_hist.append(missile.position.copy())

#     target_hist = np.array(target_hist)
#     missile_hist = np.array(missile_hist)

#     miss_distance = np.linalg.norm(target_hist[-1] - missile_hist[-1])
#     print(f"Final miss distance (truth-guided): {miss_distance:.2f} meters")

#     # --- Plot trajectories ---
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot(target_hist[:, 0], target_hist[:, 1], target_hist[:, 2], 'r-', label='Target')
#     ax.plot(missile_hist[:, 0], missile_hist[:, 1], missile_hist[:, 2], 'b-', label='Missile')
#     ax.set_xlabel("X (m)")
#     ax.set_ylabel("Y (m)")
#     ax.set_zlabel("Z (m)")
#     ax.set_title("Truth-Based Guidance (No PF, No Quantum)")
#     ax.legend()
#     plt.show()

# if __name__ == "__main__":
#     main()
