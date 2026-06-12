n1=1 #1:CTPF, 0: QCTPF
import numpy as np
import matplotlib.pyplot as plt

from Simulation.target_dynamics import Target
from Simulation.missile_dynamics import Missile
from Simulation.sensor_model import Sensor
from Quantum_Core.ctpf import CTPF
from config import SimOptions
import numpy as np
import matplotlib.pyplot as plt

from Simulation.target_dynamics import Target
from Simulation.missile_dynamics import Missile
from Simulation.sensor_model import Sensor
from Quantum_Core.ctpf import CTPF
from config import SimOptions
from Quantum_Core.ctpf import QCTPF 
import pickle

np.random.seed(42)

def generate_target_trajectory(sim_opts, num_steps):
    dt = sim_opts.dt

    target = Target(
        initial_position=[12000.0, 0.0, 10000.0],
        initial_velocity=[-250.0, 40.0, 0.0]
    )

    traj = []

    for step in range(num_steps):
        if step < 80:
            maneuver = ('turn', {'g_force': 3.0})
        elif step < 160:
            maneuver = ('jink', {'frequency': 1.2, 'amplitude': 30.0})
        else:
            maneuver = ('climb_dive', {'vertical_g': 2.0})

        target.update(dt, maneuver)

        # store full state
        traj.append((
            target.position.copy(),
            target.velocity.copy()
        ))

    return traj


def run_ctpf_guidance_viewer(target_traj):
    dt = sim_opts.dt

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

    # --- CTPF classical filter ---
    num_particles = 500
    pf = CTPF(num_particles=num_particles)

    init_state = np.concatenate([target_traj[0][0], target_traj[0][1]])
    init_state += np.random.normal(0, [200,200,200, 50,50,50])
    pf.initialize_swarm(init_state, pos_sigma=100.0, vel_sigma=30.0, omega_sigma=0.03)

    target_hist = []
    missile_hist = []
    est_hist = []
    pf_pos_errors = []
    guid_state = None          # [x,y,z,vx,vy,vz]
    smooth_alpha = 0.1         # 0<alpha<=1; smaller = more smoothing
    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    target_line, = ax.plot([], [], [], 'r-', label='Target (truth)')
    missile_line, = ax.plot([], [], [], 'b-', label='Missile')
    est_line,    = ax.plot([], [], [], 'g--', label='CTPF Estimate')
    t_start = ax.scatter([], [], [], c='red',   s=60, marker='o')
    t_end   = ax.scatter([], [], [], c='red',   s=80, marker='x')

    m_start = ax.scatter([], [], [], c='blue',  s=60, marker='o')
    m_end   = ax.scatter([], [], [], c='blue',  s=80, marker='x')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("CTPF Guidance Viewer")
    ax.legend()
    ax.set_xlim(-2000, 13000)
    ax.set_ylim(-4000, 4000)
    ax.set_zlim(8000, 12000)

    for step in range(num_steps):
        pos, vel = target_traj[step]
        target.position = pos.copy()
        target.velocity = vel.copy()
        target._time = step * dt

        obs = sensor.observe(target, missile.position)

        pf.predict(dt)
        pf.update(obs, missile.position)

        if pf.effective_sample_size() < 0.5 * pf.num_particles:
            pf.resample()

        est_state = pf.estimate_state()
        est_pos = est_state[:3]
        est_vel = est_state[3:6]

        pf_err = np.linalg.norm(est_pos - target.position)
        pf_pos_errors.append(pf_err)

        # --- Smooth the estimate before feeding it to guidance ---
        if guid_state is None:
            # Initialize with first estimate
            guid_state = np.concatenate([est_pos, est_vel])
        else:
            guid_state = (
                (1.0 - smooth_alpha) * guid_state +
                smooth_alpha * np.concatenate([est_pos, est_vel])
            )

        guid_pos = guid_state[:3]
        guid_vel = guid_state[3:6]

        # Missile guided by SMOOTHED PF estimate
        missile.update(dt, guid_pos, guid_vel)



        target_hist.append(target.position.copy())
        missile_hist.append(missile.position.copy())
        est_hist.append(est_pos.copy())

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
            t_start._offsets3d = (
                [th[0, 0]], [th[0, 1]], [th[0, 2]]
            )
            t_end._offsets3d = (
                [th[-1, 0]], [th[-1, 1]], [th[-1, 2]]
            )

            # --- Missile start/end markers ---
            m_start._offsets3d = (
                [mh[0, 0]], [mh[0, 1]], [mh[0, 2]]
            )
            m_end._offsets3d = (
                [mh[-1, 0]], [mh[-1, 1]], [mh[-1, 2]]
            )
            miss_now = np.linalg.norm(target.position - missile.position)
            ax.set_title(
                f"CTPF Guidance | step {step}/{num_steps}  "
                f"Miss: {miss_now:7.1f} m  PF err: {pf_err:7.1f} m"
            )

            plt.draw()
            plt.pause(0.001)

    target_hist = np.array(target_hist)
    missile_hist = np.array(missile_hist)
    est_hist = np.array(est_hist)

    final_miss = np.linalg.norm(target_hist[-1] - missile_hist[-1])
    mean_pf_err = float(np.mean(pf_pos_errors))
    final_pf_err = float(pf_pos_errors[-1])

    print(f"\n[CTPF] Final miss distance: {final_miss:.2f} meters")
    print(f"[CTPF] Mean PF position error: {mean_pf_err:.2f} meters")
    print(f"[CTPF] Final PF position error: {final_pf_err:.2f} meters")
    results = {
    "final_miss": final_miss,
    "mean_pf_error": mean_pf_err,
    "final_pf_error": final_pf_err
    }

    with open("phase5_ctpf_results.pkl", "wb") as f:
        pickle.dump(results, f)
    plt.ioff()
    plt.show()

def run_qctpf_guidance_viewer(target_traj):
    dt = sim_opts.dt
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

    # Load champion genome
    with open("champion_genome_6_4.pkl", "rb") as f:
        champion = pickle.load(f)

    num_particles = 500
    pf = QCTPF(num_particles=num_particles, trained_genome=champion)


    init_state = np.concatenate([target_traj[0][0], target_traj[0][1]])
    init_state += np.random.normal(0, [200,200,200, 50,50,50])

    pf.initialize_swarm(init_state, pos_sigma=100.0, vel_sigma=30.0, omega_sigma=0.03)

    target_hist = []
    missile_hist = []
    est_hist = []
    pf_pos_errors = []
    guid_state = None          # [x,y,z,vx,vy,vz]
    smooth_alpha = 0.1         # 0<alpha<=1; smaller = more smoothing
    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    target_line, = ax.plot([], [], [], 'r-', label='Target (truth)')
    missile_line, = ax.plot([], [], [], 'b-', label='Missile')
    est_line,    = ax.plot([], [], [], 'g--', label='QCTPF Estimate')
    # --- Start / End markers ---
    t_start = ax.scatter([], [], [], c='red',   s=60, marker='o')
    t_end   = ax.scatter([], [], [], c='red',   s=80, marker='x')

    m_start = ax.scatter([], [], [], c='blue',  s=60, marker='o')
    m_end   = ax.scatter([], [], [], c='blue',  s=80, marker='x')

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("QCTPF Guidance Viewer")
    ax.legend()
    ax.set_xlim(-2000, 13000)
    ax.set_ylim(-4000, 4000)
    ax.set_zlim(8000, 12000)

    for step in range(num_steps):
        pos, vel = target_traj[step]
        target.position = pos.copy()
        target.velocity = vel.copy()
        target._time = step * dt


        obs = sensor.observe(target, missile.position)

        # pf.predict(dt, missile.position, alpha_q=0.1)
        pf.predict(dt, missile.position, alpha_q=0.03, shots=64)
        pf.update(obs, missile.position)

        if pf.effective_sample_size() < 0.5 * pf.num_particles:
            pf.resample()

        est_state = pf.estimate_state()
        est_pos = est_state[:3]
        est_vel = est_state[3:6]

        pf_err = np.linalg.norm(est_pos - target.position)
        pf_pos_errors.append(pf_err)

        # --- Smooth the estimate before feeding it to guidance ---
        if guid_state is None:
            # Initialize with first estimate
            guid_state = np.concatenate([est_pos, est_vel])
        else:
            guid_state = (
                (1.0 - smooth_alpha) * guid_state +
                smooth_alpha * np.concatenate([est_pos, est_vel])
            )

        guid_pos = guid_state[:3]
        guid_vel = guid_state[3:6]

        # Missile guided by SMOOTHED PF estimate
        missile.update(dt, guid_pos, guid_vel)



        target_hist.append(target.position.copy())
        missile_hist.append(missile.position.copy())
        est_hist.append(est_pos.copy())

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
            # --- Target start/end markers ---
            t_start._offsets3d = (
                [th[0, 0]], [th[0, 1]], [th[0, 2]]
            )
            t_end._offsets3d = (
                [th[-1, 0]], [th[-1, 1]], [th[-1, 2]]
            )

            # --- Missile start/end markers ---
            m_start._offsets3d = (
                [mh[0, 0]], [mh[0, 1]], [mh[0, 2]]
            )
            m_end._offsets3d = (
                [mh[-1, 0]], [mh[-1, 1]], [mh[-1, 2]]
            )

            miss_now = np.linalg.norm(target.position - missile.position)
            ax.set_title(
                f"QCTPF Guidance | step {step}/{num_steps}  "
                f"Miss: {miss_now:7.1f} m  PF err: {pf_err:7.1f} m"
            )

            plt.draw()
            plt.pause(0.001)

    target_hist = np.array(target_hist)
    missile_hist = np.array(missile_hist)
    est_hist = np.array(est_hist)

    final_miss = np.linalg.norm(target_hist[-1] - missile_hist[-1])
    mean_pf_err = float(np.mean(pf_pos_errors))
    final_pf_err = float(pf_pos_errors[-1])

    print(f"\n[QCTPF] Final miss distance: {final_miss:.2f} meters")
    print(f"[QCTPF] Mean PF position error: {mean_pf_err:.2f} meters")
    print(f"[QCTPF] Final PF position error: {final_pf_err:.2f} meters")
    results = {
    "final_miss": final_miss,
    "mean_pf_error": mean_pf_err,
    "final_pf_error": final_pf_err
    }

    with open("phase5_qctpf_results.pkl", "wb") as f:
        pickle.dump(results, f)

    plt.ioff()
    plt.show()

sim_opts = SimOptions()
num_steps = 300

target_traj = generate_target_trajectory(sim_opts, num_steps)

print("\n--- Running CTPF on frozen trajectory ---")
run_ctpf_guidance_viewer(target_traj)

print("\n--- Running QCTPF on SAME frozen trajectory ---")
run_qctpf_guidance_viewer(target_traj)



