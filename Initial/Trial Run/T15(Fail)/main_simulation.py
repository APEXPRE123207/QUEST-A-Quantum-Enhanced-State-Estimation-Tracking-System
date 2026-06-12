import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# --- Import Project Modules ---
# Quantum Core
from Quantum_Core.qneat import Genome
from Quantum_Core.nqpf import NQPF
from Quantum_Core.qaoa import QAOAOptimizer, build_quadratic_program

# Simulation Environment
from Simulation.target_dynamics import Target
from Simulation.sensor_model import Sensor

# Configuration
from config import SimOptions, QAOAOptions

def run_simulation():
    """
    Executes the full, end-to-end missile guidance simulation.
    """
    print("--- Initializing Simulation ---")

    # --- Load the Trained NQPF Model ---
    # Allow override via ENV or default next to script
    champion_path = os.environ.get("CHAMPION_GENOME_PATH", "champion_genome.pkl")
    if not os.path.exists(champion_path):
        print(f"Error: Champion genome not found at '{champion_path}'.")
        print("Please run train_main.py to generate the model.")
        return
    
    try:
        with open(champion_path, 'rb') as f:
            champion_genome = pickle.load(f)
        print(f"Successfully loaded champion genome with fitness: {champion_genome.fitness:.2f}")
    except Exception as e:
        print(f"Error loading champion genome: {e}")
        return

    # --- Initialize Simulation and Quantum Components ---
    sim_opts = SimOptions()
    # Seed reproducibility
    np.random.seed(sim_opts.seed)
    qaoa_opts = QAOAOptions()

    target = Target(
        initial_position=[15000, 2000, 10000], 
        initial_velocity=[-300, 0, -20]
    )
    
    sensor = Sensor(
        radar_noise_std={'range': 50.0, 'velocity': 5.0, 'azimuth': 0.005},
        irst_noise_std=0.1
    )
    
    # Instantiate the core quantum engines
    nqpf = NQPF(num_particles=sim_opts.num_particles, trained_genome=champion_genome)
    qaoa_optimizer = QAOAOptimizer(options=vars(qaoa_opts))
    
    # Initialize missile state and NQPF swarm
    missile_pos = np.array([0.0, 0.0, 9000.0])
    missile_vel = np.array([300.0, 0.0, 0.0])
    initial_observation = sensor.observe(target, missile_pos)
    initial_estimate = np.array([
        target.position[0], target.position[1], target.position[2],
        target.velocity[0], target.velocity[1], target.velocity[2]
    ])
    nqpf.initialize_swarm(initial_estimate, initial_uncertainty=100.0)

    # --- Data Logging ---
    target_history = []
    missile_history = []
    n_steps = int(sim_opts.total_time / sim_opts.dt)

    print("--- Starting Main Simulation Loop ---")
    fallback_counter = 0
    for step in range(n_steps):
        # 1. Define Target Maneuver for this phase of flight
        print(f"\nProcessing Step {step}...")
        if step < 50:
            maneuver = ('straight', {})
        elif step < 150:
            maneuver = ('turn', {'g_force': 7.0})
        else:
            maneuver = ('jink', {'frequency': 1.0, 'amplitude': 40.0})

        # 2. Update Target and Generate Sensor Data
        target.update(sim_opts.dt, maneuver)
        observation = sensor.observe(target, missile_pos)
        
        # 3. NQPF Prediction & Update Cycle
        print(" -> Running NQPF predict...")
        nqpf.predict(dt=sim_opts.dt)
        print(" -> NQPF predict complete.")
        nqpf.update(observation, ownship_position=missile_pos)
        
        # Adaptive resampling by ESS
        ess = nqpf.effective_sample_size()
        if ess < sim_opts.ess_resample_threshold_ratio * sim_opts.num_particles:
            nqpf.resample()

        # 4. Prepare Inputs for QAOA (multi-step horizon, quantum forecast)
        horizon = max(1, int(sim_opts.planning_horizon))
        step_ahead = float(sim_opts.waypoint_step)

        # Quantum forecast: use mean velocity estimate for target to roll ahead
        predicted_target_pos0 = np.mean(nqpf.particles[:, 0:3], axis=0)
        predicted_target_vel0 = np.mean(nqpf.particles[:, 3:6], axis=0)
        predicted_target_traj = np.zeros((horizon, 3))
        for t in range(horizon):
            predicted_target_traj[t] = predicted_target_pos0 + predicted_target_vel0 * sim_opts.dt * (t + 1)

        # Generate multi-timestep waypoints: smaller grid to reduce QAOA problem size
        num_waypoints = 4  # 2x2 grid instead of 3x3 to reduce memory
        waypoints = np.zeros((num_waypoints, horizon, 3))
        offsets = np.array([
            [-300, 300, 30], [300, 300, -30],
            [-300, -300, 30], [300, -300, -30]
        ])
        for t in range(horizon):
            base_waypoint = missile_pos + np.array([step_ahead * (t + 1), 0, 0])
            for i in range(num_waypoints):
                waypoints[i, t, :] = base_waypoint + offsets[i]

        # Risk forecast across the horizon (normalized per timestep)
        prob_forecast = np.zeros((num_waypoints, horizon))
        for t in range(horizon):
            d = np.linalg.norm(waypoints[:, t, :] - predicted_target_traj[t], axis=1)
            # Distance as risk; normalize to avoid zeros
            d = d + 1e-9
            prob_forecast[:, t] = d / np.sum(d)

        # 5. QAOA Optimization with safe fallback
        print(" -> Running QAOA optimize...")
        try:
            qp = build_quadratic_program(prob_forecast, missile_pos, waypoints, vars(qaoa_opts))
            solution_vector, _ = qaoa_optimizer.optimize(qp)
            fallback_counter = 0
            print(" -> QAOA optimize complete.")
        except Exception as e:
            print(f"QAOA failed: {e}")
            # Quick quantum-centric retry with lower reps and fewer shots
            try:
                qaoa_opts.reps = max(1, qaoa_opts.reps - 1)
                qaoa_opts.shots = max(64, qaoa_opts.shots // 2)
                qaoa_opts.max_iterations = max(25, qaoa_opts.max_iterations // 2)
                # Recreate optimizer so updated options take effect
                qaoa_optimizer = QAOAOptimizer(options=vars(qaoa_opts))
                qp = build_quadratic_program(prob_forecast, missile_pos, waypoints, vars(qaoa_opts))
                solution_vector, _ = qaoa_optimizer.optimize(qp)
                print(" -> QAOA retry complete.")
            except Exception as e2:
                print(f"QAOA retry failed: {e2}")
                solution_vector = None
                fallback_counter += 1
        
        # 6. Update Missile State (bounded kinematics)
        if solution_vector is None:
            print("Aborting: QAOA did not return a valid solution and fallback is disabled.")
            break
        chosen_trajectory = qaoa_optimizer.decode_solution(solution_vector, waypoints)
        next_missile_waypoint = chosen_trajectory[0]

        desired_velocity = (next_missile_waypoint - missile_pos) / sim_opts.dt
        desired_delta_v = desired_velocity - missile_vel
        max_delta_v = sim_opts.missile_max_accel * sim_opts.dt
        delta_v_norm = np.linalg.norm(desired_delta_v) + 1e-9
        if delta_v_norm > max_delta_v:
            desired_delta_v = desired_delta_v * (max_delta_v / delta_v_norm)
        missile_vel = missile_vel + desired_delta_v
        # Cap speed
        speed = np.linalg.norm(missile_vel)
        if speed > sim_opts.missile_max_speed:
            missile_vel = missile_vel * (sim_opts.missile_max_speed / (speed + 1e-9))
        missile_pos = missile_pos + missile_vel * sim_opts.dt
        
        # 7. Log Data and Print Status
        target_history.append(target.position.copy())
        missile_history.append(missile_pos.copy())
        
        if step % 50 == 0:
            distance = np.linalg.norm(target.position - missile_pos)
            print(f"Step: {step}/{n_steps} | Intercept Range: {distance:.2f} m")

        # 8. Check for Intercept
        if np.linalg.norm(target.position - missile_pos) < 100.0:
            print(f"\n--- INTERCEPT at Step {step} ---")
            break
            
    # --- Visualization ---
    if target_history and missile_history:
        target_traj = np.array(target_history)
        missile_traj = np.array(missile_history)

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot(target_traj[:, 0], target_traj[:, 1], target_traj[:, 2], label='Target Trajectory', color='r', linewidth=2)
        ax.plot(missile_traj[:, 0], missile_traj[:, 1], missile_traj[:, 2], label='Missile Trajectory', color='b', linewidth=2)
        
        ax.scatter(target_traj[0, 0], target_traj[0, 1], target_traj[0, 2], s=100, color='red', marker='^', label='Target Start')
        ax.scatter(missile_traj[0, 0], missile_traj[0, 1], missile_traj[0, 2], s=100, color='blue', marker='^', label='Missile Start')
        if len(missile_traj) > 1:
            ax.scatter(missile_traj[-1, 0], missile_traj[-1, 1], missile_traj[-1, 2], s=200, color='orange', marker='*', label='Final Point')

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("3D Engagement Simulation")
        ax.legend()
        plt.show()
    else:
        print("No trajectory data to visualize.")


if __name__ == '__main__':
    run_simulation()