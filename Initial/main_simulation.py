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
    champion_path = "champion_genome.pkl"
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
        nqpf.update(observation)
        
        # Avoid resampling on every single step to maintain diversity
        if step % 5 == 0:
            nqpf.resample()

        # 4. Prepare Inputs for QAOA
        # Generate a probabilistic forecast from the NQPF particle cloud
        # For simplicity, we use the mean of the particle cloud as the forecast
        predicted_target_pos = np.mean(nqpf.particles[:, 0:3], axis=0)

        # Generate waypoints for the missile to choose from
        # A simple grid of 3x3 waypoints ahead of the missile
        num_waypoints = 9
        waypoints = np.zeros((num_waypoints, 1, 3)) # Shape for QAOA function
        base_waypoint = missile_pos + np.array([1000, 0, 0]) # Fly forward
        offsets = np.array([
            [-500, 500, 50], [0, 500, 0], [500, 500, -50],
            [-500, 0, 50],   [0, 0, 0],   [500, 0, -50],
            [-500, -500, 50],[0, -500, 0],[500, -500, -50]
        ])
        for i in range(num_waypoints):
            waypoints[i, 0, :] = base_waypoint + offsets[i]

        # Use predicted target position to create a "risk" forecast for QAOA
        prob_forecast = np.zeros((num_waypoints, 1))
        distances = np.linalg.norm(waypoints[:, 0, :] - predicted_target_pos, axis=1)
        prob_forecast[:, 0] = distances / np.sum(distances) # Lower distance = higher cost/risk

        # 5. QAOA Optimization
        print(" -> Running QAOA optimize...")
        qp = build_quadratic_program(prob_forecast, missile_pos, waypoints, vars(qaoa_opts))
        solution_vector, _ = qaoa_optimizer.optimize(qp)
        print(" -> QAOA optimize complete.")
        
        # 6. Update Missile State
        chosen_trajectory = qaoa_optimizer.decode_solution(solution_vector, waypoints)
        next_missile_waypoint = chosen_trajectory[0]
        
        # Simple kinematic update for the missile
        missile_velocity = (next_missile_waypoint - missile_pos) / sim_opts.dt
        missile_pos += missile_velocity * sim_opts.dt
        
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
    target_traj = np.array(target_history)
    missile_traj = np.array(missile_history)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(target_traj[:, 0], target_traj[:, 1], target_traj[:, 2], label='Target Trajectory', color='r', linewidth=2)
    ax.plot(missile_traj[:, 0], missile_traj[:, 1], missile_traj[:, 2], label='Missile Trajectory', color='b', linewidth=2)
    
    ax.scatter(target_traj[0, 0], target_traj[0, 1], target_traj[0, 2], s=100, color='red', marker='^', label='Target Start')
    ax.scatter(missile_traj[0, 0], missile_traj[0, 1], missile_traj[0, 2], s=100, color='blue', marker='^', label='Missile Start')
    ax.scatter(missile_traj[-1, 0], missile_traj[-1, 1], missile_traj[-1, 2], s=200, color='orange', marker='*', label='Intercept Point')

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D Engagement Simulation")
    ax.legend()
    plt.show()


if __name__ == '__main__':
    run_simulation()