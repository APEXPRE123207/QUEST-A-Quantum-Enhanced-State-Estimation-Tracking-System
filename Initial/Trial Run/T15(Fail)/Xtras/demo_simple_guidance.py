import numpy as np
import matplotlib.pyplot as plt

from Simulation.target_dynamics import Target
from Simulation.missile_dynamics import Missile
from Simulation.sensor_model import Sensor
from config import SimOptions

def main():
    sim_opts = SimOptions()
    dt = sim_opts.dt
    num_steps = 300  # 30 seconds

    # Target: ahead & slightly above
    target = Target(
        initial_position=[12000.0, 0.0, 10000.0],
        initial_velocity=[-250.0, 40.0, 0.0]
    )

    # Missile: near origin, flying forward
    missile = Missile(
        initial_position=[0.0, 0.0, 9000.0],
        initial_velocity=[300.0, 0.0, 0.0]
    )

    # Sensor (we won't use PF, just for completeness)
    sensor = Sensor(
        radar_noise_std={'range': 50.0, 'velocity': 5.0, 'azimuth': 0.005},
        irst_noise_std=0.1
    )

    target_hist = []
    missile_hist = []

    for step in range(num_steps):
        # Target does a 4G turn
        target.update(dt, ('turn', {'g_force': 4.0}))

        # (Optionally get noisy obs, but we don't use it here)
        _obs = sensor.observe(target, missile.position)

        # SIMPLE GUIDANCE: use TRUE target state (no PF, no quantum)
        missile.update(dt, target.position, target.velocity)

        target_hist.append(target.position.copy())
        missile_hist.append(missile.position.copy())

    target_hist = np.array(target_hist)
    missile_hist = np.array(missile_hist)

    miss = np.linalg.norm(target_hist[-1] - missile_hist[-1])
    print(f"[demo_simple_guidance] Final miss distance: {miss:.2f} m")

    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(target_hist[:, 0], target_hist[:, 1], target_hist[:, 2], 'r-', label='Target')
    ax.plot(missile_hist[:, 0], missile_hist[:, 1], missile_hist[:, 2], 'b-', label='Missile')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Simple Guidance (Truth-based)")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()
