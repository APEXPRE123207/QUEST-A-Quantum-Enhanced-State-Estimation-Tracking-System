# import matplotlib.pyplot as plt
# import numpy as np
# import pickle
# import os
# import time
# from Simulation.target_dynamics import Target
# from Simulation.missile_dynamics import Missile
# from Simulation.sensor_model import Sensor
# from config import SimOptions, SensorOptions
# from Quantum_Core.nqpf import NQPF

# SAVE_PATH = "population_state.pkl"

# def watch_training():
#     print("--- Watching Training Progress ---")
#     print(f"Waiting for '{SAVE_PATH}'...")

#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     last_mod_time = 0

#     while plt.fignum_exists(fig.number):  # stop if window closed
#         try:
#             if not os.path.exists(SAVE_PATH):
#                 time.sleep(1)
#                 continue

#             mod_time = os.path.getmtime(SAVE_PATH)
#             if mod_time <= last_mod_time:
#                 time.sleep(1)
#                 continue

#             last_mod_time = mod_time

#             # Load population with retry logic
#             for _ in range(5):
#                 try:
#                     with open(SAVE_PATH, 'rb') as f:
#                         population = pickle.load(f)
#                     break
#                 except Exception:
#                     time.sleep(0.1)
#             else:
#                 continue

#             if not population.population:
#                 continue

#             best_genome = max(population.population, key=lambda g: g.fitness)
#             print(f"[Viewer] Gen {population.generation} | Best fitness: {best_genome.fitness:.4f}")

#             # Run Simulation
#             sim_opts = SimOptions()
#             sensor_opts = SensorOptions()

#             target = Target([10000, 0, 10000], [-250, 50, 0])
#             sensor = Sensor(
#                 radar_noise_std={'range': sensor_opts.range_std,
#                                  'velocity': sensor_opts.velocity_std,
#                                  'azimuth': sensor_opts.azimuth_std},
#                 irst_noise_std=sensor_opts.irst_std
#             )
#             nqpf = NQPF(num_particles=sim_opts.num_particles, trained_genome=best_genome)

#             missile = Missile(
#                 initial_position=np.array([0.0, 0.0, 9000.0]),
#                 initial_velocity=np.array([300.0, 0.0, 0.0])
#             )



#             initial_est = np.array([
#                 target.position[0], target.position[1], target.position[2],
#                 target.velocity[0], target.velocity[1], target.velocity[2]
#             ])
#             nqpf.initialize_swarm(initial_est, 100.0)

#             target_hist = []
#             missile_hist = []

#             for _ in range(50):
#                 target.update(sim_opts.dt, ('turn', {'g_force': 4.0}))
#                 obs = sensor.observe(target, missile.position)
#                 nqpf.predict(dt=sim_opts.dt)
#                 nqpf.update(obs, missile.position)

#                 missile.update(sim_opts.dt, target.position, target.velocity)

#                 target_hist.append(target.position.copy())
#                 missile_hist.append(missile.position.copy())

#             target_hist = np.array(target_hist)
#             missile_hist = np.array(missile_hist)

#             ax.clear()
#             ax.plot(target_hist[:, 0], target_hist[:, 1], target_hist[:, 2], 'r-', label='Target')
#             ax.plot(missile_hist[:, 0], missile_hist[:, 1], missile_hist[:, 2], 'b-', label='Missile')
#             ax.set_title(f"Gen {population.generation} | Best Fitness: {best_genome.fitness:.2f}")
#             ax.legend()
#             plt.pause(0.1)

#         except Exception as e:
#             print(f"[Viewer Error]: {e}")
#             time.sleep(1)

# if __name__ == "__main__":
#     plt.ion()
#     plt.show(block=False)
#     watch_training()

import matplotlib.pyplot as plt
import pickle
import os
import time

SAVE_PATH = "population_state.pkl"

def watch_training():
    print("--- Watching Training Progress (Fitness Only) ---")
    print(f"Waiting for '{SAVE_PATH}'...")

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 5))
    generations = []
    best_fitnesses = []
    last_mod_time = 0

    while True:
        # If figure is closed, stop
        if not plt.fignum_exists(fig.number):
            print("Figure closed, stopping watcher.")
            break

        try:
            if not os.path.exists(SAVE_PATH):
                time.sleep(1)
                continue

            mod_time = os.path.getmtime(SAVE_PATH)
            if mod_time <= last_mod_time:
                time.sleep(0.5)
                continue

            last_mod_time = mod_time

            # Try to load population
            for _ in range(5):
                try:
                    with open(SAVE_PATH, "rb") as f:
                        population = pickle.load(f)
                    break
                except Exception:
                    time.sleep(0.1)
            else:
                continue

            if not population.population:
                time.sleep(0.5)
                continue

            best_genome = max(population.population, key=lambda g: g.fitness)
            gen = getattr(population, "generation", len(generations))

            print(f"[Viewer] Gen {gen} | Best fitness: {best_genome.fitness:.4f}")

            generations.append(gen)
            best_fitnesses.append(best_genome.fitness)

            # Update 2D plot
            ax.clear()
            ax.plot(generations, best_fitnesses, marker="o")
            ax.set_xlabel("Generation")
            ax.set_ylabel("Best Fitness")
            ax.set_title("Training Progress")
            ax.grid(True)

            plt.pause(0.1)

        except Exception as e:
            print(f"[Viewer Error]: {e}")
            time.sleep(1)

if __name__ == "__main__":
    watch_training()
