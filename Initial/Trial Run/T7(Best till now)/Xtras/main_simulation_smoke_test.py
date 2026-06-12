import sys
import numpy as np
import pyqtgraph.opengl as gl
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QThread, pyqtSignal
from typing import Dict, Any

# --- Import all project modules ---
from Simulation.target_dynamics import Target
from Simulation.sensor_model import Sensor
from Quantum_Core.qneat import Genome
from Quantum_Core.nqpf_placeholder import NQPF
from Quantum_Core.qaoa import build_quadratic_program, QAOAOptimizer
from config import SimOptions, QNEATOptions, QAOAOptions

class SimulationWorker(QThread):
    """
    Runs the entire simulation in a separate thread to keep the GUI responsive.
    """
    update_plot = pyqtSignal(np.ndarray, np.ndarray)
    simulation_finished = pyqtSignal(str)

    def __init__(self, sim_options, qneat_options, qaoa_options, target_params, missile_params, graphics_only=False):
        super().__init__()
        self.sim_options = sim_options
        self.qneat_options = qneat_options
        self.qaoa_options = qaoa_options
        self.target_params = target_params
        self.missile_params = missile_params
        self.graphics_only = graphics_only

    def run(self):
        """The main simulation logic that runs in the background."""
        try: # --- ADD THIS TRY BLOCK ---
            print("--- Simulation Thread Started ---")
            
            dt = self.sim_options.dt
            timesteps = int(self.sim_options.total_time / dt)
            
            target = Target(initial_position=self.target_params['initial_pos'], initial_velocity=self.target_params['initial_vel'])
            missile_pos = np.array(self.missile_params['initial_pos'], dtype=float)
            missile_trajectory = [missile_pos.copy()]
            
            sensor = Sensor(radar_noise_std={'range': 50.0, 'velocity': 5.0, 'azimuth': 0.005}, irst_noise_std=0.1)
            
            placeholder_genome = Genome(num_qubits=16, options=self.qneat_options, global_innovation_counter={'count': 0}, global_innovation_dict={})
            nqpf = NQPF(num_particles=self.sim_options.num_particles, trained_genome=placeholder_genome)
            nqpf.initialize_swarm(initial_state_estimate=np.concatenate([target.position, target.velocity]), initial_uncertainty=100.0)
            
            qaoa_optimizer = QAOAOptimizer(options=dict(vars(self.qaoa_options)))

            for step in range(timesteps):
                maneuver = ('turn', {'g_force': 9.0}) if step * dt > 20 else ('straight', {})
                target.update(dt, maneuver)

                if not self.graphics_only:
                    observation = sensor.observe(target, ownship_position=np.zeros(3))
                    nqpf.predict(dt)
                    nqpf.update(observation)
                    nqpf.resample()
                    
                    if step % int(2 / dt) == 0:
                        print(f"Step {step}: Re-optimizing trajectory...")
                        prob_forecast = np.random.rand(8, 3)
                        waypoints = np.random.rand(8, 3, 3) * 10000 + target.position
                        
                        qp = build_quadratic_program(prob_forecast, missile_pos, waypoints, dict(vars(self.qaoa_options)))
                        solution_vector, cost = qaoa_optimizer.optimize(qp)
                        print(f"    -> New Optimized Cost: {cost:.4f}")
                        new_trajectory = qaoa_optimizer.decode_solution(solution_vector, waypoints)
                        
                        if len(new_trajectory) > 0:
                            missile_pos = new_trajectory[0]
                else:
                    direction_to_target = (target.position - missile_pos)
                    missile_pos += direction_to_target * 0.01

                missile_trajectory.append(missile_pos.copy())
                self.update_plot.emit(np.array(target.trajectory), np.array(missile_trajectory))
                self.msleep(int(dt * 100))

                if np.linalg.norm(missile_pos - target.position) < 1000:
                    self.simulation_finished.emit(f"--- INTERCEPT at step {step} ---")
                    return

            self.simulation_finished.emit("--- Simulation Complete (Time Limit Reached) ---")

        except Exception as e:
            # --- THIS WILL CATCH AND PRINT THE HIDDEN ERROR ---
            import traceback
            print("--- AN ERROR OCCURRED IN THE SIMULATION THREAD ---")
            traceback.print_exc()
            self.simulation_finished.emit("--- ERROR: Simulation Failed ---")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    GRAPHICS_ONLY_MODE = True
    view = gl.GLViewWidget()
    view.setWindowTitle('Engagement Simulation')
    view.setCameraPosition(distance=120000)
    view.show()
    grid = gl.GLGridItem()
    view.addItem(grid)
    target_plot = gl.GLLinePlotItem(color=(1, 0, 0, 1), width=3, antialias=True)
    missile_plot = gl.GLLinePlotItem(color=(0, 0, 1, 1), width=3, antialias=True)
    view.addItem(target_plot)
    view.addItem(missile_plot)

    def update_plot_data(target_traj, missile_traj):
        target_plot.setData(pos=target_traj)
        missile_plot.setData(pos=missile_traj)

    def on_simulation_finish(message):
        print(message)
        
    sim_opts = SimOptions()
    qneat_opts = QNEATOptions()
    qaoa_opts = QAOAOptions()
    target_cfg = {'initial_pos': [100000, 20000, 10000], 'initial_vel': [-300, 0, 0]}
    missile_cfg = {'initial_pos': [0, 0, 10000]}

    worker = SimulationWorker(sim_opts, qneat_opts, qaoa_opts, target_cfg, missile_cfg, graphics_only=GRAPHICS_ONLY_MODE)
    worker.update_plot.connect(update_plot_data)
    worker.simulation_finished.connect(on_simulation_finish)
    worker.start()

    print("--- Main GUI thread is running. Simulation is running in the background. ---")
    sys.exit(app.exec())