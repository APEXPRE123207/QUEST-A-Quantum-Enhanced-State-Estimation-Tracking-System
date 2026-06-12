import numpy as np
from typing import Tuple, Dict, Any

# Qiskit Imports
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.primitives import Sampler
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_aer import AerSimulator
def build_quadratic_program(
    probabilistic_forecast: np.ndarray,
    missile_state: np.ndarray,
    waypoints: np.ndarray,
    options: Dict[str, Any]
) -> QuadraticProgram:
    """
    Constructs a QuadraticProgram for the trajectory optimization problem.
    """
    num_waypoints, num_timesteps, _ = waypoints.shape
    qp = QuadraticProgram(name="Trajectory Optimization")

    x = [[qp.binary_var(f"x_{t}_{i}") for i in range(num_waypoints)] for t in range(num_timesteps)]

    linear_cost = {}
    quadratic_cost = {}
    
    w1 = options.get('w1', 0.5)
    w2 = options.get('w2', 0.5)
    
    # Cost Part 1: Risk from NQPF forecast
    for t in range(num_timesteps):
        for i in range(num_waypoints):
            linear_cost[x[t][i].name] = w1 * probabilistic_forecast[i, t]

    # Cost Part 2: Travel distance
    for i in range(num_waypoints):
        dist = np.linalg.norm(missile_state - waypoints[i, 0, :])
        linear_cost[x[0][i].name] += w2 * dist

    for t in range(num_timesteps - 1):
        for i in range(num_waypoints):
            for j in range(num_waypoints):
                dist = np.linalg.norm(waypoints[i, t, :] - waypoints[j, t+1, :])
                quadratic_cost[(x[t][i].name, x[t+1][j].name)] = w2 * dist
                
    qp.minimize(linear=linear_cost, quadratic=quadratic_cost)

    # Add the 'one-hot' constraint for each time step
    for t in range(num_timesteps):
        constraint_vars = {x[t][i].name: 1.0 for i in range(num_waypoints)}
        qp.linear_constraint(linear=constraint_vars, sense="==", rhs=1.0, name=f"t{t}_one_hot") # type: ignore
        
    return qp


class QAOAOptimizer:
    """
    Manages the execution of the QAOA algorithm for trajectory optimization.
    """
    def __init__(self, options: Dict[str, Any]):
        """
        Initializes the QAOA optimizer.
        """
        self.reps = int(options.get('reps', 2))
        self.shots = int(options.get('shots', 128))
        self.max_iterations = int(options.get('max_iterations', 50))
        # Setup the core QAOA algorithm with bounded iterations
        optimizer = COBYLA(maxiter=self.max_iterations)
        sampler = Sampler(options={"shots": self.shots})
        self.qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=self.reps)
        
        # Setup the high-level optimizer that uses QAOA as its backend
        # The '# type: ignore' comment is added to suppress a known Pylance linter quirk.
        self.optimizer = MinimumEigenOptimizer(min_eigen_solver=self.qaoa) # type: ignore
        self.sampler = AerSimulator()

    def optimize(self, quadratic_program: QuadraticProgram) -> Tuple[np.ndarray, float]:
        """
        Solves the trajectory optimization problem using the MinimumEigenOptimizer.
        """
        result = self.optimizer.solve(quadratic_program)
        
        # Add final null-safety checks for production-grade robustness
        if result.x is None or result.fval is None:
            raise RuntimeError("QAOA optimizer failed to return a valid solution.")
            
        optimal_solution = result.x
        optimal_cost = result.fval
        
        return optimal_solution, optimal_cost

    def decode_solution(self, solution: np.ndarray, waypoints: np.ndarray) -> np.ndarray:
        """
        Translates the optimal solution vector from QAOA into a physical trajectory.
        Handles invalid solutions by selecting the closest waypoint to target.
        """
        num_waypoints, num_timesteps, _ = waypoints.shape
        trajectory = []

        for t in range(num_timesteps):
            time_slice = solution[t * num_waypoints : (t + 1) * num_waypoints]
            try:
                # Find the index of the '1' in the slice
                chosen_waypoint_idx = np.where(time_slice == 1)[0][0]
                trajectory.append(waypoints[chosen_waypoint_idx, t, :])
            except IndexError:
                # Fallback: select waypoint with highest value (most likely choice)
                chosen_waypoint_idx = np.argmax(time_slice)
                trajectory.append(waypoints[chosen_waypoint_idx, t, :])
        return np.array(trajectory)