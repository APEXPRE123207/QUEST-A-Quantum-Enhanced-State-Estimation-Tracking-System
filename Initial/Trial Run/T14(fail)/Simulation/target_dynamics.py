import numpy as np
from typing import List, Tuple, Dict, Any

class Target:
    """
    Manages and simulates the state and dynamics of a target aircraft.

    This class provides a production-grade framework for simulating the 3D
    kinematics of an agile target. It supports various flight maneuvers and
    maintains a history of its trajectory for data generation.

    Attributes:
        position (np.ndarray): The current 3D position vector [x, y, z] in meters.
        velocity (np.ndarray): The current 3D velocity vector [vx, vy, vz] in m/s.
        acceleration (np.ndarray): The current 3D acceleration vector [ax, ay, az] in m/s^2.
        trajectory (List[np.ndarray]): A time-ordered list of position vectors.
    """

    def __init__(self,
                 initial_position: List[float],
                 initial_velocity: List[float]) -> None:
        """
        Initializes the target's state.

        Args:
            initial_position: 3D list or tuple [x, y, z] for initial position.
            initial_velocity: 3D list or tuple [vx, vy, vz] for initial velocity.
        """
        if len(initial_position) != 3 or len(initial_velocity) != 3:
            raise ValueError("Position and velocity must be 3D vectors.")

        self.position: np.ndarray = np.array(initial_position, dtype=float)
        self.velocity: np.ndarray = np.array(initial_velocity, dtype=float)
        self.acceleration: np.ndarray = np.zeros(3, dtype=float)
        self.trajectory: List[np.ndarray] = [self.position.copy()]
        self._time: float = 0.0 # Internal clock for maneuvers

    def _execute_straight_flight(self, dt: float) -> None:
        """Applies a simple linear motion model."""
        self.acceleration = np.zeros(3, dtype=float)
        self.position += self.velocity * dt

    def _execute_coordinated_turn(self, dt: float, g_force: float) -> None:
        """
        Executes a constant-G coordinated turn in the horizontal plane (x-y).

        Args:
            dt: The simulation time step.
            g_force: The G-force applied in the turn (e.g., 9.0).
        """
        speed = np.linalg.norm(self.velocity)
        if speed == 0:
            return

        acceleration_magnitude = g_force * 9.81
        turn_radius = (speed ** 2) / acceleration_magnitude
        turn_rate = speed / turn_radius

        current_heading = np.arctan2(self.velocity[1], self.velocity[0])
        new_heading = current_heading + turn_rate * dt

        self.velocity[0] = speed * np.cos(new_heading)
        self.velocity[1] = speed * np.sin(new_heading)
        self.position += self.velocity * dt
        
        # Update acceleration (centripetal)
        accel_direction = -np.array([np.cos(current_heading), np.sin(current_heading), 0])
        self.acceleration = acceleration_magnitude * accel_direction

    def _execute_jink(self, dt: float, frequency: float, amplitude: float) -> None:
        """
        Executes a defensive jinking (side-to-side) maneuver.

        Args:
            dt: The simulation time step.
            frequency: The frequency of the sine wave for the jink (in Hz).
            amplitude: The amplitude of the lateral acceleration (in m/s^2).
        """
        # Calculate lateral acceleration based on a sine wave
        lateral_accel = amplitude * np.sin(2 * np.pi * frequency * self._time)
        
        # Get the direction perpendicular to the velocity vector in the horizontal plane
        velocity_direction = self.velocity / np.linalg.norm(self.velocity)
        lateral_direction = np.array([-velocity_direction[1], velocity_direction[0], 0])
        
        self.acceleration = lateral_accel * lateral_direction
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt

    def _execute_climb_dive(self, dt: float, vertical_g: float) -> None:
        """
        Executes a climb or dive maneuver.

        Args:
            dt: The simulation time step.
            vertical_g: The G-force to apply vertically (+ for climb, - for dive).
        """
        vertical_accel = vertical_g * 9.81
        self.acceleration = np.array([0, 0, vertical_accel])
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt

    def update(self, dt: float, maneuver: Tuple[str, Dict[str, Any]]) -> None:
        """
        Updates the target's state by executing a given maneuver.

        Args:
            dt: The time step for the simulation.
            maneuver: A tuple describing the maneuver and its parameters.
                      Examples: ('straight', {})
                                ('turn', {'g_force': 9.0})
                                ('jink', {'frequency': 1.0, 'amplitude': 50.0})
                                ('climb_dive', {'vertical_g': 3.0})
        """
        maneuver_type, params = maneuver
        self._time += dt # Update internal clock

        if maneuver_type == 'straight':
            self._execute_straight_flight(dt)
        elif maneuver_type == 'turn':
            self._execute_coordinated_turn(dt, **params)
        elif maneuver_type == 'jink':
            self._execute_jink(dt, **params)
        elif maneuver_type == 'climb_dive':
            self._execute_climb_dive(dt, **params)
        else:
            raise ValueError(f"Unknown maneuver type: {maneuver_type}")

        self.trajectory.append(self.position.copy())

    def get_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns the current full kinematic state of the target."""
        return self.position, self.velocity, self.acceleration