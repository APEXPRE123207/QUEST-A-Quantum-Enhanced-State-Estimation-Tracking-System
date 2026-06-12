import numpy as np
from typing import List
from config import SimOptions

class Missile:
    """
    Simple 3-DOF missile kinematics model.
    - Point-mass with position, velocity, acceleration
    - Uses a basic pure-pursuit style guidance in update()
    """

    def __init__(self,
                 initial_position: List[float],
                 initial_velocity: List[float]) -> None:
        if len(initial_position) != 3 or len(initial_velocity) != 3:
            raise ValueError("Position and velocity must be 3D vectors.")

        self.position = np.array(initial_position, dtype=float)
        self.velocity = np.array(initial_velocity, dtype=float)
        self.acceleration = np.zeros(3, dtype=float)

        sim_opts = SimOptions()
        self.max_speed = sim_opts.missile_max_speed
        self.max_accel = sim_opts.missile_max_accel

    def update(self, dt: float,
               target_position: np.ndarray,
               target_velocity: np.ndarray) -> None:
        """
        Very simple pure-pursuit-like update:
        - Accelerate toward line-of-sight to target
        - Respect max accel and max speed
        """

        target_position = np.array(target_position, dtype=float)
        rel_pos = target_position - self.position
        distance = np.linalg.norm(rel_pos)

        if distance < 1e-6:
            # Already on top of target
            self.acceleration[:] = 0.0
            return

        # Direction from missile to target
        los_dir = rel_pos / distance

        # Try to align velocity with LOS while obeying speed & accel limits
        current_speed = np.linalg.norm(self.velocity)
        if current_speed < 1e-3:
            current_speed = self.max_speed * 0.5  # give it some speed to start

        desired_speed = min(self.max_speed, current_speed + self.max_accel * dt)
        desired_vel = los_dir * desired_speed

        # Commanded acceleration to move current v → desired v
        accel_cmd = (desired_vel - self.velocity) / dt
        accel_norm = np.linalg.norm(accel_cmd)

        if accel_norm > self.max_accel:
            accel_cmd = accel_cmd / accel_norm * self.max_accel

        self.acceleration = accel_cmd

        # Integrate
        self.velocity += self.acceleration * dt
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed and speed > 0:
            self.velocity = self.velocity / speed * self.max_speed

        self.position += self.velocity * dt
