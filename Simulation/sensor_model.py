import numpy as np
from typing import Dict, Tuple
from Simulation.target_dynamics import Target # Import the Target class

class Sensor:
    """
    Simulates a sensor suite (e.g., AESA Radar, IRST) observing a target.

    This class takes the ground truth state of a Target object and generates
    realistic, noisy sensor measurements that can be used as input for a
    predictive model.

    Attributes:
        radar_noise_std (Dict[str, float]): Standard deviations for radar noise.
        irst_noise_std (float): Standard deviation for IRST noise.
    """

    def __init__(self, radar_noise_std: Dict[str, float], irst_noise_std: float) -> None:
        """
        Initializes the sensor suite with noise parameters.

        Args:
            radar_noise_std: A dict defining the noise for radar measurements.
                             Example: {'range': 50.0, 'velocity': 5.0, 'azimuth': 0.005}
            irst_noise_std: The noise level for thermal intensity measurements.
        """
        self.radar_noise_std = radar_noise_std
        self.irst_noise_std = irst_noise_std

    def _simulate_radar(self, target_state: Tuple, ownship_position: np.ndarray) -> Dict[str, float]:
        """
        Simulates AESA radar measurements.

        Args:
            target_state: The ground truth state (pos, vel, acc) of the target.
            ownship_position: The 3D position of the friendly aircraft.

        Returns:
            A dictionary of noisy radar measurements.
        """
        target_pos, target_vel, _ = target_state
        
        relative_pos = target_pos - ownship_position
        true_range = np.linalg.norm(relative_pos)
        # xy_norm = np.linalg.norm(relative_pos[:2])
        true_elevation = np.arctan2(relative_pos[2], np.linalg.norm(relative_pos[0:2]) + 1e-6)

        relative_vel = target_vel
        true_closing_velocity = -np.dot(relative_vel, relative_pos) / true_range if true_range > 0 else 0.0
        
        true_azimuth = np.arctan2(relative_pos[1], relative_pos[0])
        
        # Add Gaussian noise
        noisy_range = true_range + np.random.normal(0, self.radar_noise_std['range'])
        noisy_closing_velocity = true_closing_velocity + np.random.normal(0, self.radar_noise_std['velocity'])
        noisy_azimuth = true_azimuth + np.random.normal(0, self.radar_noise_std['azimuth'])
        noisy_elevation = true_elevation + np.random.normal(
            0, self.radar_noise_std.get('elevation', 0.005)
        )

        # return {
        #     'range': float(noisy_range),
        #     'closing_velocity': float(noisy_closing_velocity),
        #     'azimuth': float(noisy_azimuth),
        #     'elevation': float(noisy_elevation)
        # }
        # Convert noisy spherical radar measurement to Cartesian (sensor-level)
        x = noisy_range * np.cos(noisy_elevation) * np.cos(noisy_azimuth)
        y = noisy_range * np.cos(noisy_elevation) * np.sin(noisy_azimuth)
        z = noisy_range * np.sin(noisy_elevation)

        return {
            'range': float(noisy_range),
            'closing_velocity': float(noisy_closing_velocity),
            'azimuth': float(noisy_azimuth),
            'elevation': float(noisy_elevation),
            'position': np.array([x, y, z])
        }


    def _simulate_irst(self, target_state: Tuple) -> Dict[str, float]:
        """
        Simulates IRST (Infrared Search and Track) measurements.

        Args:
            target_state: The ground truth state (pos, vel, acc) of the target.
        
        Returns:
            A dictionary of noisy thermal measurements.
        """
        _, _, target_acc = target_state
        
        base_intensity = 1.0 
        
        g_force = np.linalg.norm(target_acc) / 9.81
        afterburner_effect = max(0, g_force - 1.0)
        
        true_intensity = base_intensity + afterburner_effect
        
        noisy_intensity = true_intensity + np.random.normal(0, self.irst_noise_std)
        
        return {'thermal_intensity': float(noisy_intensity)}

    def observe(self, target: Target, ownship_position: np.ndarray) -> Dict[str, float]:
        """
        Performs a full observation of a target from an ownship position.

        Args:
            target: An instance of the Target class.
            ownship_position: The 3D position of the friendly (ownship) aircraft.

        Returns:
            A dictionary containing all simulated sensor measurements.
        """
        target_state = target.get_state()
        
        radar_data = self._simulate_radar(target_state, ownship_position)
        irst_data = self._simulate_irst(target_state)
        
        observation = {**radar_data, **irst_data}
        
        return observation