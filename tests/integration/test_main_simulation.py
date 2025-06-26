import os
import tempfile
import unittest
import yaml
import subprocess
import sys
from pathlib import Path


class TestMainSimulationIntegration(unittest.TestCase):
    """Integration test for the main simulation pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        # Get the project root directory
        self.project_root = Path(__file__).parent.parent.parent
        self.main_py_path = self.project_root / "src" / "main.py"

        # Create a minimal test configuration
        self.test_config = {
            "constellation": {
                "name": "TestConstellation",
                "num_orbits": 2,
                "num_sats_per_orbit": 2,
                "phase_diff": True,
                "inclination_degree": 60,
                "eccentricity": 0.0000001,
                "arg_of_perigee_degree": 0.0,
                "mean_motion_rev_per_day": 15.19,
                "tle_output_filename": "test_constellation.txt"
            },
            "simulation": {
                "dynamic_state_algorithm": "shortest_path_link_state",
                "end_time_hours": 1,  # Very short simulation
                "time_step_minutes": 30,  # Only 2 time steps
                "offset_ns": 0
            },
            "satellite": {
                "altitude_m": 600000,
                "cone_angle_degrees": 29.0
            },
            "earth": {
                "radius_m": 6378135.0,
                "isl_min_altitude_m": 80000
            },
            "ground_stations": [
                {
                    "name": "TestGS1",
                    "latitude": 0.0,
                    "longitude": 0.0,
                    "elevation_m": 0.0
                },
                {
                    "name": "TestGS2",
                    "latitude": 45.0,
                    "longitude": 90.0,
                    "elevation_m": 0.0
                }
            ],
            "network": {
                "gsl_interfaces": {
                    "number_of_interfaces": 1,
                    "aggregate_max_bandwidth": 1.0
                }
            },
            "logging": {
                "is_debug": False,
                "file_name": "test_simulation.log"
            }
        }

    def test_end_to_end_simulation_execution(self):
        """Test that the main simulation runs successfully with a minimal topology."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config file in temp directory
            config_path = os.path.join(temp_dir, "test_config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(self.test_config, f)

            # Run the simulation
            cmd = [
                sys.executable,
                str(self.main_py_path),
                "--config",
                config_path
            ]

            # Change to temp directory to contain generated files
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)

                # Execute the simulation
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60  # 1 minute timeout
                )

                # Check that simulation completed successfully
                self.assertEqual(
                    result.returncode,
                    0,
                    f"Simulation failed with return code {result.returncode}.\n"
                    f"STDOUT: {result.stdout}\n"
                    f"STDERR: {result.stderr}"
                )

                # Check that log file was created
                log_files = [f for f in os.listdir(temp_dir) if f.startswith("test_simulation") and f.endswith(".log")]
                self.assertGreater(
                    len(log_files),
                    0,
                    f"No log file found. Files in temp dir: {os.listdir(temp_dir)}"
                )

                # Check that TLE file was created
                tle_file = "test_constellation.txt"
                self.assertTrue(
                    os.path.exists(tle_file),
                    f"TLE file {tle_file} was not created. Files in temp dir: {os.listdir(temp_dir)}"
                )

                # Check log content for success indicators
                log_file_path = os.path.join(temp_dir, log_files[0])
                with open(log_file_path, 'r') as f:
                    log_content = f.read()

                # Verify key simulation steps completed
                self.assertIn("Logger initialized", log_content, "Logger was not initialized")
                self.assertIn("Created 4 Satellite objects", log_content, "Satellites were not created")
                self.assertIn("Created 2 GroundStation objects", log_content, "Ground stations were not created")
                self.assertIn("Starting dynamic state generation", log_content, "Dynamic state generation did not start")
                self.assertIn("Simulation finished. ✅", log_content, "Simulation did not complete successfully")

            finally:
                os.chdir(original_cwd)

    def test_simulation_with_invalid_config(self):
        """Test that the simulation fails gracefully with invalid configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create an invalid config (missing required fields)
            invalid_config = {"invalid": "config"}
            config_path = os.path.join(temp_dir, "invalid_config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(invalid_config, f)

            # Run the simulation
            cmd = [
                sys.executable,
                str(self.main_py_path),
                "--config",
                config_path
            ]

            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)

                # Execute the simulation - should fail
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                # Check that simulation failed as expected
                self.assertNotEqual(
                    result.returncode,
                    0,
                    "Simulation should have failed with invalid config"
                )

            finally:
                os.chdir(original_cwd)

    def test_simulation_with_nonexistent_config(self):
        """Test that the simulation fails gracefully when config file doesn't exist."""
        # Run with non-existent config file
        cmd = [
            sys.executable,
            str(self.main_py_path),
            "--config",
            "nonexistent_config.yaml"
        ]

        # Execute the simulation - should fail
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10
        )

        # Check that simulation failed as expected
        self.assertNotEqual(
            result.returncode,
            0,
            "Simulation should have failed with nonexistent config"
        )

        # Check that appropriate error message is shown
        self.assertIn(
            "Configuration file not found",
            result.stdout + result.stderr,
            "Expected error message not found in output"
        )


if __name__ == "__main__":
    unittest.main()
