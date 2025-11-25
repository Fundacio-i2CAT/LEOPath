# LEO Routing Simulation Framework

<img src="i2cat_logo.png" alt="i2CAT Logo" width="150"/>
<img src="lrsim_logo.png" alt="LRSIM Logo" width="150"/>

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A simple, user-friendly and extensible simulation framework for analyzing and comparing routing algorithms in Low Earth Orbit (LEO) satellite constellations.

This tool enables researchers and engineers to study network dynamics, routing performance, and connectivity patterns in satellite networks such as Starlink, Kuiper, and other LEO constellations.

## Table of Contents

- [LEO Routing Simulation Framework](#leo-routing-simulation-framework)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Why LRSIM?](#why-lrsim)
    - [What LRSIM Does](#what-lrsim-does)
    - [What LRSIM Does Not Do](#what-lrsim-does-not-do)
  - [Features](#features)
  - [Prerequisites](#prerequisites)
    - [For Docker Deployment (Recommended)](#for-docker-deployment-recommended)
    - [For Local Development](#for-local-development)
  - [Installation](#installation)
    - [Using Docker (Recommended)](#using-docker-recommended)
    - [Local Installation](#local-installation)
  - [Quick Start](#quick-start)
  - [Usage](#usage)
    - [Running Simulations](#running-simulations)
    - [Configuration](#configuration)
    - [Visualization](#visualization)
  - [Project Structure](#project-structure)
  - [Routing Algorithms](#routing-algorithms)
    - [1. Shortest Path Link-State Routing](#1-shortest-path-link-state-routing)
    - [2. Topological Routing](#2-topological-routing)
    - [Adding Custom Routing Algorithms](#adding-custom-routing-algorithms)
  - [Output Format](#output-format)
  - [Testing](#testing)
  - [Contributing](#contributing)
    - [Reporting Issues](#reporting-issues)
    - [Pull Requests](#pull-requests)
    - [Code Style](#code-style)
    - [Adding New Features](#adding-new-features)
  - [Copyright](#copyright)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)
  - [Attributions](#attributions)
  - [Contact](#contact)

## Overview

This project provides a simple yet complete toolkit for quickly simulating and analyzing LEO satellite network routing running in commodity hardware.

* **Very easy to run**: Can run everywhere using docker containers.
* **Constellation Simulation**: Model realistic LEO satellite constellations with configurable parameters (altitude, inclination, number of satellites, etc.). Just edit te configuration file and that's it.
* **Real or Custom Constellations**: Simulate well-known constellations like Starlink using the data from Celestrak or create custom configurations just with orbital parameters with synthetic TLE generation.
* **Routing Algorithms**: Compare different routing strategies and GSL attachment policies through a pluggable architecture.
* **Network Dynamics**: Simulate time-varying network states with dynamic Inter-Satellite Links (ISLs) and Ground Station Links (GSLs)
* **Visualization**: Generate interactive 3D visualizations of satellite orbits, connectivity patterns, and ground station coverage using Cesium

The framework currently supports the following routing algorithms:
- **Shortest Path Link-State Routing**: Traditional Dijkstra-based routing over satellite networks
- **Topological Routing**: Novel routing algorithm based on the [6G-RUPA](https://6grupa.com) architecture for improved scalability and reduced control overhead

## Why LRSIM?

LRSIM is essentially a fork of an existing simulator called [Hypatia](https://github.com/snkas/hypatia). While Hypatia provides valuable foundations for satellite network simulation, it has limitations when executing it or extending it, such as tightly coupled routing logic that makes it difficult to implement and compare new protocols or inflexible data output formats that complicate analysis.

LRSIM uses Hypatia core logic but makes it extremely easy to use, maintain and extend.

### What LRSIM Does

- **Satellite Movement Simulation**: Modeling of satellite orbital mechanics using SGP4 propagation model.
- **Dynamic Link Management**: Automatic management of Inter-Satellite Links (ISLs) and Ground-to-Satellite Links (GSLs) based on visibility and distance constraints
- **Synthetic TLE Generation**: Creates Two-Line Element sets from orbital parameters, enabling simulation of arbitrary constellation configurations
- **Pluggable GSL Attachment Strategies**: Flexible framework for implementing different ground station attachment policies (currently includes nearest satellite strategy with extensible architecture)
- **Modular Routing Protocol Framework**: Clean separation of routing logic allowing easy implementation and comparison of different routing protocols, including:
  - Traditional link-state shortest path routing as a baseline
  - Novel topological routing scheme with hierarchical addressing
  - Easy extension for custom routing algorithms

### What LRSIM Does Not Do

LRSIM focuses on **network topology and routing state generation**, not full protocol stack simulation. It does not simulate:
- Complete network protocol stacks (TCP/IP, application layers)
- Packet-level processing and queueing
- Physical layer details (modulation, coding, interference)

For packet-level simulations, LRSIM's forwarding state output can be integrated with network simulators like NS-3 (as demonstrated in Hypatia), enabling end-to-end performance evaluation when needed.


## Features

- 🛰️ **Realistic Satellite Modeling**: Uses Two-Line Element (TLE) sets and SGP4 propagation for accurate orbital mechanics
- 🌐 **Dynamic Network State**: Simulates time-varying network topology with configurable time steps
- 📊 **Multiple Routing Algorithms**: Pluggable routing algorithm architecture for easy comparison and extension
- 🗺️ **Ground Station Integration**: Support for multiple ground stations with realistic visibility constraints
- 🎨 **3D Visualization**: Interactive web-based visualization using Cesium for constellation analysis
- 🐳 **Docker Support**: Containerized deployment for reproducible simulations
- 📝 **Flexible Configuration**: YAML-based configuration for easy simulation setup

## Prerequisites

### For Docker Deployment (Recommended)

The `run-simulations.sh` script handles all Docker operations automatically. You only need:

- Docker Engine 20.10+
- Docker Compose 1.29+
- Bash shell (available by default on Linux/macOS)

### For Local Development

- Python 3.8 or higher
- pip (Python package installer)
- virtualenv (recommended)

## Installation

### Using Docker (Recommended)

The easiest way to run simulations is using the `run-simulations.sh` convenience script, which handles Docker operations automatically:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Fundacio-i2CAT/leo-routing-simu.git
   cd leo-routing-simu
   ```

2. **Run simulations with the convenience script**:
   ```bash
   # Run simulation with a specific configuration
   ./run-simulations.sh run -c src/config/ether_simple.yaml

   # Generate visualization for the simulation
   ./run-simulations.sh visualise -c src/config/ether_simple.yaml

   # Clean up all generated files and containers
   ./run-simulations.sh clean
   ```

   The script provides:
   - Automatic Docker image building on first run
   - Output directory creation and management
   - Configuration file validation
   - Color-coded status messages
   - Built-in HTTP server for visualizations (port 8080)

3. **Alternatively, use Docker Compose directly**:
   ```bash
   # Build the Docker image
   docker-compose build

   # Run a simulation
   docker compose run --rm leo-routing-simu src/config/ether_simple.yaml

   # Generate visualization
   docker compose run --rm --entrypoint "python -m src.satellite_visualisation.cesium_builder.main" leo-routing-viz src/config/ether_simple.yaml

   # Start web server to view results
   docker compose up -d viz-server
   ```

**Output Organization**:

All outputs are automatically organized in the `output/` directory:
- `output/logs/` - Simulation logs
- `output/simulation/` - Simulation results and forwarding states
- `output/tles/` - Generated TLE files for satellite orbits
- `output/visualizations/` - Interactive HTML visualizations

View visualizations at: <http://localhost:8080>

### Local Installation

For development or customization, you can install the framework locally:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/leo-routing-simu.git
   cd leo-routing-simu
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Using virtualenv
   virtualenv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Or using venv
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python -m pytest tests/
   ```

## Quick Start

Get started with a simple simulation in minutes using the convenience script:

```bash
# 1. Run a basic simulation with a simplified constellation (uses Docker)
./run-simulations.sh run -c src/config/ether_simple.yaml

# 2. Check the generated output
ls -l output/logs/
ls -l output/simulation/

# 3. Generate and view visualization (starts HTTP server on port 8080)
./run-simulations.sh visualise -c src/config/ether_simple.yaml

# 4. Open your browser to http://localhost:8080
# The visualization will be available at the web interface

# 5. Clean up when done
./run-simulations.sh clean
```

## Usage

### Running Simulations

The main simulation entry point accepts a configuration file that defines all simulation parameters:

```bash
python -m src.main --config <path-to-config.yaml>
```

**Available example configurations**:
- `src/config/ether_simple.yaml` - Simplified constellation (18 orbits, 18 satellites per orbit) for quick testing
- `src/config/starlink.yaml` - Full-scale Starlink-like constellation (22 orbits, 72 satellites per orbit)

**Simulation outputs** are stored in timestamped directories:

```
output/
├── logs/                      # Simulation execution logs
│   └── simulation.log
├── simulation/                # Forwarding state data per time step
├── tles/                      # Generated TLE files for satellite orbits
└── visualizations/            # HTML visualization files
```

### Configuration

The simulation is configured through YAML files that define constellation parameters, simulation settings, ground stations, and network characteristics.

**Example configuration** (`src/config/ether_simple.yaml`):

```yaml
constellation:
  name: "Starlink-550"
  num_orbits: 18
  num_sats_per_orbit: 18
  phase_diff: true
  inclination_degree: 60
  eccentricity: 0.0000001
  arg_of_perigee_degree: 0.0
  mean_motion_rev_per_day: 15.19
  tle_output_filename: "ether_simple_tles.txt"

simulation:
  dynamic_state_algorithm: shortest_path_link_state  # or topological_routing
  end_time_hours: 24
  time_step_minutes: 10
  offset_ns: 0

satellite:
  altitude_m: 600000
  cone_angle_degrees: 29.0

ground_stations:
  - name: "London"
    latitude: 51.5074
    longitude: -0.1278
    elevation_m: 30.0
  - name: "Perth"
    latitude: -31.9505
    longitude: 115.8605
    elevation_m: 30.0

network:
  gsl_interfaces:
    number_of_interfaces: 1
    aggregate_max_bandwidth: 1.0

logging:
  is_debug: false
  file_name: "simulation.log"
```

**Key configuration parameters**:

| Parameter                 | Description                     | Example Values                                    |
| ------------------------- | ------------------------------- | ------------------------------------------------- |
| `num_orbits`              | Number of orbital planes        | 18, 22, 72                                        |
| `num_sats_per_orbit`      | Satellites per orbital plane    | 15, 18, 72                                        |
| `inclination_degree`      | Orbital inclination angle       | 40, 53, 60                                        |
| `altitude_m`              | Satellite altitude in meters    | 550000-1200000                                    |
| `dynamic_state_algorithm` | Routing algorithm to use        | `shortest_path_link_state`, `topological_routing` |
| `end_time_hours`          | Simulation duration in hours    | 1, 24, 96                                         |
| `time_step_minutes`       | Time between topology snapshots | 5, 10, 30                                         |
| `cone_angle_degrees`      | Satellite antenna beam width    | 27-40                                             |

### Visualization

After running a simulation, generate interactive 3D visualizations using Cesium.

**Note**: To use the visualization, you need a Cesium Ion Access Token.
1. Sign up for a free account at [Cesium Ion](https://ion.cesium.com/).
2. Get your default access token.
3. Open `src/satellite_visualisation/static_html/top.html` and replace `YOUR_CESIUM_ION_TOKEN` with your actual token.

```bash
# Generate visualization from simulation config
python -m src.satellite_visualisation.cesium_builder.main src/config/ether_simple.yaml

# Start a local web server
python -m http.server 8000

# Open in browser
# Navigate to: http://localhost:8000/src/satellite_visualisation/visualisation_output/
```

The visualization includes satellite orbits and positions over time

## Project Structure

```
leo-routing-simu/
├── src/
│   ├── main.py                          # Main simulation entry point
│   ├── logger.py                        # Logging configuration
│   ├── config/                          # Configuration files
│   ├── network_state/                   # Network state generation
│   │   ├── generate_network_state.py   # Dynamic state computation
│   │   ├── gsl_attachment/             # Ground station link strategies
│   │   └── routing_algorithms/         # Routing algorithm implementations
│   │       ├── shortest_path_link_state_routing/
│   │       └── topological_routing/
│   ├── satellite_visualisation/         # Visualization tools
│   │   ├── visualise_constellation.py
│   │   └── cesium_builder/             # Cesium-based 3D visualization
│   ├── tles/                            # TLE generation and parsing
│   │   ├── generate_tles_from_scratch.py
│   │   └── read_tles.py
│   └── topology/                        # Topology modeling
│       ├── constellation.py            # Constellation data structures
│       ├── topology.py                 # Network topology management
│       ├── isl.py                      # Inter-Satellite Link modeling
│       ├── ground_station.py           # Ground station management
│       └── satellite/                  # Satellite models
├── tests/                               # Comprehensive test suite
├── output/                              # Simulation outputs (generated)
│   ├── logs/
│   ├── simulation/
│   ├── tles/
│   └── visualizations/
```

## Routing Algorithms

The framework supports multiple routing algorithms through a pluggable architecture:

### 1. Shortest Path Link-State Routing

**Description**: Traditional Dijkstra-based shortest path routing where each node maintains a complete view of the network topology and computes shortest paths to all destinations. This is basically what Hypatia implements.

**Configuration**: `dynamic_state_algorithm: shortest_path_link_state`

### 2. Topological Routing

**Description**: Novel routing algorithm based on the [6G-RUPA](https://6grupa.com) architecture that exploits the regular topology of LEO constellations. Uses hierarchical topological addressing to reduce routing table sizes and control overhead.

**Configuration**: `dynamic_state_algorithm: topological_routing`

### Adding Custom Routing Algorithms

To implement a new routing algorithm:

1. Create a new class inheriting from `RoutingAlgorithm` in `src/network_state/routing_algorithms/`
2. Implement the required abstract methods:
   - `compute_next_hop()` - Calculate next hop for given source-destination pair
   - `update_topology()` - Update internal state based on topology changes
3. Register the algorithm in `routing_algorithm_factory.py`
4. Update configuration files to use your new algorithm

Example:

```python
from src.network_state.routing_algorithms.routing_algorithm import RoutingAlgorithm

class MyCustomRoutingAlgorithm(RoutingAlgorithm):
    def compute_next_hop(self, src, dst, topology):
        # Your routing logic here
        pass
    
    def update_topology(self, topology):
        # Update internal state
        pass
```


## Output Format

The simulation generates forwarding state files that describe the routing decisions at each time step.

**Forwarding State Format**:

Each line in the forwarding state file represents a routing entry:

```
[src-node],[dst-node],[next-hop],[src-interface-id],[next-hop-interface-id]
```

**Fields**:
- `src-node`: Source node ID (satellite or ground station)
- `dst-node`: Destination node ID
- `next-hop`: Next hop node ID on the path to destination
- `src-interface-id`: Interface ID on the source node to send the packet
- `next-hop-interface-id`: Interface ID on the next-hop node that will receive the packet

**Example**:

```
301,992,340,3,5
```

This translates to: A packet at node 301 destined for node 992 will be forwarded to node 340. Node 301 will enqueue the packet on its interface 3, which connects to interface 5 of node 340.

**Output Files**:
- `output/simulation/fstate_<timestamp>.txt` - Forwarding state at specific simulation time
- `output/logs/simulation.log` - Detailed simulation execution log
- `output/tles/<constellation>_tles.txt` - Generated TLE data for satellites

## Testing

The project includes a comprehensive test suite covering all major components:

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/topology/          # Topology generation tests
pytest tests/network_state/     # Network state computation tests
pytest tests/forwarding_state/  # Routing algorithm tests
pytest tests/integration/       # End-to-end integration tests

# Run with coverage report
pytest --cov=src --cov-report=html tests/

# Run specific test file
pytest tests/topology/test_gsl_attachment_integration.py -v
```

**Test Categories**:
- **Topology Tests**: Validate constellation generation, ISL establishment, GSL attachment
- **Network State Tests**: Verify dynamic state computation and routing algorithm correctness
- **Forwarding State Tests**: Test routing decisions and path computation
- **Integration Tests**: End-to-end simulation workflows

## Contributing

### Reporting Issues

- Use the GitHub issue tracker to report bugs or suggest features
- Provide detailed information including:
  - Steps to reproduce the issue
  - Expected vs actual behavior
  - Configuration files used
  - Relevant log outputs

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with clear commit messages
4. Add tests for new functionality
5. Ensure all tests pass (`pytest tests/`)
6. Format code with Black (`black src/ tests/`)
7. Run linters (`flake8 src/ tests/`)
8. Submit a pull request

**Note**: By contributing to this project, you agree that your contributions will be licensed under the AGPL-3.0 license.

### Code Style

- Follow PEP 8 style guidelines
- Use Black for code formatting (line length: 100)
- Use isort for import sorting
- Write docstrings for public functions and classes
- Add type hints where appropriate

### Adding New Features

**New Routing Algorithms**:
1. Create implementation in `src/network_state/routing_algorithms/`
2. Inherit from `RoutingAlgorithm` base class
3. Register in `routing_algorithm_factory.py`
4. Add tests in `tests/forwarding_state/`
5. Update documentation

**New GSL Attachment Strategies**:
1. Implement strategy in `src/network_state/gsl_attachment/`
2. Register with decorator pattern
3. Add integration tests
4. Document in configuration guide

## Copyright

This code has been developed by **Fundació Privada Internet i Innovació Digital a Catalunya (i2CAT)**.

i2CAT is a non-profit research and innovation centre that promotes mission-driven knowledge to solve business challenges, co-create solutions with a transformative impact, empower citizens through open and participative digital social innovation with territorial capillarity, and promote pioneering and strategic initiatives. i2CAT aims to transfer research project results to private companies in order to create social and economic impact via the out-licensing of intellectual property and the creation of spin-offs.

Find more information about i2CAT projects and IP rights at <https://i2cat.net/tech-transfer/>

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)** - see the [LICENSE](LICENSE) file for details.

This code is licensed under the terms of the AGPL. Information about the license can be located at <https://www.gnu.org/licenses/agpl-3.0.html>.

This strong copyleft license was chosen to ensure that improvements to satellite network routing research remain accessible to the broader research community.

If you find that this license doesn't fit with your requirements regarding the use, distribution or redistribution of our code for your specific work, please, don’t hesitate to contact the intellectual property managers in i2CAT at the following address: techtransfer@i2cat.net Also, in the following page you’ll find more information about the current commercialization status or other licensees: Under Development.

## Acknowledgements

This simulator is heavily inspired by [**Hypatia**](https://github.com/snkas/hypatia). Hypatia provides a comprehensive framework for simulating LEO satellite networks, and many of the core concepts, architectural patterns, and simulation methodologies used in this project are adapted from Hypatia's design.

**Key Technologies**:
- [SGP4](https://pypi.org/project/sgp4/) - Satellite orbit propagation
- [Ephem](https://rhodesmill.org/pyephem/) - Astronomical ephemeris calculations
- [Cesium](https://cesium.com/) - 3D geospatial visualization
- [NetworkX](https://networkx.org/) - Graph algorithms and network analysis
- [Hypatia](https://github.com/snkas/hypatia) - LEO satellite network simulator (inspiration and foundation)

## Attributions

Attributions of Third Party Components of this work:

- **SGP4** Version 2.24 -  <https://pypi.org/project/sgp4/> - MIT license
- **ephem** Version 4.2 -  <https://rhodesmill.org/pyephem/> - MIT license
- **NetworkX** Version 3.4.2 -  <https://networkx.org/> - BSD 3-Clause License
- **NumPy** Version 2.2.4 -  <https://numpy.org/> - BSD 3-Clause License
- **Matplotlib** Version 3.10.1 -  <https://matplotlib.org/> - PSF License (Python Software Foundation)
- **PyYAML** Version 6.0.2 -  <https://pyyaml.org/> - MIT license
- **python-dateutil** Version 2.9.0 -  <https://pypi.org/project/python-dateutil/> - dual license - either Apache 2.0 License or the BSD 3-Clause License
- **geopy** Version 2.4.1 -  <https://pypi.org/project/geopy/> - MIT license
- **pandas** Version 2.2.3 -  <https://pandas.pydata.org/> - BSD 3-Clause License
- **czml3** Version 2.3.4 -  <https://pypi.org/project/czml3/> - MIT license
- **exputil** (from Hypatia) -  <https://github.com/snkas/exputilpy> - MIT license
- **astropy** Version 7.0.1 -  <https://www.astropy.org/> - BSD 3-Clause License
- **pytest** Version 8.3.5 -  <https://pytest.org/> - MIT license

## Contact

- **Project Maintainer**: Sergio Giménez
- **Email**: sergio.gimenez@i2cat.net

