# LEO Routing Simulation Framework

![lrsim_logo.jpeg](lrsim_logo.png)

A simulation framework for analyzing routing algorithms in Low Earth Orbit (LEO) satellite constellations.

## Overview

This project provides tools for:

* Simulating LEO satellite constellations (e.g., Starlink-like networks)
* Analyzing satellite network routing algorithms
* Visualizing satellite orbits, connectivity, and ground stations
* Currently it only supports a link state routing algorithm and a topological routing algorithm based on a novel [6G-RUPA](https://6grupa.com) architecture.

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/leo-routing-simu.git
cd leo-routing-simu
```

Create and activate a virtual environment:

```bash
virtualenv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running Simulations

Run the main simulation with one of the provided configurations:

```bash
python -m src.main --config src/config/ether.yaml
```

The result of the simulation will be stored in a log file under the root of the project. Note that the data structure that represents the forwarding state in every time step is as follows:

```source
[src-node],[dst-node],[next-hop],[src-interface-id],[next-hop-interface-id]
[src-node],[dst-node],[next-hop],[src-interface-id],[next-hop-interface-id]
...
[src-node],[dst-node],[next-hop],[src-interface-id],[next-hop-interface-id]
```

Example:

```source
301,992,340,3,5
```

Translates to: a packet at node 301 destined for 992 will be sent to 340. Node 301 will enqueue it in interface 3 and will send it to the interface 5 of node 340.

You can modify existing configuration files or create your own in the config directory.

### Visualization

After running a simulation, TLE data is generated for the satellites, which can be visualized using Cesium:

```bash
python -m src.satellite_visualisation.cesium_builder.main src/config/ether.yaml
```

This generates an HTML file in the visualisation_output directory. To view the visualization:

```bash
python -m http.server
```

Then open your browser to <http://localhost:8000/visualisation_output/>

### Configuration

The simulation is configured through YAML files:

* `ether.yaml` - Full Starlink-like constellation (22 orbits, 72 satellites per orbit)
* `ether_simple.yaml` - Simplified constellation (15 orbits, 15 satellites per orbit)

Key configuration parameters:

```yaml
constellation:
  name: "Starlink-550"
  num_orbits: 15
  num_sats_per_orbit: 15
  inclination_degree: 40
  mean_motion_rev_per_day: 15.19
  
simulation:
  dynamic_state_algorithm: shortest_path_link_state
  end_time_hours: 24
  time_step_minutes: 5
  
satellite:
  altitude_m: 600000
  cone_angle_degrees: 27.0
  
ground_stations:
  - name: "London"
    latitude: 51.5074
    longitude: -0.1278
```

## Project Structure

* `main.py` - Main simulation entry point
* `config` - Configuration files
* `satellite_visualisation` - Visualization components
* `requirements.txt` - Project dependencies

## License

TODO

## Acknowledgements

* This project is part of ongoing PhD research in satellite network routing optimization.
TODO Ether?