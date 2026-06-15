# Configuration

LEOPath uses YAML configuration files under `leopath/config/`.

## Core settings

- `constellation`: orbital layout and TLE generation
- `simulation`: time horizon, time step, and routing algorithm
- `satellite`: altitude and antenna cone
- `ground_stations`: list of ground stations
- `network`: interface counts and bandwidth
- `logging`: log output settings

## Example

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
  dynamic_state_algorithm: shortest_path_link_state
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

network:
  gsl_interfaces:
    number_of_interfaces: 1
    aggregate_max_bandwidth: 1.0

logging:
  is_debug: false
  file_name: "simulation.log"
```

## Example configs

- `leopath/config/ether_simple.yaml`
- `leopath/config/starlink.yaml`
- `leopath/config/kuiper.yaml`
- `leopath/config/oneweb.yaml`
- `leopath/config/telesat.yaml`
- `leopath/config/dense_synthetic.yaml`
