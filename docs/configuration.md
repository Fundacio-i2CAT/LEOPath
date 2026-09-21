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
  num_orbits: 72
  num_sats_per_orbit: 22
  phase_diff: true
  inclination_degree: 53
  eccentricity: 0.0000001
  arg_of_perigee_degree: 0.0
  mean_motion_rev_per_day: 15.05
  tle_output_filename: "tles_starlink_550_sgp.txt"

simulation:
  dynamic_state_algorithm: shortest_path_link_state
  end_time_hours: 24
  time_step_minutes: 10
  offset_ns: 0

satellite:
  altitude_m: 550000
  cone_angle_degrees: 25.0

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

## Constellation layout

A shell is a Walker constellation, and three things about it are easy to get wrong without any error message.

**Planes versus satellites per plane.** `num_orbits` counts orbital planes; `num_sats_per_orbit` counts the satellites in each one. Swap them and the logical graph stays the same torus, so nothing complains, but the physics flips:

```
   72 planes x 22 sats (Starlink, FCC 21-48)     22 planes x 72 sats (swapped)

   planes  5.0 deg apart                         planes 16.4 deg apart
   in-plane links ~1 960 km                      in-plane links ~600 km
   cross-plane links 619 - 1 426 km              cross-plane links 881 - 2 153 km
```

Since the pivot estimator exists to exploit ISL geometry, a swapped shell measures a different network.

**Mean motion and altitude have to agree.** SGP4 flies `mean_motion_rev_per_day`; `satellite.altitude_m` never reaches the propagator. It sets the ground-link range and the ISL length limit instead, so when the two disagree, visibility gets computed for an orbit the satellites aren't on. The shipped configs pair them like this, measured by propagating each shell:

| config | planes x sats | mean motion | SGP4 flies | layout from |
|---|---|---|---|---|
| `starlink.yaml` | 72 x 22 | 15.05 | 551 km | FCC 21-48 |
| `kuiper.yaml` | 34 x 34 | 14.80 | 629 km | FCC 20-102 |
| `oneweb.yaml` | 12 x 49 | 13.16 | 1 202 km | FCC DA 23-362 |
| `telesat.yaml` | 27 x 13 | 13.66 | 1 015 km | Hypatia |

Hypatia's Starlink value, 15.19, flies about 508 km despite its "~550 km" label; live Starlink satellites between 550 and 559 km show a median of 15.03.

**Delta or star.** `raan_spread_degree` (default 360) sets the arc the ascending nodes cover:

```
   WALKER DELTA, 360 deg                     WALKER STAR, 180 deg
   Starlink, Kuiper, Telesat                 OneWeb

   last plane --> wrap --> plane 0           last plane ==><== plane 0
   same direction                            opposite directions
   wrap link: ordinary length                wrap link: up to 15 000 km
```

Near-polar shells fly as stars, because at about 88 degrees a node at RAAN and one at RAAN + 180 trace nearly the same ground track in opposite directions. The wrap between the last and first plane then joins satellites up to half an orbit apart, so the `grid` scenario builds the cylinder on a star shell, exactly as `grid_seam` does. On a delta shell the wrap is an ordinary co-rotating link, and `grid_seam` becomes a stress test for losing it. Run metadata records which one happened as `isl_seam_wrap`.

## Example configs

- `leopath/config/ether_simple.yaml`
- `leopath/config/starlink.yaml`
- `leopath/config/kuiper.yaml`
- `leopath/config/oneweb.yaml`
- `leopath/config/telesat.yaml`
- `leopath/config/dense_synthetic.yaml`
