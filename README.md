# LEOPath

<p align="center">
  <img src="leopath_logo.png" alt="LEOPath logo" width="520"/>
</p>

<p align="center">
  <strong>Research-grade LEO satellite routing simulator with dynamic topology, routing-state evaluation, and CesiumJS constellation visualization.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/leopath/"><img src="https://img.shields.io/pypi/v/leopath.svg" alt="PyPI"/></a>
  <a href="https://pypi.org/project/leopath/"><img src="https://img.shields.io/pypi/status/leopath.svg" alt="PyPI status"/></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0-blue.svg" alt="License: AGPL-3.0"/></a>
  <a href="https://fundacio-i2cat.github.io/LEOPath/"><img src="https://img.shields.io/badge/docs-online-brightgreen.svg" alt="Documentation"/></a>
  <a href="https://fundacio-i2cat.github.io/LEOPath/cesium/"><img src="https://img.shields.io/badge/demo-CesiumJS-6f42c1.svg" alt="CesiumJS demo"/></a>
  <a href="https://github.com/Fundacio-i2CAT/LEOPath/stargazers"><img src="https://img.shields.io/github/stars/Fundacio-i2CAT/LEOPath?style=flat" alt="GitHub stars"/></a>
  <a href="https://github.com/Fundacio-i2CAT/LEOPath/issues"><img src="https://img.shields.io/github/issues/Fundacio-i2CAT/LEOPath" alt="GitHub issues"/></a>
  <a href="https://github.com/Fundacio-i2CAT/LEOPath/commits"><img src="https://img.shields.io/github/last-commit/Fundacio-i2CAT/LEOPath" alt="Last commit"/></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"/></a>
</p>

LEOPath helps researchers and network engineers study how routing algorithms behave when the network itself is moving. It generates synthetic LEO constellations, builds time-varying inter-satellite and ground-to-satellite links, computes forwarding state, and exports metrics for scalability studies.

The short version: use LEOPath when you want to compare routing strategies under realistic orbital dynamics without building a full packet-level simulator first.

## Highlights

- Dynamic LEO topology generation from configurable constellation parameters and SGP4/TLE data.
- Inter-satellite links (ISLs), ground-to-satellite links (GSLs), and nearest-satellite attachment models.
- Pluggable routing algorithms for link-state, topological, predictive, and explicit-path approaches.
- Evaluation harness for stretch, churn, forwarding-state size, and compute-time metrics.
- Browser-based CesiumJS visualization for inspecting constellations, ISLs, GSLs, and orbital motion.
- Docker-first workflow for reproducible runs, plus editable local Python installs for development.

## Visualization

One of LEOPath's best features is the interactive constellation viewer:

**Open the live Cesium demo:** https://fundacio-i2cat.github.io/LEOPath/cesium/

<p align="center">
  <a href="https://fundacio-i2cat.github.io/LEOPath/cesium/">
    <img src="docs/assets/cesium-demo.gif" alt="LEOPath Cesium constellation visualization demo" width="720"/>
  </a>
</p>

The viewer runs in the browser and lets you inspect satellite motion, ring vs +grid ISL topologies, ground stations, nearest-visible GSL attachments, and dense constellation samples.

## Quickstart

### Run with Docker

```bash
git clone https://github.com/Fundacio-i2CAT/LEOPath.git
cd LEOPath
./run-simulations.sh run -c leopath/config/ether_simple.yaml
```

### Generate a local visualization

```bash
./run-simulations.sh visualise -c leopath/config/ether_simple.yaml
```

Open `http://localhost:8080`.

### Run from source

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
leopath --config leopath/config/ether_simple.yaml
```

## What You Can Study

| Question | LEOPath output |
| --- | --- |
| How much stretch does a routing strategy introduce? | Hop and distance stretch vs shortest-path baseline |
| How often does forwarding state change? | Per-step churn metrics |
| How much state does each satellite need? | Forwarding-state size metrics |
| How expensive is route recomputation? | Per-step compute-time metrics |
| How does topology shape behavior? | Ring and +grid ISL scenarios |

## Routing Algorithms

LEOPath currently includes:

- `shortest_path_link_state`: Dijkstra baseline over each dynamic topology snapshot.
- `topological_routing`: 6G-RUPA-inspired forwarding using structured satellite addresses.
- `predictive_link_state`: link-state computed on a predicted future topology snapshot.
- `explicit_path_routing`: centrally planned explicit-path routing with pinned satellite paths.

New routing algorithms can be added under `leopath/network_state/routing_algorithms/` and registered in the routing algorithm factory.

## Evaluation Workflow

Run a compact evaluation matrix:

```bash
./run-quick-eval.sh
```

Generate paper-oriented evaluation outputs:

```bash
./run-paper-eval.sh
```

Common outputs include CSV metrics, logs, plots, and metadata describing algorithm parameters and topology scenarios.

## Documentation

- Documentation site: https://fundacio-i2cat.github.io/LEOPath/
- Cesium constellation viewer: https://fundacio-i2cat.github.io/LEOPath/cesium/
- Quickstart: https://fundacio-i2cat.github.io/LEOPath/quickstart/
- Routing algorithms: https://fundacio-i2cat.github.io/LEOPath/algorithms/
- Evaluation guide: https://fundacio-i2cat.github.io/LEOPath/evaluation/

Build and serve docs locally:

```bash
bash scripts/build-docs-site.sh
python -m http.server -d site
```

Then open `http://localhost:8000`.

## Project Scope

LEOPath focuses on topology, routing state, and scalability metrics. It is not a packet-level simulator and does not model TCP/IP stacks, queues, PHY behavior, or detailed link-layer effects.

For packet-level studies, use LEOPath to generate forwarding-state artifacts and connect them to tools such as NS-3.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pytest
flake8
black .
```

## Contributing

Contributions are welcome. Good first areas are new routing algorithms, additional constellation presets, visualization improvements, evaluation metrics, and documentation examples.

Before opening a pull request, run tests and formatters, and include a short note explaining the scenario or metric affected by the change.

## License

LEOPath is released under the [AGPL-3.0 license](LICENSE).

<p align="center">
  <img src="i2cat_logo.png" alt="i2CAT logo" width="150"/>
</p>
