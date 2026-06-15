# Quickstart

## Requirements

- Python 3.8+
- Docker (optional but recommended)

## Run a quick simulation (Docker)

```bash
./run-simulations.sh run -c leopath/config/ether_simple.yaml
```

## Generate a visualization

```bash
./run-simulations.sh visualise -c leopath/config/ether_simple.yaml
```

Open `http://localhost:8080` to view the visualization.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m leopath.main --config leopath/config/ether_simple.yaml
```

## Serve docs locally

```bash
pip install zensical
zensical serve
```
