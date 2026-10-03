"""Compare live browser wiring and brick estimates with the Python simulator."""
from pathlib import Path
from types import SimpleNamespace
import json
import math
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import networkx as nx
from leopath.main import generate_brick_isls, generate_plus_grid_isls
from leopath.network_state.routing_algorithms.topological_routing.fstate_calculation import _build_torus_weight_model, _brick_pivot_distance

fixtures = {"graphs": [], "models": []}
configs = json.loads((ROOT / "docs/cesium/constellations.json").read_text())["constellations"]
for c in configs:
    p, s = c["orbits"], c["satsPerOrbit"]
    seam = c.get("raanSpreadDeg", 360) < 360
    for topology in ["grid", "ring", "brick_a", "brick_b"]:
        try:
            if topology.startswith("brick"):
                edges = generate_brick_isls(p, s, split=topology[-1], seam=seam)
            elif topology == "grid":
                edges = generate_plus_grid_isls(p, s, seam=seam)
            else:
                edges = [(min(p0*s+r,p0*s+(r+1)%s),max(p0*s+r,p0*s+(r+1)%s)) for p0 in range(p) for r in range(s)]
            fixtures["graphs"].append(dict(shell=c["id"],planes=p,slots=s,topology=topology,planeWrap=not seam,edges=sorted(edges)))
        except ValueError:
            fixtures["graphs"].append(dict(shell=c["id"],planes=p,slots=s,topology=topology,planeWrap=not seam,invalid=True))

for p,s,seam,topology in [(6,8,False,"brick_a"),(6,8,False,"brick_b"),(6,8,True,"brick_a"),(6,8,True,"brick_b"),(5,8,True,"brick_a"),(6,7,False,"brick_a")]:
    graph = nx.Graph()
    graph.add_weighted_edges_from([(a,b,1.0) for a,b in generate_brick_isls(p,s,split=topology[-1],seam=seam)])
    sats = [SimpleNamespace(shell_id=0,plane_id=i//s,sat_index=i%s) for i in range(p*s)]
    addresses = {i:SimpleNamespace(get_satellite_address=lambda sat=sat:sat) for i,sat in enumerate(sats)}
    model = _build_torus_weight_model(graph,addresses,SimpleNamespace(n_orbits=p,n_sats_per_orbit=s),wiring=topology)
    def clean(value):
        if isinstance(value,list): return [clean(x) for x in value]
        return value if math.isfinite(value) else None
    fixtures["models"].append(dict(orbits=p,satsPerOrbit=s,topology=topology,
        rowPathCosts=clean(model["row_path_costs"]),planePathCosts=clean(model["plane_path_costs"]),
        brick=dict(rail=model["brick"]["rail"],rung=sum(model["brick"]["rung_by_row"])/s,planeWrap=model["brick"]["plane_wrap"]),
        distances=[[_brick_pivot_distance(a,b,model) for b in sats] for a in sats]))
subprocess.run(["node", str(ROOT / "scripts/viewer/check-live-network.cjs")],input=json.dumps(fixtures),text=True,check=True,cwd=ROOT)
