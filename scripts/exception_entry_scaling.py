"""Exception entries vs constellation size N and number of failures F.

Torus grids P x S with rung length falling toward high latitude, F random dead
ISLs, D fixed destination satellites; entries grown from every live satellite.
"""
import logging, math, random, sys, time
logging.disable(logging.CRITICAL)
import networkx as nx
from leopath.network_state.routing_algorithms.topological_routing.fstate_calculation import (
    _build_torus_weight_model, _grow_fixed_address_exceptions)
from leopath.topology.constellation import ConstellationData
from leopath.topology.satellite.topological_network_address import TopologicalNetworkAddress as A

def grid(P, S):
    g = nx.Graph()
    for p in range(P):
        for s in range(S):
            sat = p*S+s
            lat = math.radians(53) * math.sin(2*math.pi*s/S)        # argument of latitude -> latitude
            g.add_edge(sat, p*S+(s+1) % S, weight=40000/S*1000)    # rail
            g.add_edge(sat, ((p+1) % P)*S+s, weight=40000/P*1000*math.cos(lat))  # rung
    return g

def run(P, S, F, D, seed):
    rng = random.Random(seed)
    nominal = grid(P, S); live = nominal.copy()
    while True:
        dead = rng.sample(sorted(nominal.edges()), F)
        live = nominal.copy(); live.remove_edges_from(dead)
        if nx.is_connected(live): break
    addr = {n: A(0, n//S, n % S, 0) for n in nominal}
    cd = ConstellationData(P, S, "20001.0", 1e6, 5e6, [])
    model = _build_torus_weight_model(nominal, addr, cd)
    nb = {n: [(m, m, addr[m], float(live.edges[n, m]["weight"])) for m in live.neighbors(n)] for n in live}
    dests = rng.sample(sorted(live.nodes()), D)
    ex = {}
    for d in dests:
        assert _grow_fixed_address_exceptions(d, addr[d], (addr, nb, cd, "torus_weighted_pivot", model, "progress"), live, ex, {}) == 0
    damaged = {s for e in dead for s in e}
    hops = nx.multi_source_dijkstra_path_length(live, damaged, weight=lambda *_: 1)
    per_sat = {}
    for s, _ in ex: per_sat[s] = per_sat.get(s, 0) + 1
    return len(ex), max(hops[s] for s, _ in ex) if ex else 0, max(per_sat.values()) if per_sat else 0, len(per_sat)

D = 48
print(f"{'part':5} {'grid':>6} {'N':>5} {'F':>3} {'entries':>8} {'/F':>5} {'/(F*D)':>7} {'max/sat':>7} {'sats':>5} {'max hops from failure':>5}")
for part, cases in (("N", [(12,12,10),(24,24,10),(36,36,10),(48,48,10),(72,72,10)]),
                    ("F", [(36,36,f) for f in (1,2,5,10,20,40)])):
    for P, S, F in cases:
        r = [run(P, S, F, D, seed) for seed in range(5)]
        e = sum(x[0] for x in r)/5
        print(f"{part:5} {P}x{S:<3} {P*S:5} {F:3} {e:8.1f} {e/F:5.2f} {e/F/D:7.3f} {max(x[2] for x in r):7} {sum(x[3] for x in r)/5:5.1f} {max(x[1] for x in r):5}", flush=True)
