"""Exception entries for EVERY destination (a), and aggregated by address prefix (c).

Torus P x S, rung length falling with latitude, F random dead ISLs. Entries grow
per destination satellite until the guarded rule delivers from every live
satellite; (c) then rewrites each satellite's entries as region entries, blocks of
(plane range x slot range) in the address grid, only where that is exact, so
every forwarding decision is unchanged; whole-plane prefixes are reported too.

Shortcut: on the intact grid the rule delivers from everywhere with no entries,
so a walk can only break after reaching a satellite that lost an ISL; only
walks from those satellites are checked. --verify compares with brute force.
"""
import argparse, json, logging, math, random, sys
logging.disable(logging.CRITICAL)
import networkx as nx
from leopath.network_state.routing_algorithms.topological_routing.fstate_calculation import (
    _build_torus_weight_model, _fixed_rule_step, _walk_fixed_topological_address,
    _breaking_satellite, _next_on_shortest_live_path)
from leopath.topology.constellation import ConstellationData
from leopath.topology.satellite.topological_network_address import TopologicalNetworkAddress as A

MODE, GUARD = "torus_weighted_pivot", "progress"

def grid(P, S):
    g = nx.Graph()
    for p in range(P):
        for s in range(S):
            sat = p*S+s
            lat = math.radians(53)*math.sin(2*math.pi*s/S)
            g.add_edge(sat, p*S+(s+1) % S, weight=40000/S*1000, kind="rail")
            g.add_edge(sat, ((p+1) % P)*S+s, weight=40000/P*1000*math.cos(lat), kind="rung")
    return g

def setup(P, S, F, seed):
    rng = random.Random(seed)
    nominal = grid(P, S)
    while True:
        dead = rng.sample(sorted(nominal.edges()), F)
        live = nominal.copy(); live.remove_edges_from(dead)
        if nx.is_connected(live): break
    addr = {n: A(0, n//S, n % S, 0) for n in nominal}
    cd = ConstellationData(P, S, "20001.0", 1e6, 5e6, [])
    model = _build_torus_weight_model(nominal, addr, cd)
    nb = {n: [(m, m, addr[m], float(live.edges[n, m]["weight"])) for m in live.neighbors(n)] for n in live}
    return nominal, live, dead, addr, cd, model, nb

def grow(dst, sources, ctx, ex):
    live, addr, cd, model, nb = ctx
    dist = nx.single_source_dijkstra_path_length(live, dst, weight="weight")
    steps, delivering = {}, set()
    while True:
        added = False
        for s in sources:
            if s == dst or s in delivering: continue
            path, fail = _walk_fixed_topological_address(s, dst, addr[dst], addr, nb, cd, MODE, model, GUARD,
                                                         exceptions=ex, rule_steps=steps, delivering=delivering)
            if fail is None: continue
            b = _breaking_satellite(path, fail, dst, ex)
            h = _next_on_shortest_live_path(b, live, dist)
            assert h is not None
            ex[(b, dst)] = h; added = True
        if not added: return

def aggregate(ex, ctx, P, S):
    """Exact region aggregation per satellite: blocks of (plane range x slot range).

    A region entry (planes a..b, slots c..d) -> h is allowed only if every
    destination in it already leaves this satellite via h (its own entry or
    the rule), so every forwarding decision is unchanged. Greedy cover: from
    each uncovered entry grow the block along the slot axis, then the plane
    axis, as far as it stays consistent (cyclic indices). An upper bound on
    the minimum. Also returns the count with whole-plane prefixes only.
    """
    live, addr, cd, model, nb = ctx
    by_sat = {}
    for (x, d), h in ex.items(): by_sat.setdefault(x, {})[d] = h
    total, per_sat, prefix_total = 0, {}, 0
    for x, ents in by_sat.items():
        memo = {}
        def f(d):
            if d == x: return "self"
            if d not in memo:
                if d in ents: memo[d] = ents[d]
                else:
                    st = _fixed_rule_step(x, addr[d], addr, nb, cd, MODE, model, GUARD)
                    memo[d] = st[0] if st else None
            return memo[d]
        def cell(p, s): return (p % P)*S + (s % S)
        covered, blocks = set(), 0
        for d in sorted(ents):
            if d in covered: continue
            h = ents[d]; p0, s0 = divmod(d, S)
            lo = hi = 0
            while hi - lo + 1 < S and f(cell(p0, s0+hi+1)) == h: hi += 1
            while hi - lo + 1 < S and f(cell(p0, s0+lo-1)) == h: lo -= 1
            plo = phi = 0
            ok = lambda pp: all(f(cell(pp, s0+k)) == h for k in range(lo, hi+1))
            while phi - plo + 1 < P and ok(p0+phi+1): phi += 1
            while phi - plo + 1 < P and ok(p0+plo-1): plo -= 1
            for pp in range(plo, phi+1):
                for k in range(lo, hi+1):
                    c = cell(p0+pp, s0+k)
                    if c in ents and ents[c] == h: covered.add(c)
            blocks += 1
        per_sat[x] = blocks; total += blocks
        # whole-plane prefix (hierarchical) aggregation, for comparison
        for plane in {d//S for d in ents}:
            members = [plane*S+k for k in range(S) if plane*S+k != x]
            in_plane = sum(1 for d in ents if d//S == plane)
            best = in_plane
            for h in {f(d) for d in members} - {None}:
                best = min(best, 1 + sum(1 for d in members if f(d) != h))
            prefix_total += best
    return total, max(per_sat.values(), default=0), prefix_total

def run(P, S, F, seed, verify=False):
    nominal, live, dead, addr, cd, model, nb = setup(P, S, F, seed)
    ctx = (live, addr, cd, model, nb)
    damaged = sorted({s for e in dead for s in e})
    ex = {}
    for d in sorted(live.nodes()):
        grow(d, damaged, ctx, ex)
    if verify:   # every live source delivers to every destination with these entries
        for d in sorted(live.nodes()):
            for s in live.nodes():
                if s != d:
                    _p, fail = _walk_fixed_topological_address(s, d, addr[d], addr, nb, cd, MODE, model, GUARD, exceptions=ex)
                    assert fail is None, (P, S, F, seed, s, d)
    rails = sum(1 for e in dead if nominal.edges[e]["kind"] == "rail")
    agg, agg_max, prefix = aggregate(ex, ctx, P, S)
    per_sat = {}
    for x, _ in ex: per_sat[x] = per_sat.get(x, 0)+1
    return dict(P=P, S=S, N=P*S, F=F, seed=seed, dead_rails=rails, entries=len(ex),
                max_per_sat=max(per_sat.values(), default=0), sats=len(per_sat),
                prefix_aggregated=prefix, region_aggregated=agg, region_max_per_sat=agg_max)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("P", type=int); ap.add_argument("S", type=int)
    ap.add_argument("F", type=int); ap.add_argument("seed", type=int); ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    print(json.dumps(run(a.P, a.S, a.F, a.seed, a.verify)), flush=True)
