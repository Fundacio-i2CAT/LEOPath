# ISL Topology

LEOPath builds inter-satellite links as a `+Grid`: four laser terminals per satellite, two along the orbit and two across to the neighbouring planes. Every published result uses it, and so does most of the literature LEOPath compares against.

Starlink's own technology page says each satellite carries three space lasers, not four. That single missing terminal changes the shape of the network enough to break the pivot distance estimator outright, and this page works out why, what the repair looks like, and what it would cost to build.

**Nothing on this page is implemented.** There is no `brick` ISL topology and no row-pair estimator in the simulator. Every number below comes from all-pairs BFS on the logical torus, and they exist to decide whether the work is worth doing.

## Four terminals, and what they buy

A link burns one terminal at each end. With four, a satellite reaches its orbital neighbours ahead and behind, plus one satellite in each adjacent plane:

```
          fore  (next satellite in the same orbit, fixed geometry)
            |
  port -----o----- starboard     +Grid: 4 terminals
            |      (one satellite in each neighbouring plane)
           aft
```

Lay the shell out with planes as columns and satellite index as rows, wrapping both ways into a torus:

```
         p0    p1    p2    p3    p4
   s0     o-----o-----o-----o-----o
          |     |     |     |     |
   s1     o-----o-----o-----o-----o
          |     |     |     |     |
   s2     o-----o-----o-----o-----o
          |     |     |     |     |
   s3     o-----o-----o-----o-----o

   |  intra-plane (fore/aft)      -  cross-plane (port/starboard)
```

Distance is Manhattan on the torus, `|dplane| + |drow|`, and the whole topological forwarding argument leans on that.

## Three terminals force a matching

Keep fore and aft, which cost nothing because two satellites in the same orbit never move relative to each other, and one cross-plane terminal is left:

```
            |
        ?---o        one of port / starboard, not both
            |
```

One cross-plane terminal per satellite means the cross-plane links form a perfect matching over the whole shell. Within any plane, some satellites point left and the rest point right. That isn't a design decision anyone gets to make; it follows from counting terminals.

Pairing whole planes fails immediately:

```
         p0    p1        p2    p3
   s0     o-----o         o-----o
          |     |         |     |
   s1     o-----o         o-----o
          |     |         |     |
   s2     o-----o         o-----o
         \_____/         \_____/
          island          island

   every satellite in p1 spent its cross terminal on p0,
   so nothing is left to reach p2
```

Two diagonal variants also fail. Running the rung from `(p,s)` to `(p+1,s+1)` and staggering on plane parity splits the shell into the same isolated plane pairs; staggering the same diagonal on `(p+s)` parity hands half the satellites two cross links and the other half none, so it isn't a three-laser layout at all.

What survives is one rule: a rung joins `(p,s)` and `(p+1,s)` when `(p + s)` is even.

```
         p0    p1    p2    p3    p4
   s0     o-----o     o-----o     o
          |     |     |     |     |
   s1     o     o-----o     o-----o
          |     |     |     |     |
   s2     o-----o     o-----o     o
          |     |     |     |     |
   s3     o     o-----o     o-----o
```

Brick courses, which is where the name comes from; graph theory calls it the hexagonal lattice. Every satellite has degree 3, the shell stays connected, and no satellite is special. Given that the intra-plane ring stays, it's the only uniform option.

The rule needs an even number of planes and an even number of satellites per plane, or the parity fails to close on the wrap. Telesat's 27 by 13 grid fails both, and patching it with a seam leaves a column of degree-2 and degree-4 satellites. Starlink (72 by 22), Kuiper (34 by 34) and OneWeb (36 by 18) all close cleanly.

## The rungs never line up

Send a packet from `(p0,s0)` to `(p4,s0)`: four planes right, same row.

On `+Grid` that's four hops in a straight line:

```
         p0    p1    p2    p3    p4
   s0     S====>o====>o====>o====>D
```

On the brick wall it costs eight:

```
         p0    p1    p2    p3    p4
   s0     (1)===(2)   (5)===(6)   (9)
           |     |     |     |     |
   s1      o    (3)===(4)   (7)===(8)
```

Cross, step down, cross, step back up, cross, step down, cross, step back up. Half those hops net out to zero row movement and buy nothing; the packet pays them because crossing a plane flips the parity that decides where the next rung sits. Crossing one plane costs two hops rather than one, unless the forced row shift happens to be movement the packet wanted anyway.

That "unless" is why the penalty depends on the shape of the grid and not on its size:

```
  Starlink, 72 planes x 22 rows, worst case crosses 36 planes
     +Grid:  36 rungs                              =  36 hops
     brick:  36 rungs + 35 forced row hops         =  71 hops
             the row hops go +1, -1, +1, -1, netting zero;
             22 rows cannot absorb 35 of them

  Kuiper, 34 x 34, worst case crosses 17 planes
     a typical trip already needs about 17 row hops,
     so the forced shifts disappear into movement the packet wanted
```

All-pairs BFS on the logical torus, unit hop costs:

| shell | grid | ISLs, +Grid to brick | mean path, +Grid | mean path, brick | penalty | diameter |
|---|---|---|---|---|---|---|
| Kuiper | 34 x 34 | 2312 to 1734 | 17.01 | 19.84 | +17% | 34 to 34 |
| Telesat* | 27 x 13 | 702 to 527 | 10.00 | 13.90 | +39% | 19 to 27 |
| OneWeb | 36 x 18 | 1296 to 972 | 13.52 | 18.77 | +39% | 27 to 36 |
| Starlink | 72 x 22 | 3168 to 2376 | 23.51 | 36.58 | **+56%** | 47 to **72** |

\* Telesat's grid is odd in both dimensions, so its brick row uses a seam and its satellites aren't all degree 3. Treat that row as indicative.

Starlink's diameter lands on exactly `2 x 36`: every crossing paid double, nothing absorbed. Cutting a quarter of the links costs more than a quarter of the path efficiency, and how much more depends entirely on aspect ratio.

## What breaks in the estimator

`_torus_weighted_pivot_distance` (`fstate_calculation.py:1322`) picks a single pivot row, walks the source plane to it, crosses every plane along that one row, then walks down the destination plane to the target:

```
         p0    p1    p2    p3    p4
   s0     S     o     o     o     o
          |
   s1     o     o     o     o     o
          |
   s2     x-----o-----o-----o-----y   <- pivot row: cross everything here
                                  |
   s3     o     o     o     o     D
```

On a brick wall no row ever has two consecutive rungs. Row `s2` carries a rung only where `(p+2)` is even:

```
         p0    p1    p2    p3    p4
   s2     x=====o  X  o=====o  X  y
                gap         gap
```

`plane_edge_costs` starts every entry at infinity and records only rungs that exist (`fstate_calculation.py:1244`), and `_sum_torus_edges` returns infinity the moment it walks into one gap. Both directions round the torus hit gaps. So for any `|dplane|` of 2 or more, every pivot row returns infinity and the estimator reports every satellite beyond the adjacent plane as unreachable. The model doesn't degrade on a brick wall; it stops producing a number.

An idealised estimator that fails gracefully instead, plain Manhattan with no knowledge of which rungs exist, still strands a couple of percent of traffic with no failures injected anywhere:

```
         p0    p1    p2    p3    p4
   s0     o-----N     o     o     D    N needs D, three planes right
          <     |                      Manhattan(N, D) = 3
   s1           o
                                       N's only rung points left    -> 4
                                       both row neighbours          -> 4
                                       no neighbour is closer, so N is stuck
```

## The repair: pivot row pairs

Drop the assumption that one row can carry the whole crossing. Use two adjacent rows, and the staircase is always available:

```
         p0    p1    p2    p3    p4
   r      x=====o     o=====o     y
           |     |     |     |
   r+1     o     o=====o     o=====o
```

Every rung on that staircase exists by construction, for any starting row. Crossing one plane costs a rung plus a vertical hop, and the vertical hops come out of `row_edge_costs`, which the weight model already builds. The pivot loop at `fstate_calculation.py:1352` iterates row pairs instead of rows; table sizes and query cost don't change.

In closed form, crossing `a` planes with a row ring-distance of `rd`:

```
L = max(rd, (a - 1) + [first crossing needs a shift])
if (L - rd) is odd:
    L += 1
distance = a + L        # take the cheaper way round the torus
```

Checked against BFS over 4 269 904 ordered pairs across OneWeb, Kuiper, Starlink and two small tori, with zero mismatches. It evaluates in constant time, same as Manhattan.

| shell | current pivot model | Manhattan on brick | row-pair closed form |
|---|---|---|---|
| 28 x 14 (even stand-in for Telesat) | unreachable | 3.32% stuck | **0.0000%** |
| OneWeb 36 x 18 | unreachable | 2.63% stuck | **0.0000%** |
| Kuiper 34 x 34 | unreachable | 1.39% stuck | **0.0000%** |
| Starlink 72 x 22 | unreachable | 2.21% stuck | **0.0000%** |

Greedy forwarding on an exact hop metric always finds a strictly closer neighbour, so the local minima go away entirely rather than getting rarer.

## Why it's worth building

Whether a rung exists is `(plane_id + sat_index) mod 2`, and a satellite already carries both fields in its topological address. No table, no flooding, no extra per-satellite state: one bit of arithmetic over something the satellite knows about itself.

So a three-laser shell isn't a case where structured addressing gives up and falls back to topology state. The grid stops being uniform, forwarding still comes out of the address, and state stays flat. Degree-4 `+Grid` was the easy instance of that claim; this is the harder one.

Keep the two costs apart when reporting. The 17% to 56% path penalty is what the missing third laser costs *anyone* routing on that graph, and link-state pays it in full on the same topology. LEOPath's stretch metric compares topological forwarding against shortest path on the brick graph itself, and with the row-pair model it should sit near 1.0. Reporting the two together would be wrong.

## Caveats

The closed form is exact for unit hops. Under measured kilometre weights it becomes an approximation, the same way the single-row model is already an approximation today, and `forwarding_guard: progress` covers whatever residual error is left. Parity has to close, which rules out Telesat unless someone writes a seam variant. SpaceX publishes the laser count but not the wiring, so the brick wall is an argued assumption rather than a disclosure, and any paper text should carry the counting argument alongside it.

## Effort

Roughly a day and a half, plus a couple of hours of runs:

- an `--isl-topology brick` flag in the link builder, dropping rungs where `(p + s)` is odd, which is an hour or two
- `_build_torus_weight_model` computing plane costs per pivot row pair rather than per row, which is the bulk of it
- the pivot loop in `_torus_weighted_pivot_distance` iterating pairs, roughly an hour
- tests against a BFS reference, which already exists and already verified clean
- a sweep variant covering Starlink, Kuiper and OneWeb

## Reproducing the numbers

```python
from collections import deque


def build(planes, sats, brick):
    adjacency = {(p, s): [] for p in range(planes) for s in range(sats)}
    edges = set()
    for p in range(planes):
        for s in range(sats):
            edges.add(((p, s), (p, (s + 1) % sats)))
            if not brick or (p + s) % 2 == 0:
                edges.add(((p, s), ((p + 1) % planes, s)))
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    return adjacency


def ring(a, b, n):
    d = abs(a - b) % n
    return min(d, n - d)


def brick_distance(p0, s0, p1, s1, planes, sats):
    """Closed-form hop distance on a brick wall. Needs planes and sats both even."""
    best = None
    for crossings, needs_shift in (
        ((p1 - p0) % planes, (p0 + s0) % 2 == 1),
        ((p0 - p1) % planes, (p0 + s0) % 2 == 0),
    ):
        rd = ring(s0, s1, sats)
        if crossings == 0:
            candidate = rd
        else:
            span = max(rd, (crossings - 1) + (1 if needs_shift else 0))
            if (span - rd) % 2:
                span += 1
            candidate = crossings + span
        best = candidate if best is None else min(best, candidate)
    return best
```

Run BFS from every satellite over `build(72, 22, brick=True)` and compare each distance against `brick_distance`; the two agree on every pair.
