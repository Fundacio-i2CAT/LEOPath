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

Two words used throughout this page, by analogy with a ladder. The verticals are **rails**, running along the orbit; the horizontals are **rungs**, crossing to the neighbouring plane. Neither term is standard, and the two behave nothing alike:

```
                  ahead (same orbit)
                        o
                        |   rail
                        |
     o ---- rung ----- [X] ----- rung ---- o
   plane p-1            |             plane p+1
                        |   rail
                        o
                  behind (same orbit)

   rails  fixed range, fixed pointing, no tracking loop
   rungs  range and pointing change constantly,
          worst near the poles where the planes converge
```

Distance is Manhattan on the torus, `|dplane| + |drow|`, and the whole topological forwarding argument leans on that.

## Counting terminals

Three terminals mean one kind of neighbour gets bought once. Working the arithmetic through shows there are exactly two candidate layouts, not one, and that the choice between them isn't settled by anything SpaceX has published.

A link burns one terminal at each end, so a terminal points at exactly one satellite and nothing else:

```
        [X] ============== [Y]
         ^                  ^
    one terminal       one terminal
```

Buy two terminals of a kind and those links form an unbroken line. Buy one and they can only form pairs:

```
  TWO terminals of a kind              ONE terminal of a kind

   o---o---o---o---o---o                o---o   o---o   o---o
   ^   ^   ^   ^   ^   ^                ^   ^   ^   ^   ^   ^
   spends 2, links both                 spends 1, links ONE
   neighbours                           neighbour

   -> unbroken line                     -> disjoint pairs, a matching
```

The pairs can't all face the same way either:

```
  all pairs aligned                    pairs alternate

  p0   p1    p2   p3                   p0   p1   p2   p3
   o---o     o---o                      o---o    o---o
   |   |     |   |                      |   |    |   |
   o---o     o---o                      o    o---o    o
   |   |     |   |                      |   |    |   |
   o---o     o---o                      o---o    o---o
  \____/    \____/
   island    island                     connected, every plane reachable
                 ^
        no rungs cross here
```

Every satellite in p1 spent its one rung terminal pointing at p0, so the p0/p1 boundary carries a rung at *every* row, far more than it needs, while the p1/p2 boundary gets nothing. The shell falls into disconnected components, 36 two-plane islands on Starlink, and a packet sitting in p0 can never reach p2 by any route. Alternation isn't a preference; it's what stops that happening.

Two diagonal layouts fail for related reasons. Running the rung from `(p,s)` to `(p+1,s+1)` and staggering on plane parity produces the same isolated plane pairs; staggering that diagonal on `(p+s)` parity hands half the satellites two rungs and the other half none, so it isn't a three-terminal layout at all.

## The two splits

```
             rails (along orbit)      rungs (across planes)      degree
          +------------------------+-------------------------+
  +Grid   |  2 terminals -> LINE   |  2 terminals -> LINE    |    4
          +------------------------+-------------------------+
  SPLIT A |  2 terminals -> LINE   |  1 terminal  -> STAGGER |    3
          +------------------------+-------------------------+
  SPLIT B |  1 terminal  -> STAGGER|  2 terminals -> LINE    |    3
          +------------------------+-------------------------+
```

Split A keeps the orbital ring whole and staggers the rung, so a rung joins `(p,s)` and `(p+1,s)` when `(p + s)` is even:

```
SPLIT A   rails complete, rungs staggered

      p0    p1    p2    p3    p4
 s0    o-----o     o-----o     o
       |     |     |     |     |          every vertical present
 s1    o     o-----o     o-----o          horizontals alternate
       |     |     |     |     |
 s2    o-----o     o-----o     o
       |     |     |     |     |
 s3    o     o-----o     o-----o
```

Brick courses, which is where the name comes from; graph theory calls it the hexagonal lattice.

Split B keeps the cross-plane mesh whole and staggers the rail instead, breaking each orbital ring into pairs:

```
SPLIT B   rungs complete, rails staggered

      p0    p1    p2    p3    p4
 s0    o-----o-----o-----o-----o
       |           |           |          every horizontal present
 s1    o-----o-----o-----o-----o          verticals alternate
             |           |
 s2    o-----o-----o-----o-----o
       |           |           |
 s3    o-----o-----o-----o-----o
```

Neither is obviously the real one. Rails are cheap to hold, since two satellites in the same orbit at the same altitude never move relative to each other and the terminal can sit bolted down with no tracking loop, while rungs fight varying range and high angular rates, worst near the poles where planes converge. Cheap to keep lit isn't the same as worth buying twice, though, and the numbers below favour split B heavily on the shell that matters most.

Both layouts carry 1.5 links per satellite, down from 2.

## Which shells the parity fits

Split A needs an even plane count, split B an even number of satellites per plane, or the alternation collides with itself on the wrap:

```
   P even, alternation closes          P odd, alternation collides

   p0  p1  p2  p3  p0                  p0  p1  p2  p0
    o---o   o---o                       o---o   o   o
                   \___ wraps                       \___ wraps
                       back to p0                       back to p0

    matching stays a matching           two rungs land on the same
                                        satellite and another gets none
```

| shell | grid | split A | split B |
|---|---|---|---|
| Starlink | 72 x 22 | yes | yes |
| Kuiper | 34 x 34 | yes | yes |
| OneWeb | 36 x 18 | yes | yes |
| Telesat | 27 x 13 | no, 27 is odd | no, 13 is odd |

Patching Telesat with a seam leaves a column of degree-2 and degree-4 satellites, which is a different topology rather than a brick wall, so Telesat stays out of everything that follows.

## What each split costs a packet

On `+Grid`, crossing four planes along one row is four hops in a straight line:

```
         p0    p1    p2    p3    p4
   s0     S====>o====>o====>o====>D
```

Split A turns that into eight, and split B does the same thing to row movement:

```
SPLIT A: cross 4 planes, same row            SPLIT B: move 4 rows, same plane

      p0    p1    p2    p3    p4                  p0    p1
 s0   (1)===(2)   (5)===(6)   (9)            s0   (1)
       |     |     |     |     |                   |
 s1    o    (3)===(4)   (7)===(8)            s1   (2)===(3)
                                                          |
   rung, rail, rung, rail, ...               s2   (5)===(4)
   4 rungs + 4 rails = 8 hops                      |
                                             s3   (6)===(7)
                                                          |
                                             s4   (9)===(8)

                                               rail, rung, rail, rung, ...
                                               4 rails + 4 rungs = 8 hops
```

In split A, crossing a plane flips the parity that decides where the next rung sits, so the packet has to shift a row before it can cross again, and half its hops net out to zero row movement. Split B is the same picture rotated: moving a row flips the parity that decides where the next rail sits. Each split charges roughly one extra hop per step along the axis it taxes, unless that step was movement the packet wanted anyway.

That exception is why the penalty tracks the shape of the grid rather than its size:

```
   Starlink:  72 planes  x  22 rows            Kuiper:  34 x 34
   +----------------------------------+        +---------------+
   |                                  |        |               |
   |      long axis = PLANES          | 22     |               | 34
   |                                  |        |               |
   +----------------------------------+        +---------------+
                  72                                   34

   SPLIT A taxes plane crossings -> lands on the LONG axis
   SPLIT B taxes row movement    -> lands on the SHORT axis
   Kuiper is square, so both land on an axis of the same length
```

Worked out on Starlink's worst case:

```
  cross 36 planes, same row
     +Grid:    36 rungs                             =  36 hops
     split A:  36 rungs + 35 forced row hops        =  71 hops
               the row hops go +1, -1, +1, -1, netting zero,
               and 22 rows cannot absorb 35 of them
     split B:  36 rungs, no rail needed             =  36 hops
```

All-pairs BFS on the logical torus, unit hop costs:

| shell | grid | ISLs, 4 to 3 terminals | mean, +Grid | mean, split A | mean, split B | diameter, +Grid / A / B |
|---|---|---|---|---|---|---|
| Kuiper | 34 x 34 | 2312 to 1734 | 17.01 | 19.84 (+17%) | 19.84 (+17%) | 34 / 34 / 34 |
| OneWeb | 36 x 18 | 1296 to 972 | 13.52 | 18.77 (+39%) | 14.26 (**+5%**) | 27 / 36 / 27 |
| Starlink | 72 x 22 | 3168 to 2376 | 23.51 | 36.58 (+56%) | 24.07 (**+2%**) | 47 / 72 / 47 |

Starlink's split-A diameter lands on exactly `2 x 36`: every crossing paid double, nothing absorbed. Cutting a quarter of the links costs far more than a quarter of the path efficiency under split A and almost nothing under split B, on the same hardware budget.

## The two splits are one graph

```
   SPLIT A                              rotate 90 degrees, relabel the axes

        p ------>                              s ------>
   s   o---o   o---o                      p   o---o---o---o
   |   |   |   |   |                      |   |       |
   |   o   o---o   o                      |   o---o---o---o
   v   |   |   |   |                      v       |       |
       o---o   o---o                          o---o---o---o

                                         which is SPLIT B
```

Swap the roles of plane index and satellite index and one layout becomes the other, which is why Kuiper's two columns are identical. The closed form further down therefore covers both: same function, arguments swapped, `(planes, sats)` exchanged. Checked that way against BFS over 4 265 296 ordered pairs on OneWeb, Kuiper and Starlink, with zero mismatches.

So there's nothing to choose between them. One estimator serves both, the second layout costs a flag, and a sweep can report the pair as a range instead of defending a guess about hardware nobody has documented.

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

Split B breaks it in the mirror image. Every rung is present there, so `plane_edge_costs` comes out clean and plane crossings cost what they should; the gaps move into `row_edge_costs` instead, and walking more than one row inside a plane returns infinity. Same failure, different table.

Worth fixing whether or not the brick wall ever gets built: a distance policy handed a topology it can't represent should refuse at model-build time instead of reporting the whole constellation unreachable. Checking that each pivot row has contiguous rungs in `_build_torus_weight_model` and raising otherwise costs a few lines.

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

Every rung on that staircase exists by construction, for any starting row. Split B needs the same staircase turned ninety degrees, pairing planes rather than rows, which the transposed closed form already gives. Crossing one plane costs a rung plus a vertical hop, and the vertical hops come out of `row_edge_costs`, which the weight model already builds. The pivot loop at `fstate_calculation.py:1352` iterates row pairs instead of rows; table sizes and query cost don't change.

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

## Where this sits in the architecture

RINA splits the two things this page keeps conflating. The RMT is a stateless function that takes a PDU, reads its address field, and either delivers it locally or consults the forwarding table and posts it to an `(N-1)`-port (`rmt-spec-0002`, l.56-90), with that table keyed on `[destination-address, QoS-id]` (`rmt-spec-0003`, l.92-139). Building the table belongs to the Forwarding Table Generator, "sometimes called routing" (`rina-spec-overview-0005` section 5.3.2; Part 3-1 section 2.6.2.2, `rina-refmodel-part3-1-0015`, l.1176-1235). Interior routers do nothing beyond relaying: a border router is distinguished only by an extra level of multiplexing and PDU aggregation (`rina-refmodel-part3-1-0012`, l.935-1022).

Cutting a laser therefore touches one component. The distance estimator is an FTG policy, the brick wall needs a different policy, and the relay, the PDU format and the address layout all stay as they are. An interior satellite still holds forwarding state proportional to its degree, which on a brick wall is three.

Two things not to overclaim. The reference model has the RMT consult a table, so a policy that computes the next hop from the address rather than storing it per destination is compatible with the model rather than prescribed by it. And the policy detects nothing: `distance_mode` is configured, and in RINA terms selected per DIF at enrollment or by management, so "adapts to the topology" would be wrong.

## Caveats

The closed form is exact for unit hops. Under measured kilometre weights it becomes an approximation, the same way the single-row model is already an approximation today, and `forwarding_guard: progress` covers whatever residual error is left. Parity has to close, which rules out Telesat unless someone writes a seam variant. SpaceX publishes the laser count but not the wiring, so the brick wall is an argued assumption rather than a disclosure, and any paper text should carry the counting argument alongside it. How the three terminals get split is a second, separate guess on top of that one, which is why both layouts belong in the sweep.

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
