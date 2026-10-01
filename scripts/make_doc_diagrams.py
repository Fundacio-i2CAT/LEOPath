"""Draw the explanatory diagrams in docs/assets/diagrams/ as SVG.

Every diagram is plain SVG; the animated ones cycle through frames with a CSS
animation inside the file, so they play in a browser, on the docs site and on
GitHub without any script. ``--frames DIR`` also writes each frame as its own
static SVG, which is how the diagrams are checked by eye (rsvg-convert ignores
the animation).

    python scripts/make_doc_diagrams.py [--frames /tmp/frames]
"""

from __future__ import annotations

import argparse
import heapq
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "assets" / "diagrams"

# Okabe-Ito, the palette the paper figures use.
BLUE, GREEN, PINK = "#0072B2", "#009E73", "#CC79A7"
ORANGE, RED, SKY = "#E69F00", "#D55E00", "#56B4E9"
INK, GREY, LIGHT, PAPER = "#222222", "#8a8a8a", "#d9d9d9", "#ffffff"
NORTH, SOUTH = BLUE, ORANGE
FONT = "font-family='Inter, Helvetica, Arial, sans-serif'"


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=13, color=INK, anchor="middle", weight="normal", italic=False):
    style = " font-style='italic'" if italic else ""
    return (
        f"<text x='{x:.1f}' y='{y:.1f}' font-size='{size}' fill='{color}' "
        f"text-anchor='{anchor}' font-weight='{weight}'{style}>{esc(s)}</text>"
    )


def line(x1, y1, x2, y2, color=LIGHT, width=2, dash=None, marker=None, opacity=1.0):
    d = f" stroke-dasharray='{dash}'" if dash else ""
    m = f" marker-end='url(#{marker})'" if marker else ""
    return (
        f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' stroke='{color}' "
        f"stroke-width='{width}' stroke-linecap='round' opacity='{opacity}'{d}{m}/>"
    )


def path(points, color=GREEN, width=4, dash=None, marker=None, opacity=0.9):
    d = " ".join(("M" if i == 0 else "L") + f"{x:.1f},{y:.1f}" for i, (x, y) in enumerate(points))
    da = f" stroke-dasharray='{dash}'" if dash else ""
    m = f" marker-end='url(#{marker})'" if marker else ""
    return (
        f"<path d='{d}' fill='none' stroke='{color}' stroke-width='{width}' "
        f"stroke-linecap='round' stroke-linejoin='round' opacity='{opacity}'{da}{m}/>"
    )


def circle(x, y, r=7, fill=PAPER, stroke=INK, width=1.5):
    return (
        f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{r}' fill='{fill}' stroke='{stroke}' "
        f"stroke-width='{width}'/>"
    )


def rect(x, y, w, h, fill=PAPER, stroke=LIGHT, rx=8, width=1.5, dash=None):
    d = f" stroke-dasharray='{dash}'" if dash else ""
    return (
        f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' rx='{rx}' "
        f"fill='{fill}' stroke='{stroke}' stroke-width='{width}'{d}/>"
    )


def triangle(x, y, up=True, size=8, fill=NORTH):
    s = size
    pts = (
        [(x, y - s), (x - s, y + s * 0.7), (x + s, y + s * 0.7)]
        if up
        else [(x, y + s), (x - s, y - s * 0.7), (x + s, y - s * 0.7)]
    )
    return (
        "<polygon points='"
        + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts)
        + f"' fill='{fill}' stroke='{INK}' stroke-width='1'/>"
    )


def station(x, y, label=None, color=INK):
    out = [
        f"<path d='M{x - 10:.1f},{y + 8:.1f} L{x:.1f},{y - 8:.1f} L{x + 10:.1f},{y + 8:.1f} Z' "
        f"fill='{PAPER}' stroke='{color}' stroke-width='1.8'/>",
        circle(x, y - 10, 3.5, color, color, 1),
    ]
    if label:
        out.append(text(x, y + 24, label, 12, color))
    return "".join(out)


def defs():
    arrows = "".join(
        f"<marker id='a-{name}' viewBox='0 0 10 10' refX='8' refY='5' markerWidth='13' "
        f"markerHeight='13' markerUnits='userSpaceOnUse' orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z' "
        f"fill='{color}'/></marker>"
        for name, color in (
            ("ink", INK),
            ("green", GREEN),
            ("red", RED),
            ("blue", BLUE),
            ("orange", ORANGE),
            ("pink", PINK),
            ("grey", GREY),
        )
    )
    return f"<defs>{arrows}</defs>"


def svg(width, height, body, frames=0, seconds_per_frame=2.6, title=""):
    """Wrap a body in an SVG document; with frames > 0, cycle groups .f0 .. .fN-1."""
    style = ""
    if frames:
        total = frames * seconds_per_frame
        share = 100.0 / frames
        style = (
            "<style>"
            f".f{{opacity:0;animation:show {total:.1f}s infinite}}"
            f"@keyframes show{{0%{{opacity:1}}{share - 0.5:.2f}%{{opacity:1}}"
            f"{share:.2f}%{{opacity:0}}100%{{opacity:0}}}}"
            + "".join(
                f".f{i}{{animation-delay:{i * seconds_per_frame:.1f}s}}" for i in range(frames)
            )
            + "@media (prefers-reduced-motion: reduce){.f{animation:none}"
            f".f{frames - 1}{{opacity:1}}}}"
            "</style>"
        )
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' "
        f"width='{width}' height='{height}' {FONT} role='img'>"
        f"<title>{esc(title)}</title>{defs()}{style}"
        f"{rect(0.5, 0.5, width - 1, height - 1, PAPER, LIGHT, 12)}{body}</svg>"
    )


def frame(i, content):
    return f"<g class='f f{i}'>{content}</g>"


class Grid:
    """A logical torus drawn as columns (planes) by rows (slots)."""

    def __init__(self, planes, slots, x0, y0, dx, dy):
        self.planes, self.slots = planes, slots
        self.x0, self.y0, self.dx, self.dy = x0, y0, dx, dy

    def xy(self, p, s):
        return self.x0 + p * self.dx, self.y0 + s * self.dy

    def links(self, color=LIGHT, rungs=True, skip=()):
        out = []
        for p in range(self.planes):
            for s in range(self.slots):
                x, y = self.xy(p, s)
                if s + 1 < self.slots and ((p, s), (p, s + 1)) not in skip:
                    out.append(line(x, y, *self.xy(p, s + 1), color, 2))
                if rungs and p + 1 < self.planes and ((p, s), (p + 1, s)) not in skip:
                    out.append(line(x, y, *self.xy(p + 1, s), color, 2))
        return "".join(out)

    def sats(self, fill=PAPER, r=6, special=None):
        special = special or {}
        out = []
        for p in range(self.planes):
            for s in range(self.slots):
                f, st, rr = special.get((p, s), (fill, GREY, r))
                out.append(circle(*self.xy(p, s), rr, f, st, 1.5))
        return "".join(out)

    def route(self, cells, color=GREEN, width=5, marker="a-green", dash=None):
        return path([self.xy(*c) for c in cells], color, width, dash, marker)


def write(name, document, frames_dir, frame_docs=None):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(document)
    if frames_dir and frame_docs:
        for i, doc in enumerate(frame_docs):
            (frames_dir / f"{name[:-4]}_{i}.svg").write_text(doc)


def animated(name, width, height, static, frames, frames_dir, title, seconds=2.6):
    body = static + "".join(frame(i, f) for i, f in enumerate(frames))
    document = svg(width, height, body, len(frames), seconds, title)
    frame_docs = [svg(width, height, static + f, 0, title=title) for f in frames]
    write(name, document, frames_dir, frame_docs)


# --------------------------------------------------------------------------
# Ground-station addressing
# --------------------------------------------------------------------------


def address(frames_dir):
    g = Grid(7, 4, 110, 100, 46, 40)
    hot = (3, 1)
    body = [
        text(360, 32, "A station's address is its satellite's seat, plus x", 16, weight="bold"),
        g.links(),
        g.sats(special={hot: (GREEN, INK, 9)}),
        text(g.x0 - 30, g.y0 + 1.5 * g.dy, "slot", 12, GREY, "end"),
        text(g.x0 + 3 * g.dx, g.y0 + 3 * g.dy + 30, "plane", 12, GREY),
    ]
    hx, hy = g.xy(*hot)
    body.append(line(hx, hy, g.x0 - 12, g.y0 - 30, GREEN, 1.5))
    body.append(rect(g.x0 - 60, g.y0 - 52, 96, 24, PAPER, GREEN, 6))
    body.append(text(g.x0 - 12, g.y0 - 35, "(0, 3, 1)", 13, GREEN, weight="bold"))
    gx = 520
    for i, (dx, label) in enumerate(((-60, "x = 1"), (60, "x = 2"))):
        sx, sy = gx + dx, 250
        body.append(line(hx, hy, sx, sy - 14, GREY, 1.5, "5 4"))
        body.append(station(sx, sy))
        body.append(text(sx, sy + 34, label, 12))
        body.append(text(sx, sy + 52, f"(0, 3, 1, {i + 1})", 13, GREEN, weight="bold"))
    body.append(line(400, 262, 640, 262, GREY, 1.5))
    body.append(text(520, 120, "a satellite reading (0, 3, 1, …)", 12, GREY))
    body.append(text(520, 137, "knows which way to send it,", 12, GREY))
    body.append(text(520, 154, "with no table of ground stations", 12, GREY))
    write("address.svg", svg(720, 340, "".join(body), title="Ground-station address"), None)


def renumbering(frames_dir):
    static = [
        text(360, 30, "The satellite moves on, so the address changes", 16, weight="bold"),
        f"<path d='M40,140 Q360,40 680,140' fill='none' stroke='{LIGHT}' stroke-width='2' "
        "stroke-dasharray='6 5'/>",
        line(60, 250, 520, 250, GREY, 1.5),
        station(290, 236),
        text(290, 278, "madrid-gw-3  (name never changes)", 12, GREY),
        rect(560, 190, 140, 70, PAPER, LIGHT),
        text(630, 215, "peer station", 12),
        text(630, 233, "flow to madrid-gw-3", 11, GREY),
    ]
    names = ["(0, 3, 1)", "(0, 3, 2)", "(0, 4, 2)"]
    frames = []
    for k in range(3):
        f = []
        for j in range(3):
            sx = 290 - (j - k) * 170
            if not 20 < sx < 700:
                continue
            y = 140 - 50 * (1 - ((sx - 360) / 320) ** 2)
            on = j == k
            f.append(circle(sx, y, 10 if on else 7, GREEN if on else PAPER, INK if on else GREY))
            f.append(text(sx, y - 18, names[j], 12, GREEN if on else GREY))
        cx = 290
        cy = 140 - 50 * (1 - ((cx - 360) / 320) ** 2)
        f.append(line(290, 222, cx, cy + 12, GREEN, 2, "5 4"))
        f.append(rect(330, 150, 190, 30, PAPER, GREEN, 6))
        f.append(text(425, 170, f"address {names[k][:-1]}, 2)", 13, GREEN, weight="bold"))
        f.append(text(60, 320, f"t = {5 * k} min", 13, INK, "start", "bold"))
        if k:
            f.append(path([(330, 236), (552, 226)], PINK, 2.5, marker="a-pink"))
            f.append(text(440, 222, "flow update: new address", 11, PINK))
            f.append(text(440, 320, "flows survive: they're tied to names and ports", 12, GREY))
        else:
            f.append(text(440, 320, "attached to the satellite overhead", 12, GREY))
        frames.append("".join(f))
    animated("renumbering.svg", 720, 340, "".join(static), frames, frames_dir, "Renumbering")


def _shortest(grid, start, goal, cost, wrap=True):
    """Dijkstra on the grid, a torus when ``wrap``; cost(a, b) per hop."""
    P, S = grid.planes, grid.slots
    dist, prev = {start: 0.0}, {}
    heap = [(0.0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == goal:
            break
        if d > dist[u]:
            continue
        p, s = u
        for v in (((p + 1) % P, s), ((p - 1) % P, s), (p, (s + 1) % S), (p, (s - 1) % S)):
            if not wrap and (abs(v[0] - p) > 1 or abs(v[1] - s) > 1):
                continue
            nd = d + cost(u, v)
            if nd < dist.get(v, float("inf")):
                dist[v], prev[v] = nd, u
                heapq.heappush(heap, (nd, v))
    cells, u = [goal], goal
    while u != start:
        u = prev[u]
        cells.append(u)
    return cells[::-1], dist[goal]


def _unwrap(cells, grid):
    """Split a torus route into drawable segments, cutting at wrap-arounds."""
    segments, current = [], [cells[0]]
    for a, b in zip(cells, cells[1:]):
        if abs(a[0] - b[0]) > 1 or abs(a[1] - b[1]) > 1:
            segments.append(current)
            current = [b]
        else:
            current.append(b)
    segments.append(current)
    return segments


def two_halves(frames_dir):
    g = Grid(12, 6, 70, 80, 46, 40)
    north, south, mumbai = (2, 1), (8, 4), (3, 2)
    special = {
        north: (NORTH, INK, 10),
        south: (SOUTH, INK, 10),
        mumbai: (GREEN, INK, 10),
    }
    static = [
        text(360, 30, "Two satellites over Nairobi, half the network apart", 16, weight="bold"),
        g.links(),
        g.sats(special=special),
        text(g.xy(*north)[0], g.xy(*north)[1] - 18, "N", 13, NORTH, weight="bold"),
        text(g.xy(*south)[0], g.xy(*south)[1] - 18, "S", 13, SOUTH, weight="bold"),
        text(g.xy(*mumbai)[0] + 20, g.xy(*mumbai)[1] + 22, "Mumbai's", 11, GREEN, "start"),
        text(g.x0 - 8, g.y0 + 5 * g.dy + 40, "the grid wraps round: right edge meets left, bottom meets top", 11, GREY, "start"),
    ]
    unit = lambda a, b: 1.0  # noqa: E731
    near, _ = _shortest(g, north, mumbai, unit)
    far, _ = _shortest(g, south, mumbai, unit)
    legend_y = 375
    frames = [
        text(360, legend_y, "N (northbound) and S (southbound): both 682 km from Nairobi", 13)
        + text(360, legend_y + 20, "nearest-satellite attachment picks whichever is closer", 12, GREY),
        "".join(g.route(seg, NORTH, 5, None) for seg in _unwrap(near, g))
        + text(360, legend_y, "from N: 6 275 km to Mumbai's satellite, 23 ms", 13, NORTH, weight="bold")
        + text(360, legend_y + 20, "a station attached to N is close to Mumbai", 12, GREY),
        "".join(g.route(seg, SOUTH, 5, None) for seg in _unwrap(far, g))
        + text(360, legend_y, "from S: 52 665 km, 178 ms, about half-way round", 13, SOUTH, weight="bold")
        + text(360, legend_y + 20, "same city, same distance overhead, wrong half", 12, GREY),
    ]
    animated("two-halves.svg", 720, 410, "".join(static), frames, frames_dir, "Two halves", 3.2)


def attachment_orders(frames_dir):
    panels = [
        ("nearest", "takes the closest: S1", ["S1"]),
        ("nearest_ascending", "prefers northbound: N1", ["N1"]),
        ("one_per_half  (K = 2)", "one of each: N1 and S1", ["N1", "S1"]),
    ]
    sky = [("N1", True, -70, 62), ("S1", False, 10, 48), ("N2", True, 75, 96), ("S2", False, -120, 104)]
    body = [text(360, 30, "Which satellites a station attaches to", 16, weight="bold")]
    for i, (name, note, picked) in enumerate(panels):
        cx = 125 + i * 235
        body.append(rect(cx - 108, 48, 216, 250, PAPER, LIGHT))
        body.append(text(cx, 72, name, 13, INK, weight="bold"))
        gy = 250
        for label, up, dx, h in sky:
            sx, sy = cx + dx * 0.75, gy - 40 - h
            chosen = label in picked
            if chosen:
                body.append(line(cx, gy - 12, sx, sy + 9, GREEN, 3))
            else:
                body.append(line(cx, gy - 12, sx, sy + 9, LIGHT, 1.5, "4 4"))
            body.append(triangle(sx, sy, up, 9 if chosen else 7, NORTH if up else SOUTH))
            body.append(text(sx, sy - 13, label, 11, INK if chosen else GREY, weight="bold" if chosen else "normal"))
        body.append(station(cx, gy))
        body.append(line(cx - 90, gy + 10, cx + 90, gy + 10, GREY, 1.2))
        body.append(text(cx, 286, note, 12, GREEN))
    body.append(triangle(250, 325, True, 7, NORTH))
    body.append(text(262, 330, "northbound pass", 12, INK, "start"))
    body.append(triangle(410, 325, False, 7, SOUTH))
    body.append(text(422, 330, "southbound pass", 12, INK, "start"))
    body.append(text(360, 352, "S1 is the closest satellite; N1 the closest northbound one", 11, GREY))
    write("attachment-orders.svg", svg(720, 368, "".join(body), title="Attachment orders"), None)


def address_policies(frames_dir):
    rows = [
        ("nearest", "B's address follows its nearest satellite;|every caller uses it", ["n", "n"], "B"),
        ("sticky_nearest (default)", "B keeps its current address while it's|still attached; every caller uses it", ["n", "n"], "B"),
        ("requester_aware", "B answers each caller with its address|on that caller's half", ["n", "s"], "B"),
        ("per_flow_pair", "each source picks both ends for its flow|(beyond RINA, kept as a bound)", ["n", "s"], "source"),
    ]
    body = [text(360, 30, "Which of B's addresses a flow uses", 16, weight="bold")]
    for i, (name, note, uses, chooser) in enumerate(rows):
        y = 62 + i * 92
        body.append(rect(16, y, 688, 80, PAPER, LIGHT))
        body.append(text(30, y + 22, name, 13, INK, "start", "bold"))
        for k, part in enumerate(note.split("|")):
            body.append(text(230, y + 22 + k * 16, part, 11, GREY, "start"))
        body.append(text(30, y + 42, f"who chooses: {chooser}", 11, PINK if chooser == "source" else GREEN, "start"))
        bx, by = 640, y + 46
        body.append(station(bx, by + 6))
        body.append(text(bx, by - 26, "B", 12, INK, weight="bold"))
        body.append(triangle(bx - 30, by - 20, True, 7, NORTH))
        body.append(triangle(bx + 30, by - 20, False, 7, SOUTH))
        for j, (label, half) in enumerate((("A1 (north half)", NORTH), ("A2 (south half)", SOUTH))):
            ax, ay = 530, y + 26 + j * 30
            body.append(text(ax - 12, ay + 4, label, 11, half, "end"))
            target = (bx - 30, by - 20) if uses[j] == "n" else (bx + 30, by - 20)
            colour = NORTH if uses[j] == "n" else SOUTH
            body.append(path([(ax, ay), (target[0] - (8 if uses[j] == "n" else -8), target[1] + 2)], colour, 2.2, marker="a-blue" if uses[j] == "n" else "a-orange"))
    write("address-policies.svg", svg(720, 440, "".join(body), title="Address policies"), None)


def smart_directory(frames_dir):
    lanes = [("A", 90), ("A's satellites", 250), ("B's IPCP / directory", 470), ("B", 630)]
    static = [text(360, 30, "Smart directory: B answers with the address that suits A", 16, weight="bold")]
    for name, x in lanes:
        static.append(rect(x - 70, 48, 140, 30, PAPER, LIGHT, 6))
        static.append(text(x, 68, name, 12, INK, weight="bold"))
        static.append(line(x, 80, x, 330, LIGHT, 1.5, "4 5"))
    msgs = [
        (90, 470, 115, "allocate flow to B  (from A's north address)", INK),
        (470, 90, 165, "use B's north address (0, 28, 0, x)", GREEN),
        (90, 250, 215, "send via A's north satellite", NORTH),
        (250, 630, 265, "data: stays on the north half, ~23 ms not ~178", GREEN),
    ]
    frames = []
    for k in range(len(msgs)):
        f = []
        for j, (x1, x2, y, label, colour) in enumerate(msgs[: k + 1]):
            current = j == k
            c = colour if current else GREY
            f.append(path([(x1, y), (x2 - (8 if x2 > x1 else -8), y)], c, 2.5 if current else 1.5, marker="a-ink" if not current else ("a-green" if colour == GREEN else "a-blue" if colour == NORTH else "a-ink")))
            f.append(text((x1 + x2) / 2, y - 8, label, 12 if current else 11, c, weight="bold" if current else "normal"))
        captions = [
            "the request carries the caller's address, so B can see which half A is on",
            "B holds one address per half and returns the one on A's half",
            "A uplinks through whichever of its own satellites suits that address",
            "the flow keeps B's address until B's next renumbering",
        ]
        f.append(text(360, 358, captions[k], 12, GREY))
        frames.append("".join(f))
    animated("smart-directory.svg", 720, 378, "".join(static), frames, frames_dir, "Smart directory", 3.0)


# --------------------------------------------------------------------------
# Forwarding algorithms
# --------------------------------------------------------------------------


def _rung_cost(s, slots):
    """Inter-plane link cost: long near the equator (middle rows), short near the poles."""
    middle = (slots - 1) / 2
    return 1.0 + 1.6 * (1 - abs(s - middle) / middle)


def topological_walk(frames_dir):
    g = Grid(8, 6, 90, 70, 70, 46)
    src, dst = (0, 2), (5, 3)

    def cost(a, b):
        if a[0] == b[0]:
            return 1.0
        return _rung_cost(a[1], g.slots)

    est = {}
    for p in range(g.planes):
        for s in range(g.slots):
            est[(p, s)] = _shortest(g, (p, s), dst, cost, False)[1] if (p, s) != dst else 0.0
    route, _ = _shortest(g, src, dst, cost, False)
    static = [
        text(360, 30, "Topological forwarding: one local decision per hop", 16, weight="bold"),
        g.links(),
        g.sats(special={dst: (GREEN, INK, 10)}),
        text(g.xy(*dst)[0], g.xy(*dst)[1] + 26, "destination", 11, GREEN, weight="bold"),
    ]
    frames = []
    for k in range(len(route)):
        cur = route[k]
        f = [g.route(route[: k + 1], GREEN, 5, None)] if k else []
        f.append(circle(*g.xy(*cur), 11, PINK, INK, 2))
        if cur != dst:
            p, s = cur
            for v in ((p + 1, s), (p - 1, s), (p, s + 1), (p, s - 1)):
                if 0 <= v[0] < g.planes and 0 <= v[1] < g.slots:
                    x, y = g.xy(*v)
                    nxt = route[k + 1] == v
                    f.append(rect(x - 17, y - 29, 34, 17, PAPER, GREEN if nxt else LIGHT, 4))
                    f.append(text(x, y - 16, f"{est[v]:.1f}", 11, GREEN if nxt else GREY, weight="bold" if nxt else "normal"))
            f.append(text(360, 380, "estimate the distance from each neighbour to the destination address; take the lowest", 12, GREY))
        else:
            f.append(text(360, 380, "delivered, with no routing table: only the address and the shell's geometry", 12, GREEN, weight="bold"))
        frames.append("".join(f))
    animated("topological-walk.svg", 720, 400, "".join(static), frames, frames_dir, "Topological forwarding", 1.8)


def dra_vs_pivot(frames_dir):
    body = [text(360, 30, "Counting hops vs counting kilometres", 16, weight="bold")]
    for i, (title, colour, crossing) in enumerate(
        (("DRA: every hop costs 1", PINK, 2), ("pivot estimator: rungs cost their length", GREEN, 0))
    ):
        g = Grid(5, 5, 60 + i * 350, 90, 62, 52)
        body.append(text(g.x0 + 2 * g.dx, 66, title, 13, colour, weight="bold"))
        for s in range(g.slots):
            w = 1.5 + 9 * (_rung_cost(s, g.slots) - 1) / 1.6
            for p in range(g.planes - 1):
                body.append(line(*g.xy(p, s), *g.xy(p + 1, s), "#c4c4c4", w))
        for p in range(g.planes):
            for s in range(g.slots - 1):
                body.append(line(*g.xy(p, s), *g.xy(p, s + 1), LIGHT, 2))
        src, dst = (0, 2), (4, 2)
        body.append(g.sats(special={src: (PINK if i == 0 else GREEN, INK, 9), dst: (INK, INK, 9)}))
        if crossing == 2:
            cells = [(p, 2) for p in range(5)]
            note = "4 hops across the equator, where rungs are longest"
        else:
            cells = [(0, 2), (0, 1), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2)]
            note = "up a rail to short rungs, across, back down"
        body.append(g.route(cells, colour, 5, "a-green" if i else "a-pink"))
        body.append(text(g.x0 + 2 * g.dx, 340, note, 12, GREY))
    body.append(text(360, 372, "thick rungs are long links (low latitude); thin ones are short (near the poles)", 12, GREY))
    body.append(text(360, 392, "the pivot estimator needs each row's rung length: 7 Walker constants and the clock give it", 12, GREY))
    write("dra-vs-pivot.svg", svg(720, 410, "".join(body), title="DRA vs pivot"), None)


def link_state(frames_dir):
    g = Grid(6, 4, 70, 80, 56, 50)
    hot = (1, 1)
    body = [
        text(360, 30, "Link-state: every satellite knows every destination", 16, weight="bold"),
        g.links(),
        g.sats(special={hot: (BLUE, INK, 10)}),
    ]
    for p, s, dp, ds in ((3, 2, 1, 0), (3, 2, 0, -1), (3, 2, -1, 0), (3, 2, 0, 1)):
        x1, y1 = g.xy(p, s)
        body.append(path([(x1, y1), (x1 + dp * 40, y1 + ds * 36)], ORANGE, 2.5, marker="a-orange"))
    body.append(circle(*g.xy(3, 2), 9, ORANGE, INK))
    body.append(text(g.xy(3, 2)[0], g.y0 + 3 * g.dy + 30, "a change is flooded to every satellite", 11, ORANGE))
    hx, hy = g.xy(*hot)
    tx = 440
    body.append(line(hx + 10, hy, tx - 6, 96, BLUE, 1.5, "4 4"))
    body.append(rect(tx, 60, 250, 240, PAPER, BLUE))
    body.append(text(tx + 125, 82, "forwarding table of one satellite", 12, BLUE, weight="bold"))
    for k, (d, nh) in enumerate((("sat 0", "west"), ("sat 1", "north"), ("sat 2", "north"), ("…", "…"), ("gs 7", "east"), ("gs 8", "south"), ("…", "…"))):
        body.append(text(tx + 30, 108 + k * 24, d, 12, INK, "start"))
        body.append(text(tx + 160, 108 + k * 24, f"→ {nh}", 12, GREY, "start"))
    body.append(text(tx + 125, 290, "one entry per destination", 11, BLUE))
    body.append(text(360, 335, "state grows with the constellation, and every topology change updates it", 12, GREY))
    write("link-state.svg", svg(720, 355, "".join(body), title="Link-state"), None)


def explicit_path(frames_dir):
    g = Grid(7, 4, 70, 120, 52, 46)
    cells = [(0, 1), (1, 1), (2, 1), (3, 1), (3, 2), (4, 2), (5, 2), (6, 2)]
    body = [
        text(360, 30, "Explicit-path: the source writes the route into the packet", 16, weight="bold"),
        g.links(),
        g.sats(special={cells[0]: (PINK, INK, 10), cells[-1]: (INK, INK, 9)}),
        g.route(cells, PINK, 5, "a-pink"),
    ]
    for k, c in enumerate((cells[3], cells[4])):
        x, y = g.xy(*c)
        body.append(circle(x, y, 9, PAPER, PINK, 3))
    body.append(rect(70, 56, 320, 34, PAPER, PINK, 6))
    body.append(text(84, 78, "header: [ seg (3,1), seg (3,2), egress (6,2) ]", 12, PINK, "start", "bold"))
    body.append(text(560, 120, "the source holds a full topology", 12, GREY))
    body.append(text(560, 138, "view and replans every R snapshots", 12, GREY))
    body.append(text(560, 176, "transit satellites only follow", 12, GREY))
    body.append(text(560, 194, "the segments in the header", 12, GREY))
    body.append(text(560, 232, "the last egress can be repaired", 12, GREY))
    body.append(text(560, 250, "if it moved since planning", 12, GREY))
    body.append(text(360, 330, "small tables in transit, but a bigger header and a planner that knows everything", 12, GREY))
    write("explicit-path.svg", svg(720, 350, "".join(body), title="Explicit-path"), None)


def failures(frames_dir):
    g = Grid(7, 5, 90, 70, 72, 52)
    dst = (5, 2)
    stuck = (3, 2)
    broken = ((3, 2), (4, 2))
    dead = (4, 2)
    static_parts = [
        text(360, 30, "When links fail: guard, local repair, exception entry", 16, weight="bold"),
    ]
    sx, sy = g.xy(*broken[0])
    ex, ey = g.xy(*broken[1])
    mx, my = (sx + ex) / 2, (sy + ey) / 2
    cross = line(mx - 7, my - 7, mx + 7, my + 7, RED, 3) + line(mx - 7, my + 7, mx + 7, my - 7, RED, 3)

    def scene(skip, dead_sat=None):
        sp = {dst: (GREEN, INK, 10)}
        if dead_sat:
            sp[dead_sat] = ("#f3d3c3", RED, 8)
        return g.links(skip=skip) + g.sats(special=sp) + text(g.xy(*dst)[0], g.xy(*dst)[1] + 26, "D", 12, GREEN, weight="bold")

    walk = [(0, 2), (1, 2), (2, 2), (3, 2)]
    f0 = (
        scene({broken}) + cross + g.route(walk, PINK, 5, None) + circle(*g.xy(*stuck), 11, PINK, INK, 2)
        + text(360, 360, "guard: forward only to a neighbour strictly closer to D", 13, INK, weight="bold")
        + text(360, 380, "the only closer one is behind the failed link: stop here, never loop", 12, GREY)
    )
    detour = [(3, 2), (3, 1), (4, 1), (4, 2), (5, 2)]
    f1 = (
        scene({broken}) + cross + g.route(walk, GREY, 3, None) + g.route(detour, ORANGE, 5, "a-orange")
        + text(360, 360, "local repair: reach the same next hop around one grid square", 13, INK, weight="bold")
        + text(360, 380, "needs only the link state within two hops", 12, GREY)
    )
    dead_skip = {((3, 2), (4, 2)), ((4, 1), (4, 2)), ((4, 2), (4, 3)), ((4, 2), (5, 2))}
    f2 = (
        scene(dead_skip, dead) + g.route(walk, PINK, 5, None) + circle(*g.xy(*stuck), 11, PINK, INK, 2)
        + text(360, 360, "a whole satellite is down: no square detour reaches it", 13, INK, weight="bold")
        + text(360, 380, "the walk toward D stops at the pink satellite", 12, GREY)
    )
    exc = [(3, 2), (3, 3), (4, 3), (5, 3), (5, 2)]
    sxp, syp = g.xy(*stuck)
    f3 = (
        scene(dead_skip, dead) + g.route(walk, GREY, 3, None) + g.route(exc, GREEN, 5, "a-green")
        + rect(sxp - 120, syp - 70, 150, 34, PAPER, GREEN, 6)
        + text(sxp - 45, syp - 48, "entry: for D, go south", 12, GREEN, weight="bold")
        + text(360, 360, "exception entry: one line, only where the walk broke", 13, INK, weight="bold")
        + text(360, 380, "along the shortest working path; every flow to D shares it", 12, GREY)
    )
    animated("failures.svg", 720, 400, "".join(static_parts), [f0, f1, f2, f3], frames_dir, "Failures", 3.2)


def isl_wirings(frames_dir):
    panels = [("Ring", "ring"), ("+Grid", "grid"), ("+Grid, seam open", "seam"), ("Brick wall (3 lasers)", "brick")]
    body = [text(360, 30, "How satellites are wired", 16, weight="bold")]
    for i, (name, kind) in enumerate(panels):
        g = Grid(4, 5, 48 + i * 172, 80, 30, 36)
        body.append(text(g.x0 + 1.5 * g.dx, 62, name, 12, INK, weight="bold"))
        for p in range(g.planes):
            for s in range(g.slots):
                x, y = g.xy(p, s)
                if s + 1 < g.slots:
                    body.append(line(x, y, *g.xy(p, s + 1), BLUE, 2))
                if p + 1 < g.planes and kind != "ring":
                    if kind == "brick" and (p + s) % 2:
                        continue
                    body.append(line(x, y, *g.xy(p + 1, s), GREEN, 2))
        if kind in ("grid", "brick"):
            for s in range(g.slots):
                if kind == "brick" and (3 + s) % 2:
                    continue
                y = g.xy(0, s)[1]
                body.append(line(g.x0 - 16, y, g.x0, y, GREEN, 2, "3 3"))
                body.append(line(g.xy(3, s)[0], y, g.xy(3, s)[0] + 16, y, GREEN, 2, "3 3"))
        if kind == "seam":
            body.append(line(g.xy(3, 0)[0] + 14, 66, g.xy(3, 0)[0] + 14, g.xy(3, 4)[1] + 10, RED, 2, "5 4"))
        body.append(g.sats(r=5))
        notes = {
            "ring": "along each plane only;planes never meet",
            "grid": "4 links each;the edges wrap round",
            "seam": "no links across the seam,;where planes run;opposite ways",
            "brick": "3 links each;rungs alternate;like brickwork",
        }
        words = notes[kind].split(";")
        for k, wds in enumerate(words):
            body.append(text(g.x0 + 1.5 * g.dx, 268 + k * 15, wds.strip(), 10.5, GREY))
    body.append(line(250, 340, 280, 340, BLUE, 3))
    body.append(text(286, 344, "along the plane (rail)", 11, INK, "start"))
    body.append(line(430, 340, 460, 340, GREEN, 3))
    body.append(text(466, 344, "between planes (rung)", 11, INK, "start"))
    write("isl-wirings.svg", svg(720, 362, "".join(body), title="ISL wirings"), None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, default=None, help="also write each frame as a static SVG here")
    args = parser.parse_args()
    if args.frames:
        args.frames.mkdir(parents=True, exist_ok=True)
    for draw in (
        address,
        renumbering,
        two_halves,
        attachment_orders,
        address_policies,
        smart_directory,
        topological_walk,
        dra_vs_pivot,
        link_state,
        explicit_path,
        failures,
        isl_wirings,
    ):
        draw(args.frames)
    print(f"diagrams written to {OUT}")


if __name__ == "__main__":
    main()
