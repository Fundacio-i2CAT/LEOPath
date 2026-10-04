"""Check replay fidelity: delivery, live links, exception choices and recovery."""
import argparse
import json
import math
from pathlib import Path


def verify(path):
    data = json.loads(path.read_text())
    n = data['planes'] * data['slots']
    assert len(data['positions']) == n
    total_routes = 0
    for frame in data['frames']:
        edges = {tuple(sorted((a, b))): length for a, b, length in frame['edgeLengths']}
        failed = {tuple(sorted(edge)) for edge in frame['failedLinks']}
        assert not (set(edges) & failed), (path, frame['label'], 'failed edge still live')
        entries = {(a, d): hop for a, d, hop in frame['exceptions']}
        assert frame['stats']['rawEntries'] == len(entries)
        assert frame['stats']['regionEntries'] <= len(entries)
        for (sat, _dst), hop in entries.items():
            assert tuple(sorted((sat, hop))) in edges
        delivered = reachable = 0
        for route in frame['routes'].values():
            reachable += bool(route['linkState'])
            delivered += route['reason'] == 'delivered'
            for kind in ('topological', 'linkState'):
                walk = route[kind]
                if not walk:
                    continue
                assert walk[0] == route['sourceSatellite'] and walk[-1] == route['targetSatellite']
                assert len(walk) == len(set(walk)), (path, frame['label'], 'loop')
                for sat, hop in zip(walk, walk[1:]):
                    assert tuple(sorted((sat, hop))) in edges
                    if kind == 'topological':
                        installed = entries.get((sat, route['targetSatellite']))
                        rule = frame['decisions'][str(route['targetSatellite'])]['ruleNext'][sat]
                        assert hop == (installed if installed is not None else rule)
                assert math.isfinite(route[kind + 'DelayMs']) and route[kind + 'DelayMs'] > 0
            if route['topological']:
                assert route['topologicalDelayMs'] >= route['linkStateDelayMs'] - 0.0001
            total_routes += 1
        assert delivered == frame['stats']['delivered']
        assert reachable == frame['stats']['reachable']
        assert delivered == reachable, (path, frame['label'], 'reachable walk undelivered')
        assert frame['stats']['unresolved'] == 0
        if 'recovered' in frame['label'].lower():
            assert frame['exceptions'] == data['frames'][0]['exceptions']
            assert frame['routes'] == data['frames'][0]['routes']
    pair = data['defaultPair']
    initial, failed = data['frames'][:2]
    selected = failed['routes'][pair]
    assert selected['topological'] != initial['routes'][pair]['topological']
    assert any((sat, selected['targetSatellite']) in {(s,d) for s,d,_ in failed['exceptions']} for sat in selected['topological'])
    print(f"PASS {data['id']}: {len(data['frames'])} phases, {total_routes} flow snapshots")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', nargs='?', type=Path, default=Path(__file__).resolve().parents[1] / 'docs/cesium/replays')
    args = parser.parse_args()
    for path in sorted(args.directory.glob('*-*.json')):
        verify(path)
