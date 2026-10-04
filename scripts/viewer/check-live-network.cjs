const assert = require('node:assert/strict');
const fs = require('node:fs');
const network = require('../../docs/cesium/live-network.js');
const fixtures = JSON.parse(fs.readFileSync(0,'utf8'));
let comparisons=0;
for (const f of fixtures.graphs) {
  if (f.invalid) {
    assert.ok(network.unsupportedReason(f.planes,f.slots,f.topology,f.planeWrap));
    assert.throws(()=>network.buildGraph(f.planes,f.slots,f.topology,f.planeWrap));
    continue;
  }
  const graph = network.buildGraph(f.planes,f.slots,f.topology,f.planeWrap);
  const sorted = graph.edges.slice().sort((a,b)=>a[0]-b[0] || a[1]-b[1]);
  assert.deepEqual(sorted,f.edges,`${f.shell} ${f.topology} must match simulator wiring`);
  for (const [a,b] of graph.edges) {
    assert.ok(graph.adjacency[a].includes(b) && graph.adjacency[b].includes(a));
  }
  if (f.topology.startsWith('brick')) {
    assert.ok(graph.adjacency.every(n=>n.length <= 3),'A brick node has at most three ISLs');
    if (f.planeWrap) assert.ok(graph.adjacency.every(n=>n.length===3));
  }
  const [a,b] = graph.edges[0];
  assert.ok(!network.liveNeighbors(graph,a,new Set([b]),new Set()).includes(b));
  assert.deepEqual(network.liveNeighbors(graph,a,new Set([a]),new Set()),[]);
  assert.ok(!network.liveNeighbors(graph,a,new Set(),new Set([network.linkKey(a,b)])).includes(b));
  comparisons++;
}
let distances=0;
const infinite = v => Array.isArray(v) ? v.map(infinite) : v===null ? Infinity : v;
for (const f of fixtures.models) {
  const model = {...f,rowPathCosts:infinite(f.rowPathCosts),planePathCosts:infinite(f.planePathCosts)};
  for(let a=0;a<f.orbits*f.satsPerOrbit;a++) for(let b=0;b<f.orbits*f.satsPerOrbit;b++) {
    assert.equal(network.brickDistance(model,Math.floor(a/f.satsPerOrbit),a%f.satsPerOrbit,Math.floor(b/f.satsPerOrbit),b%f.satsPerOrbit),f.distances[a][b]);
    distances++;
  }
}
console.log(`PASS ${comparisons} shell/wiring graphs, invalid parity constraints, symmetric links and failure filtering; ${distances} brick estimates match Python`);
