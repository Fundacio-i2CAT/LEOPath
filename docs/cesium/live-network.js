/* Shared wiring for live rendering, shortest paths and pivot decisions. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.LEOPathNetwork = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";
  const mod = (n, m) => ((n % m) + m) % m;
  const linkKey = (a, b) => `${Math.min(a, b)}:${Math.max(a, b)}`;
  function unsupportedReason(planes, slots, topology, planeWrap) {
    if (topology === "brick_a" && planeWrap && planes % 2)
      return "Brick A needs an even plane count on a closed shell.";
    if (topology === "brick_b" && slots % 2)
      return "Brick B needs an even number of satellites per plane.";
    return "";
  }
  function buildGraph(planes, slots, topology, planeWrap) {
    if (!Number.isInteger(planes) || !Number.isInteger(slots) || planes < 3 || slots < 3)
      throw new Error("Invalid shell dimensions.");
    if (!["none", "ring", "grid", "brick_a", "brick_b"].includes(topology))
      throw new Error("Unknown ISL topology.");
    const reason = unsupportedReason(planes, slots, topology, planeWrap);
    if (reason) throw new Error(reason);
    const adjacency = Array.from({length:planes * slots}, () => []);
    const edges = [], edgeKeys = new Set();
    function add(a, b) {
      const key = linkKey(a, b);
      if (edgeKeys.has(key)) return;
      edgeKeys.add(key);
      edges.push([Math.min(a,b), Math.max(a,b)]);
      adjacency[a].push(b); adjacency[b].push(a);
    }
    if (topology !== "none") {
      for (let p=0; p<planes; p++) for (let s=0; s<slots; s++) {
        const a = p * slots + s;
        const even = (p+s) % 2 === 0;
        if (topology !== "brick_b" || even) add(a, p * slots + (s+1) % slots);
        if (topology !== "ring" && (planeWrap || p+1<planes) && (topology !== "brick_a" || even))
          add(a, ((p+1) % planes) * slots + s);
      }
    }
    return {planes, slots, topology, planeWrap, adjacency, edges, edgeKeys};
  }
  function liveNeighbors(graph, index, failedSatellites, failedLinks) {
    if (failedSatellites.has(index)) return [];
    return graph.adjacency[index].filter(next => !failedSatellites.has(next) && !failedLinks.has(linkKey(index,next)));
  }
  // Port of the simulator's _brick_hops and _brick_pivot_distance.
  function brickHops(start, startRow, end, endRow, crossingsModulus, rowsModulus, crossingWraps, rowsWrap) {
    const steps = mod(endRow-startRow, rowsModulus);
    const rowDistance = rowsWrap ? Math.min(steps, rowsModulus-steps) : Math.abs(endRow-startRow);
    let best = null;
    for (const [crossings, needsShift, wraps] of [
      [mod(end-start,crossingsModulus), (start+startRow)%2===1, end<start],
      [mod(start-end,crossingsModulus), (start+startRow)%2===0, end>start]
    ]) {
      if (wraps && crossings && !crossingWraps) continue;
      let moves = rowDistance;
      if (crossings) {
        moves = Math.max(rowDistance,crossings-1+(needsShift?1:0));
        if ((moves-rowDistance)%2) moves++;
      }
      const candidate = [crossings,moves];
      if (!best || crossings+moves < best[0]+best[1]) best = candidate;
    }
    return best;
  }
  function brickDistance(model, srcPlane, srcSlot, dstPlane, dstSlot) {
    const {orbits, satsPerOrbit, rowPathCosts, planePathCosts, brick} = model;
    let best = Infinity;
    if (model.topology === "brick_a") {
      for (let pivot=0; pivot<satsPerOrbit; pivot++) {
        const hops = brickHops(srcPlane,pivot,dstPlane,dstSlot,orbits,satsPerOrbit,brick.planeWrap,true);
        if (hops) best = Math.min(best,rowPathCosts[srcPlane][srcSlot][pivot]+hops[0]*brick.rung+hops[1]*brick.rail);
      }
    } else {
      for (let pivot=0; pivot<orbits; pivot++) {
        const leg = planePathCosts[srcSlot][srcPlane][pivot];
        if (!Number.isFinite(leg)) continue;
        const hops = brickHops(srcSlot,pivot,dstSlot,dstPlane,satsPerOrbit,orbits,true,brick.planeWrap);
        if (hops) best = Math.min(best,leg+hops[0]*brick.rail+hops[1]*brick.rung);
      }
    }
    return best;
  }
  return {linkKey, unsupportedReason, buildGraph, liveNeighbors, brickHops, brickDistance};
});
