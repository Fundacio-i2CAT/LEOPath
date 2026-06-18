(function () {
  "use strict";

  const state = {
    viewer: null,
    metadata: null,
    activeConfig: null,
    activeRecords: [],
    activeSource: "",
    gslCache: null,
    routeCache: null,
    baseStatsRows: [],
    loading: false,
  };

  const els = {};

  document.addEventListener("DOMContentLoaded", init);

  async function init() {
    bindElements();
    bindEvents();

    if (!window.Cesium || !window.satellite) {
      setStatus("CesiumJS or satellite.js failed to load.", true);
      return;
    }

    try {
      state.viewer = createViewer();
      resetCamera(0);
      state.metadata = await fetchJson("constellations.json");
      populateConstellations(state.metadata.constellations || []);
      populateRouteSelects(state.metadata.groundStations || []);
      applyDefaultClockSpeed();
      await loadSelectedConstellation();
    } catch (error) {
      setStatus(`Failed to initialize viewer: ${error.message}`, true);
    }
  }

  function bindElements() {
    els.container = document.getElementById("cesiumContainer");
    els.select = document.getElementById("constellationSelect");
    els.islTopology = document.getElementById("islTopology");
    els.routeSource = document.getElementById("routeSource");
    els.routeTarget = document.getElementById("routeTarget");
    els.showLinkStateRoute = document.getElementById("showLinkStateRoute");
    els.showGround = document.getElementById("showGround");
    els.showGsl = document.getElementById("showGsl");
    els.showGslLabel = document.getElementById("showGslLabel");
    els.fullDensity = document.getElementById("fullDensity");
    els.fullDensityLabel = document.getElementById("fullDensityLabel");
    els.speedSlider = document.getElementById("speedSlider");
    els.speedLabel = document.getElementById("speedLabel");
    els.resetButton = document.getElementById("resetButton");
    els.hidePanel = document.getElementById("hidePanel");
    els.showPanel = document.getElementById("showPanel");
    els.panel = document.querySelector(".panel");
    els.status = document.getElementById("status");
    els.stats = document.getElementById("stats");
    els.tleLink = document.getElementById("tleLink");
  }

  function bindEvents() {
    els.select.addEventListener("change", loadSelectedConstellation);
    els.islTopology.addEventListener("change", reloadActiveConstellation);
    els.routeSource.addEventListener("change", updateRoute);
    els.routeTarget.addEventListener("change", updateRoute);
    els.showLinkStateRoute.addEventListener("change", updateRoute);
    els.showGround.addEventListener("change", function () {
      syncGslControl();
      reloadActiveConstellation();
    });
    els.showGsl.addEventListener("change", reloadActiveConstellation);
    els.fullDensity.addEventListener("change", reloadActiveConstellation);
    els.resetButton.addEventListener("click", function () { resetCamera(0.8); });
    els.hidePanel.addEventListener("click", function () { setPanelVisible(false); });
    els.showPanel.addEventListener("click", function () { setPanelVisible(true); });
    els.speedSlider.addEventListener("input", function () {
      const speed = Number(els.speedSlider.value);
      els.speedLabel.textContent = `${speed}x`;
      if (state.viewer) {
        state.viewer.clock.multiplier = speed;
      }
    });
  }

  function setPanelVisible(visible) {
    els.panel.hidden = !visible;
    els.showPanel.hidden = visible;
  }

  function createViewer() {
    const viewer = new Cesium.Viewer("cesiumContainer", {
      animation: true,
      baseLayerPicker: false,
      fullscreenButton: true,
      geocoder: false,
      homeButton: false,
      imageryProvider: false,
      infoBox: true,
      navigationHelpButton: false,
      sceneModePicker: false,
      selectionIndicator: true,
      shouldAnimate: true,
      timeline: true,
      terrainProvider: new Cesium.EllipsoidTerrainProvider(),
    });

    viewer.scene.backgroundColor = Cesium.Color.fromCssColorString("#050914");
    viewer.scene.globe.baseColor = Cesium.Color.fromCssColorString("#0a172c");
    viewer.scene.globe.depthTestAgainstTerrain = true;
    viewer.scene.globe.enableLighting = false;
    if (viewer.scene.globe.translucency) {
      viewer.scene.globe.translucency.enabled = false;
      viewer.scene.globe.translucency.frontFaceAlpha = 1.0;
      viewer.scene.globe.translucency.backFaceAlpha = 1.0;
    }
    viewer.scene.highDynamicRange = false;
    viewer.scene.fog.enabled = false;
    viewer.clock.shouldAnimate = true;
    addEarthImagery(viewer);
    return viewer;
  }

  function addEarthImagery(viewer) {
    try {
      const naturalEarthUrl = Cesium.buildModuleUrl("Assets/Textures/NaturalEarthII");
      const providerOrPromise = Cesium.TileMapServiceImageryProvider.fromUrl(naturalEarthUrl);
      Promise.resolve(providerOrPromise)
        .then(function (provider) {
          viewer.imageryLayers.removeAll();
          viewer.imageryLayers.addImageryProvider(provider);
        })
        .catch(function () {
          addOpenStreetMapImagery(viewer);
        });
    } catch (error) {
      addOpenStreetMapImagery(viewer);
    }
  }

  function addOpenStreetMapImagery(viewer) {
    try {
      viewer.imageryLayers.removeAll();
      viewer.imageryLayers.addImageryProvider(new Cesium.OpenStreetMapImageryProvider({
        url: "https://tile.openstreetmap.org/",
      }));
    } catch (error) {
      viewer.scene.globe.baseColor = Cesium.Color.fromCssColorString("#12345a");
    }
  }

  function populateConstellations(constellations) {
    els.select.innerHTML = "";
    constellations.forEach(function (config) {
      const option = document.createElement("option");
      option.value = config.id;
      option.textContent = config.label || config.name;
      els.select.appendChild(option);
    });
  }

  function populateRouteSelects(groundStations) {
    [els.routeSource, els.routeTarget].forEach(function (select) {
      while (select.options.length > 1) {
        select.remove(1);
      }
      groundStations.forEach(function (station) {
        const option = document.createElement("option");
        option.value = station.name;
        option.textContent = station.name;
        select.appendChild(option);
      });
    });
  }

  function applyDefaultClockSpeed() {
    const speed = Number(state.metadata.defaults?.clockMultiplier || els.speedSlider.value || 120);
    els.speedSlider.value = String(speed);
    els.speedLabel.textContent = `${speed}x`;
    state.viewer.clock.multiplier = speed;
  }

  async function reloadActiveConstellation() {
    if (!state.activeConfig || state.loading) {
      return;
    }
    renderConstellation(state.activeConfig, state.activeRecords);
  }

  async function loadSelectedConstellation() {
    if (state.loading) {
      return;
    }

    const config = findSelectedConfig();
    if (!config) {
      setStatus("No constellation selected.", true);
      return;
    }

    state.loading = true;
    setStatus(`Loading ${config.name} TLE data...`);
    updateDensityControl(config);

    try {
      const parsed = await loadRecords(config);
      state.activeConfig = Object.assign({}, config, parsed.header);
      state.activeRecords = parsed.records;
      state.activeSource = parsed.source;
      renderConstellation(state.activeConfig, state.activeRecords);
      setStatus(`Ready: ${state.activeConfig.name} (${state.activeSource}).`);
    } catch (error) {
      clearScene();
      setStatus(`Failed to load constellation: ${error.message}`, true);
    } finally {
      state.loading = false;
    }
  }

  function findSelectedConfig() {
    const id = els.select.value;
    return (state.metadata.constellations || []).find(function (config) {
      return config.id === id;
    });
  }

  async function loadRecords(config) {
    try {
      const tleText = await fetchTextWithFallback(config.tlePath, config.rawTleUrl);
      const parsed = parseTle(tleText, config);
      parsed.source = "TLE data";
      return parsed;
    } catch (error) {
      const records = createSyntheticRecords(config);
      return {
        header: {
          orbits: Number(config.orbits),
          satsPerOrbit: Number(config.satsPerOrbit),
        },
        records,
        source: "metadata fallback",
      };
    }
  }

  function renderConstellation(config, records) {
    clearScene();
    configureClock(config);

    const sampleStep = getSampleStep(config);
    const groundStations = getGroundStations(config);
    const topology = els.islTopology.value;
    const color = Cesium.Color.fromCssColorString(config.color || "#8cc8ff");
    state.gslCache = createGslCache(config, records, groundStations);
    const renderedSatellites = addSatellites(records, config, sampleStep, color);
    let ringLinks = 0;
    let gridLinks = 0;
    let gslLinks = 0;

    if (topology === "ring" || topology === "grid") {
      ringLinks = addLinks(records, config, sampleStep, "ring", color);
    }
    if (topology === "grid") {
      gridLinks = addLinks(records, config, sampleStep, "grid", color);
    }
    if (els.showGround.checked) {
      addGroundStations(groundStations);
      if (els.showGsl.checked) {
        addGslLinks(groundStations);
        gslLinks = countVisibleGslAttachments();
      }
    }

    updateStats(
      config,
      records.length,
      renderedSatellites,
      ringLinks,
      gridLinks,
      gslLinks,
      sampleStep,
      groundStations.length,
      []
    );
    refreshRouteForMode(records, config, groundStations);
    updateTleLink(config);
    resetCamera(0.8);
  }

  function clearScene() {
    state.viewer.entities.removeAll();
    state.routeCache = null;
    els.stats.innerHTML = "";
  }

  function configureClock(config) {
    const defaults = state.metadata.defaults || {};
    const startIso = config.epochIso || defaults.epochIso || "2000-01-01T00:00:00Z";
    const durationHours = Number(config.durationHours || defaults.durationHours || 6);
    const start = Cesium.JulianDate.fromIso8601(startIso);
    const stop = Cesium.JulianDate.addHours(start, durationHours, new Cesium.JulianDate());

    state.viewer.clock.startTime = Cesium.JulianDate.clone(start);
    state.viewer.clock.stopTime = Cesium.JulianDate.clone(stop);
    state.viewer.clock.currentTime = Cesium.JulianDate.clone(start);
    state.viewer.clock.clockRange = Cesium.ClockRange.LOOP_STOP;
    state.viewer.clock.clockStep = Cesium.ClockStep.SYSTEM_CLOCK_MULTIPLIER;
    state.viewer.clock.shouldAnimate = true;
    state.viewer.timeline.zoomTo(start, stop);
  }

  function getSampleStep(config) {
    if (els.fullDensity.checked) {
      return 1;
    }
    return Math.max(1, Number(config.defaultSampleStep || 1));
  }

  function updateDensityControl(config) {
    const isSampled = Number(config.defaultSampleStep || 1) > 1;
    els.fullDensity.disabled = !isSampled;
    els.fullDensityLabel.classList.toggle("is-disabled", !isSampled);
    els.fullDensity.checked = !isSampled;
  }

  function syncGslControl() {
    els.showGsl.disabled = !els.showGround.checked;
    els.showGslLabel.classList.toggle("is-disabled", !els.showGround.checked);
  }

  function getGroundStations(config) {
    return state.metadata.groundStations || config.groundStations || [];
  }

  function addSatellites(records, config, sampleStep, color) {
    let rendered = 0;
    records.forEach(function (record) {
      if (record.index % sampleStep !== 0) {
        return;
      }

      state.viewer.entities.add({
        id: `sat-${config.id}-${record.index}`,
        name: record.name,
        description: satelliteDescription(record, config),
        position: new Cesium.CallbackProperty(function (time) {
          return positionForRecord(record, time);
        }, false),
        point: {
          color: color.withAlpha(0.92),
          outlineColor: Cesium.Color.BLACK.withAlpha(0.7),
          outlineWidth: 1,
          pixelSize: config.pointSize || 4,
        },
      });
      rendered += 1;
    });
    return rendered;
  }

  function addLinks(records, config, sampleStep, mode, color) {
    const orbits = Number(config.orbits);
    const satsPerOrbit = Number(config.satsPerOrbit);
    if (!orbits || !satsPerOrbit) {
      return 0;
    }

    const maxLinks = Number(config.maxLinks || 2000);
    const material = color.withAlpha(mode === "grid" ? 0.34 : 0.42);
    const width = mode === "grid" ? 0.7 : 0.9;
    let count = 0;

    for (let plane = 0; plane < orbits; plane += 1) {
      for (let slot = 0; slot < satsPerOrbit; slot += 1) {
        const sourceIndex = plane * satsPerOrbit + slot;
        if (sourceIndex % sampleStep !== 0) {
          continue;
        }
        if (count >= maxLinks) {
          return count;
        }

        const targetPlane = mode === "grid" ? (plane + 1) % orbits : plane;
        const targetSlot = mode === "grid" ? slot : (slot + 1) % satsPerOrbit;
        const source = records[sourceIndex];
        const target = records[targetPlane * satsPerOrbit + targetSlot];
        if (!source || !target) {
          continue;
        }

        state.viewer.entities.add({
          name: `${mode}-isl-${source.index}-${target.index}`,
          polyline: {
            positions: new Cesium.CallbackProperty(function (time) {
              const sourcePosition = positionForRecord(source, time);
              const targetPosition = positionForRecord(target, time);
              return sourcePosition && targetPosition ? [sourcePosition, targetPosition] : [];
            }, false),
            width,
            arcType: Cesium.ArcType.NONE,
            material,
          },
        });
        count += 1;
      }
    }
    return count;
  }

  function addGroundStations(groundStations) {
    groundStations.forEach(function (station) {
      state.viewer.entities.add({
        id: `gs-${station.name}`,
        name: station.name,
        description: `<p>Ground station at ${station.latitude}, ${station.longitude}</p>`,
        position: Cesium.Cartesian3.fromDegrees(
          Number(station.longitude),
          Number(station.latitude),
          Number(station.elevationM || 0)
        ),
        point: {
          color: Cesium.Color.CYAN,
          outlineColor: Cesium.Color.BLACK,
          outlineWidth: 2,
          pixelSize: 10,
        },
        label: {
          text: station.name,
          font: "13px sans-serif",
          fillColor: Cesium.Color.WHITE,
          outlineColor: Cesium.Color.BLACK,
          outlineWidth: 2,
          pixelOffset: new Cesium.Cartesian2(0, -18),
          style: Cesium.LabelStyle.FILL_AND_OUTLINE,
        },
      });
    });
  }

  function addGslLinks(groundStations) {
    const material = Cesium.Color.CYAN.withAlpha(0.62);
    groundStations.forEach(function (station, index) {
      state.viewer.entities.add({
        id: `gsl-${index}`,
        name: `GSL ${station.name}`,
        polyline: {
          positions: new Cesium.CallbackProperty(function (time) {
            const attachment = getGslAttachment(index, time);
            if (!attachment) {
              return [];
            }
            const satellitePosition = positionForRecord(attachment.satellite, time);
            return satellitePosition ? [attachment.groundPosition, satellitePosition] : [];
          }, false),
          width: 1.8,
          arcType: Cesium.ArcType.NONE,
          material,
        },
      });
    });
  }

  function countVisibleGslAttachments() {
    return calculateGslAttachments(state.viewer.clock.currentTime).filter(Boolean).length;
  }

  function MinHeap() {
    const nodes = [];
    const keys = [];
    return {
      size: function () { return nodes.length; },
      push: function (node, key) {
        nodes.push(node);
        keys.push(key);
        let i = nodes.length - 1;
        while (i > 0) {
          const parent = (i - 1) >> 1;
          if (keys[parent] <= keys[i]) {
            break;
          }
          [keys[parent], keys[i]] = [keys[i], keys[parent]];
          [nodes[parent], nodes[i]] = [nodes[i], nodes[parent]];
          i = parent;
        }
      },
      pop: function () {
        const top = nodes[0];
        const lastNode = nodes.pop();
        const lastKey = keys.pop();
        if (nodes.length > 0) {
          nodes[0] = lastNode;
          keys[0] = lastKey;
          let i = 0;
          const length = nodes.length;
          for (;;) {
            const left = 2 * i + 1;
            const right = 2 * i + 2;
            let smallest = i;
            if (left < length && keys[left] < keys[smallest]) { smallest = left; }
            if (right < length && keys[right] < keys[smallest]) { smallest = right; }
            if (smallest === i) {
              break;
            }
            [keys[smallest], keys[i]] = [keys[i], keys[smallest]];
            [nodes[smallest], nodes[i]] = [nodes[i], nodes[smallest]];
            i = smallest;
          }
        }
        return top;
      },
    };
  }

  const ROUTE_IDS = ["topological-route", "linkstate-route"];

  function clearRouteEntities() {
    ROUTE_IDS.forEach(function (id) { state.viewer.entities.removeById(id); });
    state.routeCache = null;
  }

  function updateRoute() {
    if (!state.viewer || !state.activeConfig || state.loading) {
      return;
    }
    clearRouteEntities();
    refreshRouteForMode(state.activeRecords, state.activeConfig, getGroundStations(state.activeConfig));
  }

  function refreshRouteForMode(records, config, groundStations) {
    renderStatsTable(addRoute(records, config, groundStations));
  }

  function addRoute(records, config, groundStations) {
    clearRouteEntities();

    const srcName = els.routeSource.value;
    const dstName = els.routeTarget.value;
    if (!srcName || !dstName || srcName === dstName) {
      return [];
    }

    const orbits = Number(config.orbits);
    const satsPerOrbit = Number(config.satsPerOrbit);
    if (!orbits || !satsPerOrbit) {
      return [];
    }

    const topology = els.islTopology.value;
    if (topology !== "ring" && topology !== "grid") {
      return [];
    }

    const srcGsIndex = groundStations.findIndex(function (g) { return g.name === srcName; });
    const dstGsIndex = groundStations.findIndex(function (g) { return g.name === dstName; });
    if (srcGsIndex < 0 || dstGsIndex < 0) {
      return [];
    }

    const showLinkState = els.showLinkStateRoute.checked;
    state.routeCache = { key: null, chain: null };

    function refreshChain(time) {
      const key = Math.floor(Cesium.JulianDate.toDate(time).getTime() / 60000);
      if (state.routeCache.key === key) {
        return state.routeCache.chain;
      }
      state.routeCache.key = key;
      state.routeCache.chain = computeRouteChain(
        srcGsIndex, dstGsIndex, records, orbits, satsPerOrbit, topology, showLinkState, time
      );
      return state.routeCache.chain;
    }

    function routePositions(time, kind, lift) {
      const chain = refreshChain(time);
      if (!chain || !chain[kind]) {
        return [];
      }
      const raw = [chain.srcGround];
      chain[kind].forEach(function (record) {
        const position = positionForRecord(record, time);
        if (position) {
          raw.push(position);
        }
      });
      raw.push(chain.dstGround);
      if (raw.length < 2) {
        return [];
      }
      const lifted = (!lift || lift === 1)
        ? raw
        : raw.map(function (p) {
            // Radial lift to separate overlapping lines without z-fighting (tiny).
            return Cesium.Cartesian3.multiplyByScalar(p, lift, new Cesium.Cartesian3());
          });
      return cullOccludedPoints(lifted);
    }

    // Both lines opaque so the globe (depthTestAgainstTerrain) occludes them and
    // they don't bleed through the far side. Link-state is lifted a hair radially
    // so that, when it coincides with topological (stretch = 1), it shows as a
    // green rim hugging the yellow line instead of z-fighting it.
    if (showLinkState) {
      state.viewer.entities.add({
        id: "linkstate-route",
        name: `Shortest-path route (link-state): ${srcName} → ${dstName}`,
        polyline: {
          positions: new Cesium.CallbackProperty(function (time) {
            return routePositions(time, "ls", 1.0);
          }, false),
          width: 7,
          arcType: Cesium.ArcType.NONE,
          material: Cesium.Color.fromCssColorString("#2bff88").withAlpha(1.0),
        },
      });
    }

    state.viewer.entities.add({
      id: "topological-route",
      name: `Topological forwarding: ${srcName} → ${dstName}`,
      polyline: {
        positions: new Cesium.CallbackProperty(function (time) {
          return routePositions(time, "topo", 1.0010);
        }, false),
        width: 3,
        arcType: Cesium.ArcType.NONE,
        material: Cesium.Color.fromCssColorString("#ffd400").withAlpha(1.0),
      },
    });

    const initialChain = refreshChain(state.viewer.clock.currentTime);
    if (!initialChain) {
      return [["Route", "unreachable — try +Grid"]];
    }
    return routeStatRows(initialChain, state.viewer.clock.currentTime);
  }

  function routeStatRows(chain, time) {
    if (!chain) {
      return [];
    }
    const rows = [
      ["Topo route hops", (chain.topo.length - 1).toLocaleString()],
      ["Topo path", `${Math.round(pathPhysicalKm(chain.topo, time)).toLocaleString()} km`],
    ];
    if (chain.ls) {
      rows.push(["LS route hops", (chain.ls.length - 1).toLocaleString()]);
      rows.push(["LS path", `${Math.round(pathPhysicalKm(chain.ls, time)).toLocaleString()} km`]);
    }
    return rows;
  }

  function pathPhysicalKm(records, time) {
    let total = 0;
    for (let i = 0; i < records.length - 1; i += 1) {
      const a = positionForRecord(records[i], time);
      const b = positionForRecord(records[i + 1], time);
      if (a && b) {
        total += Cesium.Cartesian3.distance(a, b);
      }
    }
    return total / 1000;
  }

  function computeRouteChain(srcGsIndex, dstGsIndex, records, orbits, satsPerOrbit, topology, showLinkState, time) {
    const srcAttachment = getGslAttachment(srcGsIndex, time);
    if (!srcAttachment || !state.gslCache) {
      return null;
    }
    // Multi-egress: the destination GS is reachable via ANY satellite currently
    // overhead, so delivery happens at the first/nearest visible egress, matching
    // the real algorithm (and why topological forwarding ties shortest-path).
    const dstVisible = visibleSatSet(dstGsIndex, records, time);
    if (dstVisible.size === 0) {
      return null;
    }

    const topo = pivotWeightedPath(
      srcAttachment.satellite, dstVisible, records, orbits, satsPerOrbit, topology, time
    );
    if (!topo) {
      return null;
    }

    let ls = null;
    if (showLinkState) {
      ls = linkStatePath(
        srcAttachment.satellite, dstVisible, records, orbits, satsPerOrbit, topology, time
      );
    }

    return {
      srcGround: state.gslCache.groundStations[srcGsIndex].groundPosition,
      dstGround: state.gslCache.groundStations[dstGsIndex].groundPosition,
      topo,
      ls,
    };
  }

  function visibleSatSet(gsIndex, records, time) {
    const set = new Set();
    if (!state.gslCache) {
      return set;
    }
    const station = state.gslCache.groundStations[gsIndex];
    const maxDistanceM = maxGslDistanceM(state.activeConfig);
    for (let i = 0; i < records.length; i += 1) {
      const position = positionForRecord(records[i], time);
      if (!position) {
        continue;
      }
      const vector = Cesium.Cartesian3.subtract(position, station.groundPosition, new Cesium.Cartesian3());
      if (Cesium.Cartesian3.dot(vector, station.normal) <= 0) {
        continue;
      }
      if (Cesium.Cartesian3.magnitude(vector) <= maxDistanceM) {
        set.add(records[i].index);
      }
    }
    return set;
  }

  function cullOccludedPoints(points) {
    // Keep only the longest contiguous run of points visible from the camera, so
    // the route line stops at the Earth's limb instead of drawing through the
    // globe (Cesium does not reliably occlude space polylines behind the globe).
    if (points.length < 2) {
      return [];
    }
    const occluder = new Cesium.EllipsoidalOccluder(Cesium.Ellipsoid.WGS84, state.viewer.camera.positionWC);
    let bestStart = 0;
    let bestLen = 0;
    let curStart = 0;
    let curLen = 0;
    for (let i = 0; i < points.length; i += 1) {
      if (occluder.isPointVisible(points[i])) {
        if (curLen === 0) {
          curStart = i;
        }
        curLen += 1;
        if (curLen > bestLen) {
          bestLen = curLen;
          bestStart = curStart;
        }
      } else {
        curLen = 0;
      }
    }
    return bestLen >= 2 ? points.slice(bestStart, bestStart + bestLen) : [];
  }

  function neighborIndices(index, orbits, satsPerOrbit, topology) {
    const plane = Math.floor(index / satsPerOrbit);
    const slot = index % satsPerOrbit;
    const neighbors = [
      plane * satsPerOrbit + (slot + 1) % satsPerOrbit,
      plane * satsPerOrbit + (slot - 1 + satsPerOrbit) % satsPerOrbit,
    ];
    if (topology === "grid") {
      neighbors.push(((plane + 1) % orbits) * satsPerOrbit + slot);
      neighbors.push(((plane - 1 + orbits) % orbits) * satsPerOrbit + slot);
    }
    return neighbors;
  }

  function linkStatePath(srcRecord, dstSet, records, orbits, satsPerOrbit, topology, time) {
    const n = records.length;
    const srcIndex = srcRecord.plane * satsPerOrbit + srcRecord.slot;

    const positions = new Array(n);
    for (let i = 0; i < n; i += 1) {
      positions[i] = positionForRecord(records[i], time) || null;
    }
    if (!positions[srcIndex]) {
      return null;
    }

    const dist = new Float64Array(n).fill(Infinity);
    const prev = new Int32Array(n).fill(-1);
    const visited = new Uint8Array(n);
    dist[srcIndex] = 0;

    const heap = new MinHeap();
    heap.push(srcIndex, 0);
    let reached = -1;

    while (heap.size() > 0) {
      const u = heap.pop();
      if (visited[u]) {
        continue;
      }
      visited[u] = 1;
      if (dstSet.has(u)) {
        reached = u;
        break;
      }
      const pu = positions[u];
      if (!pu) {
        continue;
      }
      const neighbors = neighborIndices(u, orbits, satsPerOrbit, topology);
      for (let k = 0; k < neighbors.length; k += 1) {
        const v = neighbors[k];
        const pv = positions[v];
        if (!pv || visited[v]) {
          continue;
        }
        const candidate = dist[u] + Cesium.Cartesian3.distance(pu, pv);
        if (candidate < dist[v]) {
          dist[v] = candidate;
          prev[v] = u;
          heap.push(v, candidate);
        }
      }
    }

    if (reached === -1) {
      return null;
    }

    const path = [];
    let cursor = reached;
    while (cursor !== -1) {
      path.push(records[cursor]);
      cursor = prev[cursor];
    }
    path.reverse();
    return path;
  }

  // Port of leopath/network_state/routing_algorithms/topological_routing/
  // fstate_calculation.py's torus_weighted_pivot distance + greedy next-hop
  // selection (the actual algorithm reported in the paper), so the demo route
  // matches the real per-satellite forwarding decisions instead of a plain
  // hop-count heuristic.
  function buildPivotWeightModel(records, orbits, satsPerOrbit, topology, positions) {
    const rowEdgeCosts = [];
    for (let p = 0; p < orbits; p += 1) {
      rowEdgeCosts.push(new Array(satsPerOrbit).fill(Infinity));
    }
    const planeEdgeCosts = [];
    for (let s = 0; s < satsPerOrbit; s += 1) {
      planeEdgeCosts.push(new Array(orbits).fill(Infinity));
    }

    for (let plane = 0; plane < orbits; plane += 1) {
      for (let slot = 0; slot < satsPerOrbit; slot += 1) {
        const index = plane * satsPerOrbit + slot;
        const pos = positions[index];
        if (!pos) {
          continue;
        }
        const nextSlot = (slot + 1) % satsPerOrbit;
        const rowNeighborIndex = plane * satsPerOrbit + nextSlot;
        const rowNeighborPos = positions[rowNeighborIndex];
        if (rowNeighborPos) {
          const weight = Cesium.Cartesian3.distance(pos, rowNeighborPos);
          rowEdgeCosts[plane][slot] = Math.min(rowEdgeCosts[plane][slot], weight);
        }
        if (topology === "grid") {
          const nextPlane = (plane + 1) % orbits;
          const planeNeighborIndex = nextPlane * satsPerOrbit + slot;
          const planeNeighborPos = positions[planeNeighborIndex];
          if (planeNeighborPos) {
            const weight = Cesium.Cartesian3.distance(pos, planeNeighborPos);
            planeEdgeCosts[slot][plane] = Math.min(planeEdgeCosts[slot][plane], weight);
          }
        }
      }
    }

    function pathCost(edgeCosts, start, end) {
      const modulus = edgeCosts.length;
      if (start === end) {
        return 0.0;
      }
      const forwardSteps = ((end - start) % modulus + modulus) % modulus;
      const backwardSteps = ((start - end) % modulus + modulus) % modulus;
      let forwardCost = 0.0;
      for (let step = 0; step < forwardSteps; step += 1) {
        const cost = edgeCosts[(start + step) % modulus];
        if (!Number.isFinite(cost)) {
          forwardCost = Infinity;
          break;
        }
        forwardCost += cost;
      }
      let backwardCost = 0.0;
      for (let step = 0; step < backwardSteps; step += 1) {
        const cost = edgeCosts[(start - 1 - step + modulus * 2) % modulus];
        if (!Number.isFinite(cost)) {
          backwardCost = Infinity;
          break;
        }
        backwardCost += cost;
      }
      return Math.min(forwardCost, backwardCost);
    }

    const rowPathCosts = [];
    for (let plane = 0; plane < orbits; plane += 1) {
      const rows = [];
      for (let src = 0; src < satsPerOrbit; src += 1) {
        const row = [];
        for (let dst = 0; dst < satsPerOrbit; dst += 1) {
          row.push(pathCost(rowEdgeCosts[plane], src, dst));
        }
        rows.push(row);
      }
      rowPathCosts.push(rows);
    }

    const planePathCosts = [];
    for (let slot = 0; slot < satsPerOrbit; slot += 1) {
      const rows = [];
      for (let src = 0; src < orbits; src += 1) {
        const row = [];
        for (let dst = 0; dst < orbits; dst += 1) {
          row.push(pathCost(planeEdgeCosts[slot], src, dst));
        }
        rows.push(row);
      }
      planePathCosts.push(rows);
    }

    return { orbits, satsPerOrbit, rowPathCosts, planePathCosts };
  }

  function pivotDistance(weightModel, srcPlane, srcSlot, dstPlane, dstSlot) {
    if (srcPlane === dstPlane && srcSlot === dstSlot) {
      return 0.0;
    }
    const { satsPerOrbit, rowPathCosts, planePathCosts } = weightModel;
    let best = Infinity;
    for (let pivotRow = 0; pivotRow < satsPerOrbit; pivotRow += 1) {
      const sourceRowCost = rowPathCosts[srcPlane][srcSlot][pivotRow];
      const planeCost = planePathCosts[pivotRow][srcPlane][dstPlane];
      const destinationRowCost = rowPathCosts[dstPlane][pivotRow][dstSlot];
      const total = sourceRowCost + planeCost + destinationRowCost;
      if (total < best) {
        best = total;
      }
    }
    return best;
  }

  function tieBreakTuple(srcPlane, srcSlot, dstPlane, dstSlot, orbits, satsPerOrbit) {
    const planeForward = ((dstPlane - srcPlane) % orbits + orbits) % orbits;
    const satForward = ((dstSlot - srcSlot) % satsPerOrbit + satsPerOrbit) % satsPerOrbit;
    const samePlanePriority = srcPlane === dstPlane ? 0 : 1;
    return [samePlanePriority, satForward, planeForward];
  }

  function tieBreakLess(a, b) {
    for (let i = 0; i < a.length; i += 1) {
      if (a[i] !== b[i]) {
        return a[i] < b[i];
      }
    }
    return false;
  }

  function pivotWeightedPath(srcRecord, dstSet, records, orbits, satsPerOrbit, topology, time) {
    const n = records.length;
    const positions = new Array(n);
    for (let i = 0; i < n; i += 1) {
      positions[i] = positionForRecord(records[i], time) || null;
    }
    if (!positions[srcRecord.plane * satsPerOrbit + srcRecord.slot]) {
      return null;
    }

    const weightModel = buildPivotWeightModel(records, orbits, satsPerOrbit, topology, positions);

    // Multi-egress: pick the single destination satellite (among those
    // currently visible to the destination GS) that minimizes the pivot
    // distance estimate from the source, mirroring the real algorithm's
    // per-source target selection.
    let targetPlane = null;
    let targetSlot = null;
    let bestTargetDistance = Infinity;
    dstSet.forEach(function (index) {
      const plane = Math.floor(index / satsPerOrbit);
      const slot = index % satsPerOrbit;
      const distance = pivotDistance(weightModel, srcRecord.plane, srcRecord.slot, plane, slot);
      if (distance < bestTargetDistance) {
        bestTargetDistance = distance;
        targetPlane = plane;
        targetSlot = slot;
      }
    });
    if (targetPlane === null) {
      return null;
    }

    const maxHops = orbits + satsPerOrbit + 5;
    const path = [srcRecord];
    let current = srcRecord;
    let guard = 0;

    const inEgress = function (record) {
      return dstSet.has(record.plane * satsPerOrbit + record.slot);
    };

    while (!inEgress(current) && guard < maxHops) {
      guard += 1;
      const candidates = [
        [current.plane, (current.slot + 1) % satsPerOrbit],
        [current.plane, (current.slot - 1 + satsPerOrbit) % satsPerOrbit],
      ];
      if (topology === "grid") {
        candidates.push([(current.plane + 1) % orbits, current.slot]);
        candidates.push([(current.plane - 1 + orbits) % orbits, current.slot]);
      }

      const currentPos = positions[current.plane * satsPerOrbit + current.slot];
      let bestCandidate = null;
      let bestScore = Infinity;
      let bestTie = null;

      candidates.forEach(function (candidate) {
        const [candPlane, candSlot] = candidate;
        const candIndex = candPlane * satsPerOrbit + candSlot;
        const candPos = positions[candIndex];
        if (!candPos) {
          return;
        }
        const edgeWeight = Cesium.Cartesian3.distance(currentPos, candPos);
        const distanceToTarget = pivotDistance(weightModel, candPlane, candSlot, targetPlane, targetSlot);
        // torus_weighted_pivot scoring: real ISL edge cost + pivot estimate
        // to target (matches _neighbor_candidate_score's non-unit branch).
        const score = edgeWeight + distanceToTarget;
        const tie = tieBreakTuple(candPlane, candSlot, targetPlane, targetSlot, orbits, satsPerOrbit);

        if (score < bestScore || (score === bestScore && (!bestTie || tieBreakLess(tie, bestTie)))) {
          bestScore = score;
          bestCandidate = candidate;
          bestTie = tie;
        }
      });

      if (!bestCandidate) {
        break;
      }
      const next = records[bestCandidate[0] * satsPerOrbit + bestCandidate[1]];
      if (!next) {
        break;
      }
      path.push(next);
      current = next;
    }

    // Only a path that actually reaches a destination egress is valid. In Ring
    // (intra-plane only) cross-plane pairs are genuinely unreachable -> no route.
    return inEgress(current) ? path : null;
  }

  function createGslCache(config, records, groundStations) {
    return {
      config,
      records,
      groundStations: groundStations.map(prepareGroundStation),
      key: null,
      attachments: [],
    };
  }

  function prepareGroundStation(station) {
    const groundPosition = Cesium.Cartesian3.fromDegrees(
      Number(station.longitude),
      Number(station.latitude),
      Number(station.elevationM || 0)
    );
    const normal = Cesium.Cartesian3.normalize(groundPosition, new Cesium.Cartesian3());
    return Object.assign({}, station, { groundPosition, normal });
  }

  function getGslAttachment(groundStationIndex, time) {
    if (!state.gslCache) {
      return null;
    }

    const key = Math.floor(Cesium.JulianDate.toDate(time).getTime() / 60000);
    if (state.gslCache.key !== key) {
      state.gslCache.key = key;
      state.gslCache.attachments = calculateGslAttachments(time);
    }
    return state.gslCache.attachments[groundStationIndex] || null;
  }

  function calculateGslAttachments(time) {
    const cache = state.gslCache;
    const maxDistanceM = maxGslDistanceM(cache.config);
    return cache.groundStations.map(function (station) {
      let best = null;
      let bestDistance = Number.POSITIVE_INFINITY;

      cache.records.forEach(function (record) {
        const satellitePosition = positionForRecord(record, time);
        if (!satellitePosition) {
          return;
        }

        const vector = Cesium.Cartesian3.subtract(
          satellitePosition,
          station.groundPosition,
          new Cesium.Cartesian3()
        );
        if (Cesium.Cartesian3.dot(vector, station.normal) <= 0) {
          return;
        }

        const distance = Cesium.Cartesian3.magnitude(vector);
        if (distance <= maxDistanceM && distance < bestDistance) {
          bestDistance = distance;
          best = {
            groundPosition: station.groundPosition,
            satellite: record,
            satelliteId: record.index,
            distance,
          };
        }
      });
      return best;
    });
  }

  function maxGslDistanceM(config) {
    const defaults = state.metadata.defaults || {};
    const altitudeM = Number(config.altitudeKm || 550) * 1000;
    const coneAngleDeg = Number(config.coneAngleDeg || defaults.coneAngleDeg || 25);
    const coneRadiusM = altitudeM / Math.tan(Cesium.Math.toRadians(coneAngleDeg));
    return Math.sqrt(coneRadiusM * coneRadiusM + altitudeM * altitudeM);
  }

  function positionForRecord(record, time) {
    if (record.synthetic) {
      return syntheticPositionForRecord(record, time);
    }

    const date = Cesium.JulianDate.toDate(time);
    const propagated = satellite.propagate(record.satrec, date);
    if (!propagated || !propagated.position) {
      return undefined;
    }

    const gmst = satellite.gstime(date);
    const geodetic = satellite.eciToGeodetic(propagated.position, gmst);
    return Cesium.Cartesian3.fromRadians(
      geodetic.longitude,
      geodetic.latitude,
      geodetic.height * 1000
    );
  }

  function syntheticPositionForRecord(record, time) {
    const date = Cesium.JulianDate.toDate(time);
    const epochMs = Date.parse(record.epochIso || "2000-01-01T00:00:00Z");
    const elapsedSeconds = (date.getTime() - epochMs) / 1000;
    const angle = record.meanAnomalyRad + (2 * Math.PI * record.meanMotionRevPerDay * elapsedSeconds / 86400);
    const xOrbital = record.radiusKm * Math.cos(angle);
    const yOrbital = record.radiusKm * Math.sin(angle);
    const cosRaan = Math.cos(record.raanRad);
    const sinRaan = Math.sin(record.raanRad);
    const cosInclination = Math.cos(record.inclinationRad);
    const sinInclination = Math.sin(record.inclinationRad);
    const eci = {
      x: cosRaan * xOrbital - sinRaan * cosInclination * yOrbital,
      y: sinRaan * xOrbital + cosRaan * cosInclination * yOrbital,
      z: sinInclination * yOrbital,
    };
    const ecf = satellite.eciToEcf(eci, satellite.gstime(date));
    return Cesium.Cartesian3.fromElements(ecf.x * 1000, ecf.y * 1000, ecf.z * 1000);
  }

  async function fetchJson(url) {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`${url} returned ${response.status}`);
    }
    return response.json();
  }

  async function fetchTextWithFallback(primaryUrl, fallbackUrl) {
    const urls = [primaryUrl, fallbackUrl].filter(Boolean);
    let lastError = null;

    for (const [index, url] of urls.entries()) {
      try {
        const response = await fetch(url, { cache: "no-store" });
        if (!response.ok) {
          throw new Error(`${url} returned ${response.status}`);
        }
        return response.text();
      } catch (error) {
        lastError = error;
        if (index === 0 && isLocalPreview() && String(primaryUrl).startsWith("data/")) {
          break;
        }
      }
    }
    throw lastError || new Error("No TLE URL configured");
  }

  function isLocalPreview() {
    return ["localhost", "127.0.0.1", "0.0.0.0"].includes(window.location.hostname);
  }

  function parseTle(tleText, config) {
    const lines = tleText.split(/\r?\n/).map(function (line) {
      return line.trim();
    }).filter(Boolean);

    if (lines.length < 3) {
      throw new Error("TLE file is empty or malformed");
    }

    let cursor = 0;
    const headerMatch = lines[0].match(/^(\d+)\s+(\d+)$/);
    const header = {};
    if (headerMatch) {
      header.orbits = Number(headerMatch[1]);
      header.satsPerOrbit = Number(headerMatch[2]);
      cursor = 1;
    }

    const satsPerOrbit = Number(config.satsPerOrbit || header.satsPerOrbit || 1);
    const records = [];
    while (cursor + 2 < lines.length) {
      const name = lines[cursor];
      const line1 = lines[cursor + 1];
      const line2 = lines[cursor + 2];
      cursor += 3;

      if (!line1.startsWith("1 ") || !line2.startsWith("2 ")) {
        continue;
      }

      const index = records.length;
      records.push({
        index,
        name,
        line1,
        line2,
        plane: Math.floor(index / satsPerOrbit),
        slot: index % satsPerOrbit,
        satrec: satellite.twoline2satrec(line1, line2),
      });
    }

    if (records.length === 0) {
      throw new Error("No valid TLE records found");
    }

    if (!header.orbits && config.orbits) {
      header.orbits = config.orbits;
    }
    if (!header.satsPerOrbit && config.satsPerOrbit) {
      header.satsPerOrbit = config.satsPerOrbit;
    }

    return { header, records };
  }

  function createSyntheticRecords(config) {
    const orbits = Number(config.orbits);
    const satsPerOrbit = Number(config.satsPerOrbit);
    if (!orbits || !satsPerOrbit) {
      throw new Error("No TLE data and insufficient metadata for fallback propagation");
    }

    const defaults = state.metadata.defaults || {};
    const phaseDiff = config.phaseDiff !== false;
    const inclinationRad = Cesium.Math.toRadians(Number(config.inclinationDeg || 0));
    const meanMotionRevPerDay = Number(config.meanMotionRevPerDay || 15);
    const radiusKm = Number(config.altitudeKm || 550) + Number(config.earthRadiusKm || defaults.earthRadiusKm || 6378.135);
    const records = [];

    for (let plane = 0; plane < orbits; plane += 1) {
      const raanRad = 2 * Math.PI * plane / orbits;
      const planeShift = phaseDiff && plane % 2 === 1 ? Math.PI / satsPerOrbit : 0;
      for (let slot = 0; slot < satsPerOrbit; slot += 1) {
        const index = plane * satsPerOrbit + slot;
        records.push({
          synthetic: true,
          index,
          name: `${config.name} ${index}`,
          plane,
          slot,
          epochIso: config.epochIso || defaults.epochIso,
          inclinationRad,
          meanMotionRevPerDay,
          meanAnomalyRad: planeShift + 2 * Math.PI * slot / satsPerOrbit,
          radiusKm,
          raanRad,
          line1: "metadata fallback",
          line2: "metadata fallback",
        });
      }
    }

    return records;
  }

  function satelliteDescription(record, config) {
    return [
      `<h2>${escapeHtml(record.name)}</h2>`,
      "<table>",
      `<tr><th>Constellation</th><td>${escapeHtml(config.name)}</td></tr>`,
      `<tr><th>Plane</th><td>${record.plane}</td></tr>`,
      `<tr><th>Slot</th><td>${record.slot}</td></tr>`,
      `<tr><th>Satellite ID</th><td>${record.index}</td></tr>`,
      `<tr><th>TLE line 1</th><td><code>${escapeHtml(record.line1)}</code></td></tr>`,
      `<tr><th>TLE line 2</th><td><code>${escapeHtml(record.line2)}</code></td></tr>`,
      "</table>",
    ].join("");
  }

  function updateStats(
    config,
    totalSatellites,
    renderedSatellites,
    ringLinks,
    gridLinks,
    gslLinks,
    sampleStep,
    groundStationCount,
    routeRows
  ) {
    state.baseStatsRows = [
      ["Satellites", totalSatellites.toLocaleString()],
      ["Rendered", renderedSatellites.toLocaleString()],
      ["Orbits", Number(config.orbits).toLocaleString()],
      ["Sats/orbit", Number(config.satsPerOrbit).toLocaleString()],
      ["Altitude", `${Number(config.altitudeKm).toLocaleString()} km`],
      ["Inclination", `${config.inclinationDeg} deg`],
      ["Intra-plane links", ringLinks.toLocaleString()],
      ["Inter-plane links", gridLinks.toLocaleString()],
      ["GSL attachments", gslLinks.toLocaleString()],
      ["Ground stations", groundStationCount.toLocaleString()],
      ["Sample step", sampleStep === 1 ? "full" : `1/${sampleStep}`],
    ];

    renderStatsTable(routeRows);
  }

  function renderStatsTable(routeRows) {
    const stats = (state.baseStatsRows || []).slice();
    if (Array.isArray(routeRows) && routeRows.length > 0) {
      routeRows.forEach(function (row) { stats.push(row); });
    }

    els.stats.innerHTML = [
      "<table>",
      "<tbody>",
      stats.map(function (item) {
        return `<tr><th scope="row">${item[0]}</th><td>${item[1]}</td></tr>`;
      }).join(""),
      "</tbody>",
      "</table>",
    ].join("");
  }

  function updateTleLink(config) {
    els.tleLink.href = state.activeSource === "metadata fallback"
      ? (config.rawTleUrl || config.tlePath || "#")
      : (config.tlePath || config.rawTleUrl || "#");
    els.tleLink.textContent = state.activeSource === "metadata fallback"
      ? "Open source TLE data"
      : "Open TLE data";
  }

  function resetCamera(duration) {
    if (!state.viewer) {
      return;
    }
    state.viewer.camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(12, 18, 24500000),
      orientation: {
        heading: 0,
        pitch: Cesium.Math.toRadians(-90),
        roll: 0,
      },
      duration,
    });
  }

  function setStatus(message, isError) {
    els.status.textContent = message;
    els.status.classList.toggle("status--error", Boolean(isError));
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
}());
