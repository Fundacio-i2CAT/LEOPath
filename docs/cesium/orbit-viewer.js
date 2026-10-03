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
    routeUiTime: 0,
    graph: null,
    failedLinks: new Set(),
    failedSatellites: new Set(),
    selectedSatellite: null,
    lastFailure: null,
    rendering: false,
    failureUiSignature: "",
  };

  const els = {};

  document.addEventListener("DOMContentLoaded", init);

  async function init() {
    bindElements();
    bindEvents();

    if (!window.Cesium || !window.satellite || !window.LEOPathNetwork) {
      setStatus("Viewer libraries failed to load.", true);
      return;
    }

    try {
      state.viewer = createViewer();
      window.leopathLive = {
        get ready() { return Boolean(state.activeConfig && !state.loading); },
        get viewer() { return state.viewer; },
        get route() {
          const chain = state.routeCache?.refresh(state.viewer.clock.currentTime);
          return chain ? {reason:chain.reason, source:chain.source, destination:chain.destination,
            topo:chain.topo?.map(record => record.index) || [],
            ls:chain.ls?.map(record => record.index) || [],
            repair:chain.repair?.map(record => record.index) || [], updatedAt:chain.updatedAt} : null;
        },
        get graph() { return {topology:state.graph.topology, edges:state.graph.edges.map(edge => edge.slice()),
          activeEdges:state.graph.edges.filter(([a,b]) => linkAvailable(a,b)).map(edge => edge.slice())}; },
        get failures() { return {links:[...state.failedLinks], satellites:[...state.failedSatellites]}; },
        failLink, failSatellite, restoreLink, restoreSatellite, restoreAll:restoreFailures,
        caption(message) {
          els.demoCaption.textContent = message;
          els.demoCaption.hidden = !message;
        },
      };
      state.viewer.clock.onTick.addEventListener(syncPlayback);
      state.viewer.selectedEntityChanged.addEventListener(function (entity) {
        if (state.rendering) return;
        const match = String(entity?.id || "").match(/^sat-[a-z0-9]+-(\d+)$/i);
        state.selectedSatellite = match ? Number(match[1]) : null;
        syncFailureUi();
      });
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
    ["livePlay", "restartClock", "clockLabel", "motionStatus", "shellLabel", "replayLink", "demoCaption", "swapRoute", "routePair", "routeDelivery", "liveTopoDelay", "liveLsDelay", "liveTopoHops", "liveLsHops", "liveRouteDetail", "topologyNote", "failRouteLink", "failRouteSatellite", "isolateIngress", "restoreFailures", "allowRepair", "failureCount", "failureStatus", "failureList", "selectedSatelliteLabel", "failSelectedSatellite", "satelliteFailureForm", "satelliteFailureId", "linkFailureForm", "linkFailureA", "linkFailureB", "focusLiveFailure"].forEach(function (id) {
      els[id] = document.getElementById(id);
    });
  }

  function bindEvents() {
    els.livePlay.addEventListener("click", togglePlayback);
    els.restartClock.addEventListener("click", function () {
      state.viewer.clock.currentTime = Cesium.JulianDate.clone(state.viewer.clock.startTime);
      syncPlayback();
    });
    document.addEventListener("keydown", function (event) {
      if (["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(event.target.tagName)) return;
      if (event.code === "Space" && state.activeConfig && !state.loading) {
        event.preventDefault();
        togglePlayback();
      }
    });
    els.select.addEventListener("change", loadSelectedConstellation);
    els.islTopology.addEventListener("change", function () {
      if (!state.activeConfig || state.loading) return;
      updateTopologyOptions(state.activeConfig);
      resetFailureState();
      reloadActiveConstellation();
      setFailureStatus("New wiring selected; previous failures restored.");
    });
    els.failRouteLink.addEventListener("click", failRouteLink);
    els.failRouteSatellite.addEventListener("click", failRouteSatellite);
    els.isolateIngress.addEventListener("click", isolateIngress);
    els.restoreFailures.addEventListener("click", restoreFailures);
    els.allowRepair.addEventListener("change", updateRoute);
    els.focusLiveFailure.addEventListener("click", focusFailure);
    els.failSelectedSatellite.addEventListener("click", () => tryFailure(() => failSatellite(state.selectedSatellite)));
    els.satelliteFailureForm.addEventListener("submit", function (event) {
      event.preventDefault();
      tryFailure(() => failSatellite(Number(els.satelliteFailureId.value)));
    });
    els.linkFailureForm.addEventListener("submit", function (event) {
      event.preventDefault();
      tryFailure(() => failLink(Number(els.linkFailureA.value),Number(els.linkFailureB.value)));
    });
    els.swapRoute.addEventListener("click", function () {
      const source = els.routeSource.value;
      els.routeSource.value = els.routeTarget.value;
      els.routeTarget.value = source;
      updateRoute();
    });
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
      els.speedLabel.textContent = `${speed}×`;
      if (state.viewer) {
        state.viewer.clock.multiplier = speed;
        syncPlayback();
      }
    });
  }

  function togglePlayback() {
    if (!state.activeConfig || state.loading) return;
    state.viewer.clock.shouldAnimate = !state.viewer.clock.shouldAnimate;
    syncPlayback();
  }

  function syncPlayback() {
    if (!state.viewer) return;
    const clock = state.viewer.clock;
    const playing = clock.shouldAnimate;
    els.livePlay.textContent = playing ? "Ⅱ Pause" : "▶ Play";
    els.livePlay.setAttribute("aria-label", playing ? "Pause orbital motion" : "Play orbital motion");
    els.livePlay.setAttribute("aria-pressed", String(playing));
    els.motionStatus.textContent = state.loading ? "Loading orbit" : `${playing ? "Playing" : "Paused"} · ${clock.multiplier}×`;
    els.clockLabel.textContent = Cesium.JulianDate.toIso8601(clock.currentTime, 0).replace("T", " · ").replace("Z", "");
    document.body.classList.toggle("paused", !playing);
    if (state.activeConfig && !state.loading) syncRouteUi();
  }

  function setPanelVisible(visible) {
    document.body.classList.toggle("controls-hidden", !visible);
    els.panel.hidden = !visible;
    els.showPanel.hidden = visible;
  }

  function createViewer() {
    const viewer = new Cesium.Viewer("cesiumContainer", {
      animation: false,
      baseLayerPicker: false,
      fullscreenButton: true,
      geocoder: false,
      homeButton: false,
      baseLayer: false,
      infoBox: true,
      navigationHelpButton: false,
      sceneModePicker: false,
      selectionIndicator: true,
      shouldAnimate: true,
      timeline: true,
      terrainProvider: new Cesium.EllipsoidTerrainProvider(),
    });

    viewer.scene.backgroundColor = Cesium.Color.fromCssColorString("#081118");
    viewer.scene.skyBox.show = false;
    viewer.scene.sun.show = false;
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
    els.routeSource.value = "New York";
    els.routeTarget.value = "Perth";
  }

  function applyDefaultClockSpeed() {
    const speed = Number(state.metadata.defaults?.clockMultiplier || els.speedSlider.value || 120);
    els.speedSlider.value = String(speed);
    els.speedLabel.textContent = `${speed}×`;
    state.viewer.clock.multiplier = speed;
  }

  async function reloadActiveConstellation() {
    if (!state.activeConfig || state.loading) {
      return;
    }
    renderConstellation(state.activeConfig, state.activeRecords, false);
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
    els.livePlay.disabled = true;
    els.restartClock.disabled = true;
    setStatus(`Loading ${config.name} TLE data...`);
    updateDensityControl(config);

    try {
      const parsed = await loadRecords(config);
      resetFailureState();
      state.activeConfig = Object.assign({}, config, parsed.header);
      updateTopologyOptions(state.activeConfig);
      state.activeRecords = parsed.records;
      state.activeSource = parsed.source;
      renderConstellation(state.activeConfig, state.activeRecords);
      els.shellLabel.textContent = `${config.orbits} × ${config.satsPerOrbit} · ${config.altitudeKm} km`;
      els.replayLink.href = config.id === "dense" ? "replay.html" : `replay.html?replay=${config.id}-grid`;
      setStatus(`Ready: ${state.activeConfig.name} (${state.activeSource}).`);
      setFailureStatus("Failures persist while satellites move.");
    } catch (error) {
      clearScene();
      state.activeConfig = null;
      state.activeRecords = [];
      setStatus(`Failed to load constellation: ${error.message}`, true);
    } finally {
      state.loading = false;
      els.livePlay.disabled = !state.activeConfig;
      els.restartClock.disabled = !state.activeConfig;
      syncPlayback();
      syncFailureUi();
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
      if (config.requireBundledTle) throw error;
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

  function renderConstellation(config, records, resetClock = true) {
    state.rendering = true;
    clearScene();
    state.graph = LEOPathNetwork.buildGraph(Number(config.orbits),Number(config.satsPerOrbit),els.islTopology.value,Number(config.raanSpreadDeg || 360) >= 360);
    if (resetClock) configureClock(config);

    const sampleStep = getSampleStep(config);
    const groundStations = getGroundStations(config);
    const topology = els.islTopology.value;
    const color = Cesium.Color.fromCssColorString(config.color || "#8cc8ff");
    state.gslCache = createGslCache(config, records, groundStations);
    const renderedSatellites = addSatellites(records, config, sampleStep, color);
    let ringLinks = 0;
    let gridLinks = 0;
    let gslLinks = 0;

    [ringLinks, gridLinks] = addLinks(records, config, sampleStep, color);
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
    if (resetClock) resetCamera(0.8);
    state.rendering = false;
    syncFailureUi();
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
      if (record.index % sampleStep !== 0 && !state.failedSatellites.has(record.index)) {
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
          color: state.failedSatellites.has(record.index) ? Cesium.Color.fromCssColorString("#ff637a") : color.withAlpha(0.92),
          outlineColor: Cesium.Color.BLACK.withAlpha(0.7),
          outlineWidth: state.failedSatellites.has(record.index) ? 2 : 1,
          pixelSize: state.failedSatellites.has(record.index) ? 10 : config.pointSize || 4,
        },
      });
      rendered += 1;
    });
    return rendered;
  }

  function addLinks(records, config, sampleStep, color) {
    const counts = [0,0];
    const maxLinks = Number(config.maxLinks || 2000);
    for (const [a,b] of state.graph.edges) {
      const failed = !linkAvailable(a,b);
      if (a % sampleStep !== 0 && b % sampleStep !== 0 && !failed) continue;
      const kind = Math.floor(a / config.satsPerOrbit) === Math.floor(b / config.satsPerOrbit) ? 0 : 1;
      if (counts[kind] >= maxLinks && !failed) continue;
      const source = records[a], target = records[b];
      state.viewer.entities.add({
        id:`isl-${a}-${b}`, name:`${failed ? "Unavailable" : "Live"} ISL ${a} ↔ ${b}`,
        polyline:{
          positions:new Cesium.CallbackProperty(time => {
            const p=positionForRecord(source,time), q=positionForRecord(target,time);
            return p && q ? [p,q] : [];
          },false), width:failed ? 3 : kind ? .7 : .9, arcType:Cesium.ArcType.NONE,
          material:failed ? new Cesium.PolylineDashMaterialProperty({color:Cesium.Color.fromCssColorString("#ff637a"),dashLength:12}) : color.withAlpha(kind ? .34 : .42)
        }
      });
      counts[kind]++;
    }
    return counts;
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

  const ROUTE_IDS = ["topological-route", "linkstate-route", "repair-route"];
  const ROUTE_REFRESH_SECONDS = 30;

  function clearRouteEntities() {
    ROUTE_IDS.forEach(function (id) { state.viewer.entities.removeById(id); });
    state.routeCache = null;
  }

  function updateRoute() {
    if (!state.viewer || !state.activeConfig || state.loading) return;
    refreshRouteForMode(state.activeRecords, state.activeConfig, getGroundStations(state.activeConfig));
  }

  function refreshRouteForMode(records, config, groundStations) {
    clearRouteEntities();
    const srcName = els.routeSource.value;
    const dstName = els.routeTarget.value;
    els.routePair.textContent = srcName && dstName ? `${srcName} → ${dstName}` : "Choose a flow";
    if (!srcName || !dstName || srcName === dstName || els.islTopology.value === "none") {
      syncRouteUi(true);
      return;
    }
    const srcIndex = groundStations.findIndex(g => g.name === srcName);
    const dstIndex = groundStations.findIndex(g => g.name === dstName);
    if (srcIndex < 0 || dstIndex < 0) return;
    const topology = els.islTopology.value;
    const orbits = Number(config.orbits);
    const satsPerOrbit = Number(config.satsPerOrbit);
    const cache = {key:null, chain:null, refresh};
    state.routeCache = cache;
    function refresh(time) {
      const elapsed = Cesium.JulianDate.secondsDifference(time, state.viewer.clock.startTime);
      const key = Math.floor(elapsed / ROUTE_REFRESH_SECONDS);
      if (cache.key !== key) {
        cache.key = key;
        const snapshotTime = Cesium.JulianDate.addSeconds(state.viewer.clock.startTime, key * ROUTE_REFRESH_SECONDS, new Cesium.JulianDate());
        cache.chain = computeRouteChain(srcIndex, dstIndex, records, orbits, satsPerOrbit, topology, els.showLinkStateRoute.checked, snapshotTime);
        cache.chain.updatedAt = key * ROUTE_REFRESH_SECONDS;
      }
      return cache.chain;
    }
    function positions(time, kind, lift) {
      const chain = refresh(time);
      if (!chain[kind]) return [];
      const raw = [chain.srcGround, ...chain[kind].map(record => positionForRecord(record, time)), chain.dstGround];
      if (raw.some(p => !p)) return [];
      return cullOccludedPoints(raw.map(p => Cesium.Cartesian3.multiplyByScalar(p, lift, new Cesium.Cartesian3())));
    }
    if (els.showLinkStateRoute.checked) {
      state.viewer.entities.add({id:"linkstate-route", name:`Shortest path: ${srcName} → ${dstName}`, polyline:{
        positions:new Cesium.CallbackProperty(time => positions(time, "ls", 1), false),
        width:7, arcType:Cesium.ArcType.NONE, material:Cesium.Color.fromCssColorString("#2bff88")
      }});
    }
    state.viewer.entities.add({id:"topological-route", name:`Topological route: ${srcName} → ${dstName}`, polyline:{
      positions:new Cesium.CallbackProperty(time => positions(time, "topo", 1.001), false),
      width:3, arcType:Cesium.ArcType.NONE, material:Cesium.Color.fromCssColorString("#ffd400")
    }});
    state.viewer.entities.add({id:"repair-route", name:"Live route repair detour", polyline:{
      positions:new Cesium.CallbackProperty(time => {
        const chain = refresh(time);
        if (!chain.repair) return [];
        return cullOccludedPoints(chain.repair.map(record => Cesium.Cartesian3.multiplyByScalar(positionForRecord(record,time),1.0015,new Cesium.Cartesian3())));
      },false), width:4, arcType:Cesium.ArcType.NONE, material:Cesium.Color.fromCssColorString("#ffad63")
    }});
    syncRouteUi(true);
    syncFailureUi();
  }

  function syncRouteUi(force = false) {
    if (!force && performance.now() - state.routeUiTime < 250) return;
    state.routeUiTime = performance.now();
    const time = state.viewer.clock.currentTime;
    const chain = state.routeCache?.refresh(time);
    const reasons = {delivered:"Route available", no_visibility:"No visible attachment", partition:"No path across the live network", blocked:"Topological rule blocked"};
    let message = chain ? reasons[chain.reason] : "Choose two different endpoints";
    if (els.islTopology.value === "none") message = "Choose an ISL topology to show routes";
    if (chain?.repair) message += " · repair detour";
    els.routeDelivery.textContent = message;
    els.routeDelivery.classList.toggle("unavailable", !chain || chain.reason !== "delivered");
    for (const [kind, delayId, hopsId] of [["topo","liveTopoDelay","liveTopoHops"],["ls","liveLsDelay","liveLsHops"]]) {
      const path = chain?.[kind];
      els[delayId].textContent = path ? `${(routePhysicalKm(path, chain, time) / 299.792458).toFixed(1)} ms` : "—";
      els[hopsId].textContent = path ? `${path.length - 1} ISL hops` : kind === "ls" && !els.showLinkStateRoute.checked ? "Reference hidden" : "No route";
    }
    els.liveRouteDetail.textContent = chain?.source !== undefined
      ? `Attachments ${chain.source} → ${chain.destination}. ${chain.repair ? `Repair from satellite ${chain.repair[0].index}. ` : ""}Propagation only; GSLs included.`
      : "Routes refresh with orbital time.";
    syncFailureUi();
  }

  function routePhysicalKm(records, chain, time) {
    const first = positionForRecord(records[0], time);
    const last = positionForRecord(records[records.length-1], time);
    return pathPhysicalKm(records, time) + (Cesium.Cartesian3.distance(chain.srcGround, first) + Cesium.Cartesian3.distance(last, chain.dstGround)) / 1000;
  }

  function pathPhysicalKm(records, time) {
    let total = 0;
    for (let i = 0; i < records.length - 1; i++) {
      total += Cesium.Cartesian3.distance(positionForRecord(records[i], time), positionForRecord(records[i+1], time));
    }
    return total / 1000;
  }

  // Illustrative browser routing over live geometry, with the same fixed
  // nearest-attachment endpoints for both policies. Formal results are in replay.
  function computeRouteChain(srcGsIndex, dstGsIndex, records, orbits, satsPerOrbit, topology, showLinkState, time) {
    const attachments = calculateGslAttachments(time);
    const src = attachments[srcGsIndex];
    const dst = attachments[dstGsIndex];
    if (!src || !dst) return {reason:"no_visibility"};
    const base = {srcGround:src.groundPosition, dstGround:dst.groundPosition,
      source:src.satellite.index, destination:dst.satellite.index, topo:null, ls:null};
    const target = new Set([dst.satellite.index]);
    const shortest = linkStatePath(src.satellite, target, records, orbits, satsPerOrbit, topology, time);
    if (!shortest) return {...base, reason:"partition"};
    const topo = pivotWeightedPath(src.satellite, target, records, orbits, satsPerOrbit, topology, time);
    return {...base, topo, ls:showLinkState ? shortest : null, repair:topo?.repair || null, reason:topo ? "delivered" : "blocked"};
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

  function linkAvailable(a,b) {
    return !state.failedSatellites.has(a) && !state.failedSatellites.has(b) && !state.failedLinks.has(LEOPathNetwork.linkKey(a,b));
  }

  function neighborIndices(index, orbits, satsPerOrbit, topology) {
    return LEOPathNetwork.liveNeighbors(state.graph,index,state.failedSatellites,state.failedLinks);
  }

  function linkStatePath(srcRecord, dstSet, records, orbits, satsPerOrbit, topology, time, excluded = new Set()) {
    const n = records.length;
    const srcIndex = srcRecord.plane * satsPerOrbit + srcRecord.slot;

    const positions = new Array(n);
    for (let i = 0; i < n; i += 1) {
      positions[i] = excluded.has(i) || state.failedSatellites.has(i) ? null : positionForRecord(records[i], time) || null;
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

  // Browser preview of pivot-weighted geometry; exported simulator replay
  // remains the reference for measured forwarding and exception behavior.
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
        if (rowNeighborPos && state.graph.edgeKeys.has(LEOPathNetwork.linkKey(index,rowNeighborIndex))) {
          const weight = Cesium.Cartesian3.distance(pos, rowNeighborPos);
          rowEdgeCosts[plane][slot] = Math.min(rowEdgeCosts[plane][slot], weight);
        }
        if (topology !== "ring" && state.graph.edgeKeys.has(LEOPathNetwork.linkKey(index,((plane+1)%orbits)*satsPerOrbit+slot))) {
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

    function cyclePathCosts(edges) {
      const n = edges.length;
      return Array.from({length:n}, (_, start) => {
        const costs = new Array(n).fill(Infinity);
        costs[start] = 0;
        let forward = 0, backward = 0;
        for (let step = 1; step < n; step++) {
          const next = (start + step) % n;
          const previous = (start - step + n) % n;
          forward += edges[(next - 1 + n) % n];
          backward += edges[previous];
          costs[next] = Math.min(costs[next], forward);
          costs[previous] = Math.min(costs[previous], backward);
        }
        return costs;
      });
    }
    const rowPathCosts = rowEdgeCosts.map(cyclePathCosts);
    const planePathCosts = planeEdgeCosts.map(cyclePathCosts);

    const mean = values => {
      const finite = values.filter(Number.isFinite);
      return finite.length ? finite.reduce((a,b)=>a+b,0)/finite.length : Infinity;
    };
    const brick = topology.startsWith("brick_") ? {rail:mean(rowEdgeCosts.flat()),rung:mean(planeEdgeCosts.map(mean)),planeWrap:state.graph.planeWrap} : null;
    return { orbits, satsPerOrbit, rowPathCosts, planePathCosts, topology, brick };
  }

  function pivotDistance(weightModel, srcPlane, srcSlot, dstPlane, dstSlot) {
    if (srcPlane === dstPlane && srcSlot === dstSlot) {
      return 0.0;
    }
    if (weightModel.brick) return LEOPathNetwork.brickDistance(weightModel,srcPlane,srcSlot,dstPlane,dstSlot);
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

    // Use the fixed nearest destination attachment selected for this snapshot.
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

    const maxHops = n;
    const visited = new Set([srcRecord.index]);
    const path = [srcRecord];
    let current = srcRecord;
    let guard = 0;

    const inEgress = function (record) {
      return dstSet.has(record.plane * satsPerOrbit + record.slot);
    };

    while (!inEgress(current) && guard < maxHops) {
      guard += 1;
      const currentIndex = current.plane * satsPerOrbit + current.slot;
      const currentDistance = pivotDistance(weightModel, current.plane, current.slot, targetPlane, targetSlot);
      const candidates = neighborIndices(currentIndex, orbits, satsPerOrbit, topology).map(index => [Math.floor(index / satsPerOrbit), index % satsPerOrbit]);

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
        if (visited.has(candIndex) || !Number.isFinite(distanceToTarget)) return;
        if (!(distanceToTarget < currentDistance - 1e-6 ||
            Math.abs(distanceToTarget - currentDistance) <= 1e-6 && candIndex < currentIndex)) return;
        const score = edgeWeight + distanceToTarget;
        const tie = tieBreakTuple(candPlane, candSlot, targetPlane, targetSlot, orbits, satsPerOrbit);

        if (score < bestScore || (score === bestScore && (!bestTie || tieBreakLess(tie, bestTie)))) {
          bestScore = score;
          bestCandidate = candidate;
          bestTie = tie;
        }
      });

      if (!bestCandidate) {
        if (els.allowRepair.checked) {
          const excluded = new Set([...visited].filter(index => index !== currentIndex));
          const repair = linkStatePath(current,dstSet,records,orbits,satsPerOrbit,topology,time,excluded);
          if (repair) {
            path.repair = repair;
            path.push(...repair.slice(1));
            return path;
          }
        }
        break;
      }
      const next = records[bestCandidate[0] * satsPerOrbit + bestCandidate[1]];
      if (!next) {
        break;
      }
      path.push(next);
      visited.add(next.index);
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

    const key = Math.floor(Cesium.JulianDate.toDate(time).getTime() / (ROUTE_REFRESH_SECONDS * 1000));
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
        if (state.failedSatellites.has(record.index)) return;
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
    const key = `${time.dayNumber}:${time.secondsOfDay}`;
    if (record.positionTime === key) return record.positionValue;
    record.positionTime = key;
    record.positionValue = propagatePositionForRecord(record, time);
    return record.positionValue;
  }

  function propagatePositionForRecord(record, time) {
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

    if (config.requireBundledTle && (header.orbits !== config.orbits || header.satsPerOrbit !== config.satsPerOrbit || records.length !== config.orbits * config.satsPerOrbit)) {
      throw new Error("Released TLE records do not match the selected shell");
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
      const raanRad = Cesium.Math.toRadians(Number(config.raanSpreadDeg || 360)) * plane / orbits;
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
      ["Drawn intra-plane ISLs", ringLinks.toLocaleString()],
      ["Drawn inter-plane ISLs", gridLinks.toLocaleString()],
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

  function updateTopologyOptions(config) {
    const planeWrap = Number(config.raanSpreadDeg || 360) >= 360;
    for (const option of els.islTopology.options) {
      option.disabled = Boolean(LEOPathNetwork.unsupportedReason(Number(config.orbits),Number(config.satsPerOrbit),option.value,planeWrap));
    }
    if (els.islTopology.selectedOptions[0]?.disabled) els.islTopology.value = "grid";
    const a = LEOPathNetwork.unsupportedReason(Number(config.orbits),Number(config.satsPerOrbit),"brick_a",planeWrap);
    const b = LEOPathNetwork.unsupportedReason(Number(config.orbits),Number(config.satsPerOrbit),"brick_b",planeWrap);
    els.topologyNote.textContent = [a,b].filter(Boolean).join(" ");
  }

  function resetFailureState() {
    state.failedLinks.clear(); state.failedSatellites.clear();
    state.lastFailure = null; state.selectedSatellite = null;
  }

  function validateSatellite(index) {
    if (!Number.isInteger(index) || index < 0 || index >= state.activeRecords.length)
      throw new Error("Choose a satellite ID from the current shell.");
  }

  function setFailureStatus(message,error=false) {
    els.failureStatus.textContent = message;
    els.failureStatus.classList.toggle("error",error);
  }

  function tryFailure(action) {
    try { action(); } catch(error) { setFailureStatus(error.message,true); }
  }

  function applyFailures(message) {
    renderConstellation(state.activeConfig,state.activeRecords,false);
    setFailureStatus(message);
  }

  function failLink(a,b) {
    validateSatellite(a); validateSatellite(b);
    const key = LEOPathNetwork.linkKey(a,b);
    if (!state.graph.edgeKeys.has(key)) throw new Error("Those satellites have no ISL in this topology.");
    if (!linkAvailable(a,b)) throw new Error("That ISL is already unavailable.");
    state.failedLinks.add(key); state.lastFailure = [a,b];
    applyFailures(`ISL ${a} ↔ ${b} is down. Routes recalculated.`);
  }

  function failSatellite(index) {
    validateSatellite(index);
    if (state.failedSatellites.has(index)) throw new Error("That satellite is already offline.");
    state.failedSatellites.add(index); state.lastFailure = [index];
    applyFailures(`Satellite ${index} is offline; its ISLs and GSLs are unavailable.`);
  }

  function refreshLatestFailure() {
    state.lastFailure = state.failedSatellites.size ? [[...state.failedSatellites][0]] : state.failedLinks.size ? [...state.failedLinks][0].split(":").map(Number) : null;
  }

  function restoreLink(a,b) {
    if (state.failedLinks.delete(LEOPathNetwork.linkKey(a,b))) {
      refreshLatestFailure(); applyFailures(`ISL ${a} ↔ ${b} restored.`);
    }
  }

  function restoreSatellite(index) {
    if (state.failedSatellites.delete(index)) {
      refreshLatestFailure(); applyFailures(`Satellite ${index} is online again.`);
    }
  }

  function restoreFailures() {
    resetFailureState();
    applyFailures("All failures restored. Orbital playback is unchanged.");
  }

  function currentRoute() {
    return state.routeCache?.refresh(state.viewer.clock.currentTime);
  }

  function failRouteLink() {
    tryFailure(() => {
      const chain = currentRoute();
      const path = chain?.topo || chain?.ls;
      if (!path || path.length < 2) throw new Error("Choose a reachable route with at least one ISL.");
      const i = Math.min(path.length-2,Math.floor((path.length-1)/2));
      failLink(path[i].index,path[i+1].index);
    });
  }

  function failRouteSatellite() {
    tryFailure(() => {
      const chain = currentRoute();
      const path = chain?.topo || chain?.ls;
      if (!path?.length) throw new Error("Choose a reachable route first.");
      failSatellite(path[Math.floor(path.length/2)].index);
    });
  }

  function isolateIngress() {
    tryFailure(() => {
      const chain = currentRoute();
      const source = chain?.source;
      if (source === undefined) throw new Error("Choose a route with a visible ingress first.");
      const neighbors = neighborIndices(source);
      if (!neighbors.length) throw new Error("The ingress has no live ISLs to cut.");
      for (const next of neighbors) state.failedLinks.add(LEOPathNetwork.linkKey(source,next));
      state.lastFailure = [source];
      applyFailures(`Ingress satellite ${source} isolated; another attachment may take over as the orbit moves.`);
    });
  }

  function syncFailureUi() {
    const links = state.failedLinks.size, nodes = state.failedSatellites.size;
    els.failureCount.textContent = links || nodes ? `${links} ISLs · ${nodes} offline` : "Healthy";
    els.restoreFailures.disabled = !links && !nodes;
    els.focusLiveFailure.disabled = !links && !nodes;
    const route = state.routeCache?.chain;
    const path = route?.topo || route?.ls;
    els.failRouteLink.disabled = !path || path.length < 2;
    els.failRouteSatellite.disabled = !path?.length;
    els.isolateIngress.disabled = route?.source === undefined || !neighborIndices(route.source).length;
    els.failSelectedSatellite.disabled = state.selectedSatellite === null || state.failedSatellites.has(state.selectedSatellite);
    els.selectedSatelliteLabel.textContent = state.selectedSatellite === null ? "Click a satellite on the globe to select it." : `Selected satellite ${state.selectedSatellite}`;
    for (const id of ["satelliteFailureId","linkFailureA","linkFailureB"]) els[id].max = Math.max(0,state.activeRecords.length-1);
    const signature = [...state.failedLinks].join(",") + "|" + [...state.failedSatellites].join(",");
    if (signature === state.failureUiSignature) return;
    state.failureUiSignature = signature;
    const list = document.createDocumentFragment();
    function row(message,restore) {
      const li = document.createElement("li");
      const label = document.createElement("span"); label.textContent = message;
      const button = document.createElement("button"); button.textContent = "Restore"; button.type = "button";
      button.addEventListener("click",restore); li.append(label,button); list.append(li);
    }
    for (const key of state.failedLinks) {
      const [a,b] = key.split(":").map(Number);
      row(`ISL ${a} ↔ ${b}`,()=>restoreLink(a,b));
    }
    for (const index of state.failedSatellites) row(`Satellite ${index} offline`,()=>restoreSatellite(index));
    els.failureList.replaceChildren(list);
  }

  function focusFailure() {
    const ids = state.lastFailure || [...state.failedSatellites].slice(0,1);
    if (!ids?.length) return;
    const positions = ids.map(index => positionForRecord(state.activeRecords[index],state.viewer.clock.currentTime));
    const sphere = Cesium.BoundingSphere.fromPoints(positions);
    sphere.radius = Math.max(sphere.radius,1500000);
    state.viewer.camera.flyToBoundingSphere(sphere,{duration:.8,offset:new Cesium.HeadingPitchRange(0,-Math.PI/2,Math.max(sphere.radius*4,7000000))});
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
