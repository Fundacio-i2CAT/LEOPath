/* LEOPath replay renderer. Forwarding paths and rule choices are exported by Python. */
(function () {
  "use strict";
  const ids = [
    "expandGrid",
    "stepHop",
    "scenario",
    "source",
    "target",
    "swap",
    "showMesh",
    "showStations",
    "showReference",
    "showPrevious",
    "focus",
    "resetCamera",
    "importReplay",
    "downloadReplay",
    "presentation",
    "shellSummary",
    "loadStatus",
    "delivery",
    "deliveryDetail",
    "topoDelay",
    "topoHops",
    "lsDelay",
    "lsHops",
    "exceptionCount",
    "exceptionDetail",
    "orbitLabel",
    "phaseNumber",
    "phaseTitle",
    "phaseDetail",
    "gridShape",
    "logicalGrid",
    "failedCount",
    "componentCount",
    "inspectorTitle",
    "decisionTag",
    "inspectorBody",
    "allDelivery",
    "reachabilityDetail",
    "previous",
    "next",
    "play",
    "phases",
    "pace",
    "demoCaption",
  ];
  const els = {};
  const state = {
    data: null,
    dataset: null,
    frame: 0,
    selected: null,
    viewer: null,
    points: null,
    lines: null,
    timer: null,
    manifest: [],
    loadToken: 0,
    ready: false,
  };
  const palette = {
    topo: "#c5ed94",
    ls: "#59cfe5",
    failed: "#ff7377",
    exception: "#ffbd78",
    mesh: "#63899e",
  };
  const svgNS = "http://www.w3.org/2000/svg";
  const current = () => state.data.frames[state.frame];
  const route = () =>
    current().routes[`${els.source.value}:${els.target.value}`] || null;
  const previousRoute = () =>
    state.frame > 0
      ? state.data.frames[state.frame - 1].routes[
          `${els.source.value}:${els.target.value}`
        ]
      : null;
  const color = (hex, alpha = 1) =>
    Cesium.Color.fromCssColorString(hex).withAlpha(alpha);
  const addr = (i) =>
    `(${Math.floor(i / state.data.slots)}, ${i % state.data.slots})`;
  const text = (id, value) => {
    els[id].textContent = value;
  };
  const number = (value) => Number(value).toLocaleString();
  const km = (value) =>
    value === null || value === undefined ? "—" : (value / 1000).toFixed(1);
  const escape = (s) =>
    String(s).replace(
      /[&<>"']/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[c],
    );

  function validateReplay(data) {
    if (
      data?.schemaVersion !== 1 ||
      !Number.isInteger(data.planes) ||
      !Number.isInteger(data.slots) ||
      data.planes < 1 ||
      data.slots < 1
    )
      throw new Error("Unsupported replay schema or shell dimensions.");
    const n = data.planes * data.slots;
    if (
      typeof data.id !== "string" ||
      typeof data.topology !== "string" ||
      typeof data.epoch !== "string" ||
      !Number.isFinite(data.altitudeKm)
    )
      throw new Error("Invalid replay metadata.");
    if (
      n > 25000 ||
      !Array.isArray(data.positions) ||
      data.positions.length !== n ||
      data.positions.some(
        (p) =>
          !Array.isArray(p) ||
          p.length !== 3 ||
          p.some((x) => !Number.isFinite(x)),
      )
    )
      throw new Error("Invalid satellite positions.");
    const node = (x) => Number.isInteger(x) && x >= 0 && x < n;
    if (
      !Array.isArray(data.groundStations) ||
      data.groundStations.length < 2 ||
      data.groundStations.some(
        (g) =>
          typeof g.name !== "string" ||
          ![g.latitude, g.longitude, g.elevation_m].every(Number.isFinite),
      )
    )
      throw new Error("Invalid ground stations.");
    if (data.focusSatellite !== undefined && !node(data.focusSatellite))
      throw new Error("Invalid focus satellite.");
    if (data.defaultPair !== undefined) {
      if (
        typeof data.defaultPair !== "string" ||
        !/^\d+:\d+$/.test(data.defaultPair)
      )
        throw new Error("Invalid default flow.");
      const pair = data.defaultPair.split(":").map(Number);
      if (
        pair.some((i) => i >= data.groundStations.length) ||
        pair[0] === pair[1]
      )
        throw new Error("Invalid default flow.");
    }
    if (
      !Array.isArray(data.frames) ||
      !data.frames.length ||
      data.frames.length > 500
    )
      throw new Error("Invalid replay phases.");
    for (const f of data.frames) {
      if (
        typeof f.label !== "string" ||
        typeof f.detail !== "string" ||
        !f.stats ||
        !f.routes ||
        !f.decisions ||
        !Array.isArray(f.exceptions) ||
        !Array.isArray(f.edgeLengths) ||
        !Array.isArray(f.failedLinks) ||
        !Array.isArray(f.failedSatellites)
      )
        throw new Error("Incomplete replay phase.");
      if (
        !Array.isArray(f.attachments) ||
        f.attachments.length !== data.groundStations.length ||
        f.attachments.some(
          (row) => !Array.isArray(row) || row.some((sat) => !node(sat)),
        )
      )
        throw new Error("Invalid attachment assignments.");
      for (const name of [
        "attempted",
        "reachable",
        "delivered",
        "regionEntries",
        "rawEntries",
        "exceptionSatellites",
        "unresolved",
        "liveLinks",
        "components",
      ])
        if (!Number.isInteger(f.stats[name]) || f.stats[name] < 0)
          throw new Error("Invalid replay metrics.");
      if (
        f.failedSatellites.some((x) => !node(x)) ||
        [...f.failedLinks, ...f.edgeLengths].some(
          (e) => !Array.isArray(e) || !node(e[0]) || !node(e[1]),
        ) ||
        f.edgeLengths.some((e) => !Number.isFinite(e[2]) || e[2] <= 0) ||
        f.exceptions.some(
          (e) => !Array.isArray(e) || e.length !== 3 || e.some((x) => !node(x)),
        )
      )
        throw new Error("Invalid link or exception entry.");
      for (const r of Object.values(f.routes))
        if (
          !Number.isInteger(r.source) || !Number.isInteger(r.target) ||
          r.source < 0 || r.target < 0 || r.source >= data.groundStations.length || r.target >= data.groundStations.length ||
          (r.sourceSatellite !== undefined && !node(r.sourceSatellite)) ||
          (r.targetSatellite !== undefined && !node(r.targetSatellite)) ||
          !["delivered", "partition", "no_visibility", "forwarding_failure"].includes(r.reason) ||
          [r.topologicalDelayMs, r.linkStateDelayMs].some(delay => delay !== undefined && (!Number.isFinite(delay) || delay <= 0)) ||
          !Array.isArray(r.topological) ||
          !Array.isArray(r.linkState) ||
          [...r.topological, ...r.linkState].some((x) => !node(x))
        )
          throw new Error("Invalid exported route.");
      for (const d of Object.values(f.decisions))
        if (
          !Array.isArray(d.potential) ||
          d.potential.length !== n ||
          d.potential.some((x) => x !== null && !Number.isFinite(x)) ||
          !Array.isArray(d.ruleNext) ||
          d.ruleNext.length !== n ||
          d.ruleNext.some((x) => x !== null && !node(x))
        )
          throw new Error("Invalid exported forwarding decision.");
    }
    return data;
  }

  async function init() {
    ids.forEach((id) => (els[id] = document.getElementById(id)));
    bind();
    try {
      if (!window.Cesium)
        throw new Error(
          "Cesium did not load. Build the local viewer assets first.",
        );
      state.viewer = new Cesium.Viewer("cesiumContainer", {
        baseLayer: false,
        animation: false,
        timeline: false,
        geocoder: false,
        homeButton: false,
        sceneModePicker: false,
        navigationHelpButton: false,
        baseLayerPicker: false,
        fullscreenButton: false,
        infoBox: false,
        selectionIndicator: false,
        shouldAnimate: false,
        terrainProvider: new Cesium.EllipsoidTerrainProvider(),
        requestRenderMode: true,
        maximumRenderTimeChange: Infinity,
      });
      const scene = state.viewer.scene;
      scene.backgroundColor = color("#080f16");
      scene.globe.baseColor = color("#172f42");
      scene.globe.enableLighting = false;
      scene.highDynamicRange = false;
      scene.fog.enabled = false;
      scene.globe.depthTestAgainstTerrain = true;
      scene.skyBox.show = false;
      scene.sun.show = false;
      scene.moon.show = false;
      scene.skyAtmosphere.show = false;
      const provider = await Cesium.TileMapServiceImageryProvider.fromUrl(
        Cesium.buildModuleUrl("Assets/Textures/NaturalEarthII"),
      );
      state.viewer.imageryLayers.addImageryProvider(provider);
      state.viewer.screenSpaceEventHandler.setInputAction((click) => {
        const p = scene.pick(click.position);
        if (typeof p?.id === "string" && p.id.startsWith("sat-"))
          selectSatellite(Number(p.id.slice(4)));
      }, Cesium.ScreenSpaceEventType.LEFT_CLICK);
      const response = await fetch("replays/manifest.json");
      if (!response.ok)
        throw new Error(
          "Replay manifest is missing. Run scripts/export_viewer_replay.py.",
        );
      state.manifest = await response.json();
      for (const item of state.manifest) {
        const option = document.createElement("option");
        option.value = item.id;
        option.textContent = item.label;
        els.scenario.append(option);
      }
      const url = new URL(location.href);
      const requested = url.searchParams.get("replay");
      if (state.manifest.some((x) => x.id === requested))
        els.scenario.value = requested;
      await loadSelected();
      if (url.searchParams.has("presentation")) presentation(true);
      window.leopathReplay = {
        setFrame,
        selectSatellite,
        focusFailure,
        resetCamera,
        presentation,
        caption,
        setPair,
        loadData: installData,
        get data() {
          return state.data;
        },
        get frame() {
          return state.frame;
        },
        get ready() {
          return state.ready;
        },
        get viewer() {
          return state.viewer;
        },
        validateReplay,
      };
      state.ready = true;
    } catch (error) {
      status(error.message, true);
    }
  }

  function bind() {
    els.expandGrid.addEventListener("click", () => {
      const expanded = els.logicalGrid
        .closest(".grid-card")
        .classList.toggle("expanded");
      els.expandGrid.textContent = expanded ? "Close ×" : "Expand ↗";
      els.expandGrid.setAttribute("aria-expanded", String(expanded));
    });
    els.stepHop.addEventListener("click", () => {
      const path = route()?.topological || [];
      if (path.length)
        selectSatellite(path[(path.indexOf(state.selected) + 1) % path.length]);
    });
    els.scenario.addEventListener("change", () =>
      loadSelected().catch((e) => status(e.message, true)),
    );
    for (const id of ["source", "target"])
      els[id].addEventListener("change", () => {
        stop();
        render();
      });
    els.swap.addEventListener("click", () =>
      setPair(Number(els.target.value), Number(els.source.value)),
    );
    for (const id of [
      "showMesh",
      "showStations",
      "showReference",
      "showPrevious",
    ])
      els[id].addEventListener("change", render);
    els.previous.addEventListener("click", () => {
      stop();
      setFrame(state.frame - 1);
    });
    els.next.addEventListener("click", () => {
      stop();
      setFrame(state.frame + 1);
    });
    els.play.addEventListener("click", () => (state.timer ? stop() : play()));
    els.pace.addEventListener("change", () => {
      if (state.timer) {
        stop();
        play();
      }
    });
    els.focus.addEventListener("click", () => focusFailure());
    els.resetCamera.addEventListener("click", () => resetCamera());
    els.presentation.addEventListener("click", () =>
      presentation(!document.body.classList.contains("presentation")),
    );
    els.importReplay.addEventListener("change", async () => {
      const file = els.importReplay.files[0];
      if (!file) return;
      try {
        if (file.size > 100 * 1024 * 1024)
          throw new Error("Replay exceeds 100 MB.");
        installData(validateReplay(JSON.parse(await file.text())));
        status(`Loaded ${file.name}`);
      } catch (e) {
        status(e.message, true);
      } finally {
        els.importReplay.value = "";
      }
    });
    els.downloadReplay.addEventListener("click", () => {
      if (!state.data) return;
      const url = URL.createObjectURL(
        new Blob([JSON.stringify({...state.data, dataset: state.dataset})], { type: "application/json" }),
      );
      const a = document.createElement("a");
      a.href = url;
      a.download = `${state.data.id}.json`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        els.logicalGrid.closest(".grid-card").classList.remove("expanded");
        els.expandGrid.textContent = "Expand ↗";
        els.expandGrid.setAttribute("aria-expanded", "false");
        return;
      }
      if (
        ["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(e.target.tagName) ||
        !state.ready
      )
        return;
      if (e.code === "Space") {
        e.preventDefault();
        state.timer ? stop() : play();
      }
      if (e.key === "ArrowRight") {
        stop();
        setFrame(state.frame + 1);
      }
      if (e.key === "ArrowLeft") {
        stop();
        setFrame(state.frame - 1);
      }
      if (e.key.toLowerCase() === "p")
        presentation(!document.body.classList.contains("presentation"));
    });
  }
  function status(message, error = false) {
    text("loadStatus", message);
    els.loadStatus.classList.toggle("error", error);
  }
  async function loadSelected() {
    const token = ++state.loadToken;
    stop();
    status("Loading simulator replay…");
    const item = state.manifest.find((x) => x.id === els.scenario.value);
    if (!item) throw new Error("Select a replay.");
    const response = await fetch(`replays/${encodeURIComponent(item.path)}`);
    if (!response.ok) throw new Error("Unable to load replay.");
    const data = validateReplay(await response.json());
    const dataset = await window.leopathDatasetReady;
    if (token !== state.loadToken) return;
    installData(data, dataset, false);
    status("Ready · routes exported from LEOPath");
  }
  function installData(data, dataset = data.dataset || null, imported = true) {
    validateReplay(data);
    if (window.setLEOPathDataset) window.setLEOPathDataset(dataset, imported ? "Imported replay · unversioned" : window.leopathDatasetError ? "Dataset version unavailable" : "Local examples · unversioned");
    state.dataset = dataset;
    state.loadToken += 1;
    stop();
    state.data = data;
    state.frame = 0;
    state.selected = data.focusSatellite ?? 0;
    for (const id of ["source", "target"]) {
      els[id].replaceChildren();
      data.groundStations.forEach((g, i) => {
        const option = document.createElement("option");
        option.value = i;
        option.textContent = g.name;
        els[id].append(option);
      });
    }
    const pair = (data.defaultPair || "0:1").split(":").map(Number);
    els.source.value = pair[0];
    els.target.value = pair[1];
    text("gridShape", `${data.planes} × ${data.slots}`);
    els.shellSummary.innerHTML = `<span>${data.positions.length.toLocaleString()} satellites</span><span>${escape(data.topology.replace("grid_seam", "open seam").replace("grid", "+Grid"))}</span><span>${data.altitudeKm} km</span>`;
    text("orbitLabel", `${data.epoch.slice(11, 19)} UTC · fixed snapshot`);
    els.phases.replaceChildren();
    data.frames.forEach((f, i) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `phase-button ${f.failedLinks.length ? "fault" : ""}`;
      button.innerHTML = `<small>${String(i + 1).padStart(2, "0")}</small>${escape(f.label.replace("forwarding", "").replace("Satellite", "Sat.").replace("Network", ""))}`;
      button.addEventListener("click", () => {
        stop();
        setFrame(i);
      });
      els.phases.append(button);
    });
    render();
    resetCamera(0);
  }
  function setPair(a, b) {
    stop();
    els.source.value = a;
    els.target.value = b;
    render();
  }
  function setFrame(index) {
    if (!state.data) return;
    state.frame = Math.max(0, Math.min(state.data.frames.length - 1, index));
    render();
  }
  function selectSatellite(satellite) {
    if (
      !Number.isInteger(satellite) ||
      satellite < 0 ||
      satellite >= state.data.positions.length
    )
      return;
    state.selected = satellite;
    drawGrid();
    drawInspector();
    drawGlobe();
  }
  function stop() {
    if (state.timer) clearInterval(state.timer);
    state.timer = null;
    if (els.play) {
      els.play.textContent = "▶";
      els.play.setAttribute("aria-label", "Play failure phases");
    }
  }
  function play() {
    if (!state.data) return;
    if (state.frame === state.data.frames.length - 1) setFrame(0);
    els.play.textContent = "Ⅱ";
    els.play.setAttribute("aria-label", "Pause failure phases");
    state.timer = setInterval(
      () => {
        if (state.frame === state.data.frames.length - 1) {
          stop();
          return;
        }
        setFrame(state.frame + 1);
      },
      Number(els.pace.value) * 1000,
    );
  }
  function presentation(active) {
    document.body.classList.toggle("presentation", active);
    els.presentation.setAttribute("aria-pressed", String(active));
    els.presentation.textContent = active
      ? "Exit presentation"
      : "Presentation mode";
    requestAnimationFrame(() => {
      state.viewer.resize();
      state.viewer.scene.requestRender();
    });
  }
  function caption(message) {
    els.demoCaption.hidden = !message;
    els.demoCaption.textContent = message;
  }
  function resetCamera(duration = 0.8) {
    if (!state.data) return;
    const i = state.data.focusSatellite || 0;
    const [lon, lat] = state.data.positions[i];
    state.viewer.camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(lon, lat, 15000000),
      duration,
    });
  }
  function focusFailure(duration = 0.8) {
    const f = current();
    const satellite =
      f.failedSatellites[0] ??
      f.failedLinks[0]?.[0] ??
      state.data.focusSatellite;
    const [lon, lat] = state.data.positions[satellite];
    state.viewer.camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(lon, lat, 9000000),
      duration,
    });
    selectSatellite(satellite);
  }

  function render() {
    if (!state.data || !state.viewer) return;
    const f = current(),
      r = route();
    const labels = {
      delivered: "Delivered",
      partition: "Partitioned",
      no_visibility: "No visibility",
      forwarding_failure: "Blocked",
    };
    text(
      "delivery",
      r ? labels[r.reason] || "Unavailable" : "Select endpoints",
    );
    els.delivery.classList.toggle("bad", r?.reason !== "delivered");
    text(
      "deliveryDetail",
      `${state.data.groundStations[Number(els.source.value)]?.name || "—"} → ${state.data.groundStations[Number(els.target.value)]?.name || "—"}`,
    );
    for (const [id, kind] of [
      ["topoDelay", "topological"],
      ["lsDelay", "linkState"],
    ])
      els[id].innerHTML =
        `${r?.[kind + "DelayMs"] !== undefined ? r[kind + "DelayMs"].toFixed(2) : "—"}<em> ms</em>`;
    text(
      "topoHops",
      r?.topological.length
        ? `${r.topological.length - 1} ISL hops · GSLs included`
        : "No delivered route",
    );
    text(
      "lsHops",
      r?.linkState.length
        ? `${r.linkState.length - 1} ISL hops · same addresses`
        : "No reachable reference route",
    );
    text("exceptionCount", number(f.stats.regionEntries));
    text(
      "exceptionDetail",
      `${number(f.stats.rawEntries)} raw entries · ${number(f.stats.exceptionSatellites)} satellites`,
    );
    text(
      "phaseNumber",
      `PHASE ${String(state.frame + 1).padStart(2, "0")} / ${String(state.data.frames.length).padStart(2, "0")}`,
    );
    text("phaseTitle", f.label);
    text("phaseDetail", f.detail);
    text(
      "failedCount",
      `${number(f.failedLinks.length)} unavailable links · ${f.failedSatellites.length} satellites`,
    );
    text(
      "componentCount",
      `${f.stats.components} component${f.stats.components === 1 ? "" : "s"}`,
    );
    text(
      "allDelivery",
      `${number(f.stats.delivered)} / ${number(f.stats.attempted)}`,
    );
    text(
      "reachabilityDetail",
      `${number(f.stats.reachable)} pairs have a live path under the same attachment policy. ${number(f.stats.unresolved)} unresolved reachable walks. Counts describe routes, not packets.`,
    );
    els.stepHop.disabled = !r?.topological.length;
    els.previous.disabled = state.frame === 0;
    els.next.disabled = state.frame === state.data.frames.length - 1;
    [...els.phases.children].forEach((button, i) => {
      button.classList.toggle("active", i === state.frame);
      button.setAttribute("aria-pressed", String(i === state.frame));
    });
    drawGrid();
    drawInspector();
    drawGlobe();
  }

  function svgElement(name, attributes, parent) {
    const el = document.createElementNS(svgNS, name);
    for (const [key, value] of Object.entries(attributes))
      el.setAttribute(key, String(value));
    parent.append(el);
    return el;
  }
  function drawGrid() {
    const svg = els.logicalGrid;
    svg.replaceChildren();
    const d = state.data,
      f = current(),
      r = route();
    const width = 545,
      height = 270,
      left = 48,
      top = 45;
    const xy = (i) => [
      left + (Math.floor(i / d.slots) / Math.max(1, d.planes - 1)) * width,
      top + ((i % d.slots) / Math.max(1, d.slots - 1)) * height,
    ];
    const link = (a, b, cls) => {
      const [ax, ay] = xy(a),
        [bx, by] = xy(b);
      if (Math.abs(ax - bx) > width * 0.8) {
        svgElement(
          "path",
          {
            d: `M ${ax} ${ay} L ${ax < bx ? left - 14 : left + width + 14} ${ay} M ${bx} ${by} L ${bx < ax ? left - 14 : left + width + 14} ${by}`,
            class: cls + " grid-wrap",
          },
          svg,
        );
      } else if (Math.abs(ay - by) > height * 0.8) {
        svgElement(
          "path",
          {
            d: `M ${ax} ${ay} L ${ax} ${ay < by ? top - 14 : top + height + 14} M ${bx} ${by} L ${bx} ${by < ay ? top - 14 : top + height + 14}`,
            class: cls + " grid-wrap",
          },
          svg,
        );
      } else
        svgElement("line", { x1: ax, y1: ay, x2: bx, y2: by, class: cls }, svg);
    };
    if (els.showMesh.checked)
      f.edgeLengths.forEach((e) => link(e[0], e[1], "grid-link"));
    if (els.showPrevious.checked) {
      const p = previousRoute()?.topological || [];
      for (let i = 1; i < p.length; i++) link(p[i - 1], p[i], "grid-previous");
    }
    for (const [kind, cls] of [
      ["linkState", "grid-ls"],
      ["topological", "grid-topo"],
    ]) {
      if (kind === "linkState" && !els.showReference.checked) continue;
      const path = r?.[kind] || [];
      for (let i = 1; i < path.length; i++) link(path[i - 1], path[i], cls);
    }
    f.failedLinks.forEach((e) => link(e[0], e[1], "grid-failed"));
    const ex = new Set(
        f.exceptions
          .filter((e) => e[1] === r?.targetSatellite)
          .map((e) => e[0]),
      ),
      down = new Set(f.failedSatellites),
      onRoute = new Set(r?.topological || []);
    for (let i = 0; i < d.positions.length; i++) {
      const [x, y] = xy(i);
      const classes = ["grid-node"];
      if (onRoute.has(i)) classes.push("on-route");
      if (ex.has(i)) classes.push("exception");
      if (down.has(i)) classes.push("down");
      if (i === r?.sourceSatellite || i === r?.targetSatellite)
        classes.push("endpoint");
      if (i === state.selected) classes.push("selected");
      const circle = svgElement(
        "circle",
        {
          cx: x,
          cy: y,
          r: d.planes > 50 ? 2 : 3,
          class: classes.join(" "),
          "data-satellite": i,
          tabindex: i === state.selected ? 0 : -1,
          role: "button",
          "aria-label": `Satellite ${i}, plane ${Math.floor(i / d.slots)}, slot ${i % d.slots}`,
        },
        svg,
      );
      const title = svgElement("title", {}, circle);
      title.textContent = `Satellite ${i} · plane/slot ${addr(i)}`;
      circle.addEventListener("click", () => selectSatellite(i));
      circle.addEventListener("keydown", (e) => {
        let next = i;
        const plane = Math.floor(i / d.slots),
          slot = i % d.slots;
        if (e.key === "ArrowRight")
          next = ((plane + 1) % d.planes) * d.slots + slot;
        else if (e.key === "ArrowLeft")
          next = ((plane - 1 + d.planes) % d.planes) * d.slots + slot;
        else if (e.key === "ArrowUp")
          next = plane * d.slots + ((slot - 1 + d.slots) % d.slots);
        else if (e.key === "ArrowDown")
          next = plane * d.slots + ((slot + 1) % d.slots);
        else return;
        e.preventDefault();
        e.stopPropagation();
        selectSatellite(next);
        svg.querySelector(`[data-satellite="${next}"]`)?.focus();
      });
    }
    const xTitle = svgElement(
      "text",
      {
        x: left + width / 2,
        y: 352,
        "text-anchor": "middle",
        class: "grid-axis",
      },
      svg,
    );
    xTitle.textContent = "ORBITAL PLANE →";
    const yTitle = svgElement(
      "text",
      {
        x: 15,
        y: 180,
        transform: "rotate(-90 15 180)",
        "text-anchor": "middle",
        class: "grid-axis",
      },
      svg,
    );
    yTitle.textContent = "SLOT IN PLANE →";
    for (const p of [
      ...new Set([0, Math.floor((d.planes - 1) / 2), d.planes - 1]),
    ]) {
      const el = svgElement(
        "text",
        {
          x: left + (p / Math.max(1, d.planes - 1)) * width,
          y: 27,
          "text-anchor": "middle",
          class: "grid-axis",
        },
        svg,
      );
      el.textContent = p;
    }
    for (const s of [
      ...new Set([0, Math.floor((d.slots - 1) / 2), d.slots - 1]),
    ]) {
      const el = svgElement(
        "text",
        {
          x: 34,
          y: top + (s / Math.max(1, d.slots - 1)) * height + 4,
          "text-anchor": "end",
          class: "grid-axis",
        },
        svg,
      );
      el.textContent = s;
    }
  }

  function drawInspector() {
    const f = current(),
      r = route(),
      sat = state.selected;
    if (sat === null) return;
    text("inspectorTitle", `Satellite ${sat} · ${addr(sat)}`);
    const dst = r?.targetSatellite,
      decisions = f.decisions[String(dst)];
    const ex = f.exceptions.find((e) => e[0] === sat && e[1] === dst);
    const failed = f.failedSatellites.includes(sat);
    const isDest = sat === dst;
    const rule = decisions?.ruleNext[sat];
    const chosen = ex ? ex[2] : rule;
    text(
      "decisionTag",
      failed
        ? "OUTAGE"
        : isDest
          ? "DESTINATION"
          : ex
            ? "EXCEPTION"
            : rule !== null && rule !== undefined
              ? "RULE"
              : "NO PROGRESS",
    );
    if (!decisions) {
      els.inspectorBody.innerHTML =
        '<p class="muted">Choose two different attached stations to inspect a destination-specific decision.</p>';
      return;
    }
    const adjacent = new Map();
    for (const [a, b, len] of f.edgeLengths) {
      if (a === sat) adjacent.set(b, len);
      if (b === sat) adjacent.set(a, len);
    }
    const lost = new Set(
      f.failedLinks
        .filter((e) => e.includes(sat))
        .map((e) => (e[0] === sat ? e[1] : e[0])),
    );
    const own = decisions.potential[sat];
    let explanation = failed
      ? "This satellite is unavailable and forwards no traffic."
      : isDest
        ? "The packet has reached the destination attachment satellite."
        : ex
          ? `<strong>Exception → satellite ${chosen}.</strong> The installed entry overrides the default rule${rule === null ? ", which has no admissible neighbor here" : ""}.`
          : rule !== null && rule !== undefined
            ? `Rule → satellite ${rule}. Candidates must strictly decrease (remaining distance, satellite ID).`
            : "No neighbor passes the progress guard. The rule cannot advance toward this address.";
    const rows = [...new Set([...adjacent.keys(), ...lost])]
      .sort((a, b) => a - b)
      .map((next) => {
        const live = adjacent.has(next) && !failed;
        const potential = decisions.potential[next];
        const progress =
          live &&
          potential !== null &&
          own !== null &&
          (potential < own || (potential === own && next < sat));
        const isChosen = live && next === chosen;
        return `<tr class="${!live ? "unavailable" : isChosen ? "chosen" : !progress ? "blocked" : ""}"><td>${next} ${addr(next)}</td><td>${live ? km(adjacent.get(next)) : "—"}</td><td>${km(potential)}</td><td>${!live ? "Down" : isChosen ? (ex ? "Entry ✓" : "Rule ✓") : progress ? "Progress" : "Blocked"}</td></tr>`;
      })
      .join("");
    els.inspectorBody.innerHTML = `<p class="inspector-meta">Locator <code>(0, ${Math.floor(sat / state.data.slots)}, ${sat % state.data.slots}, 0)</code><br>Destination ${dst} ${addr(dst)} · remaining estimate ${km(own)} km</p><p class="decision-explanation">${explanation}</p><table class="neighbor-table"><thead><tr><th>Neighbor</th><th>First hop<br>km</th><th>Remaining<br>km</th><th>Decision</th></tr></thead><tbody>${rows}</tbody></table><p class="muted">${f.exceptions.filter((e) => e[0] === sat).length} raw exception entries installed here across destinations.</p>`;
  }

  function drawGlobe() {
    const viewer = state.viewer,
      d = state.data,
      f = current(),
      r = route();
    if (state.points) viewer.scene.primitives.remove(state.points);
    if (state.lines) viewer.scene.primitives.remove(state.lines);
    viewer.entities.removeAll();
    state.points = viewer.scene.primitives.add(
      new Cesium.PointPrimitiveCollection(),
    );
    state.lines = viewer.scene.primitives.add(new Cesium.PolylineCollection());
    const positions = d.positions.map((p) =>
        Cesium.Cartesian3.fromDegrees(...p),
      ),
      onRoute = new Set(r?.topological || []),
      down = new Set(f.failedSatellites),
      ex = new Set(
        f.exceptions
          .filter((e) => e[1] === r?.targetSatellite)
          .map((e) => e[0]),
      );
    positions.forEach((position, i) =>
      state.points.add({
        id: `sat-${i}`,
        position,
        pixelSize:
          i === state.selected
            ? 10
            : down.has(i)
              ? 8
              : ex.has(i)
                ? 7
                : onRoute.has(i)
                  ? 5
                  : 2.5,
        color: color(
          down.has(i)
            ? palette.failed
            : ex.has(i)
              ? palette.exception
              : onRoute.has(i)
                ? palette.topo
                : palette.mesh,
          onRoute.has(i) || ex.has(i) || down.has(i) ? 1 : 0.55,
        ),
        outlineColor:
          i === state.selected ? Cesium.Color.WHITE : Cesium.Color.TRANSPARENT,
        outlineWidth: i === state.selected ? 2 : 0,
      }),
    );
    if (els.showMesh.checked) {
      const material = Cesium.Material.fromType("Color", {
        color: color(palette.mesh, 0.13),
      });
      f.edgeLengths.forEach(([a, b]) =>
        state.lines.add({
          positions: [positions[a], positions[b]],
          width: 1,
          material,
        }),
      );
    }
    const unavailable = Cesium.Material.fromType("PolylineDash", {
      color: color(palette.failed, 0.95),
      dashLength: 12,
    });
    f.failedLinks.forEach(([a, b]) =>
      state.lines.add({
        positions: [positions[a], positions[b]],
        width: 2.5,
        material: unavailable,
      }),
    );
    const routeLine = (path, hex, width, dashed = false) => {
      if (!path?.length) return;
      const material = Cesium.Material.fromType(
        dashed ? "PolylineDash" : "Color",
        { color: color(hex, dashed ? 0.45 : 1) },
      );
      for (let i = 1; i < path.length; i++)
        state.lines.add({
          positions: [positions[path[i - 1]], positions[path[i]]],
          width,
          material,
        });
    };
    if (els.showPrevious.checked)
      routeLine(previousRoute()?.topological, "#bbc8d0", 2, true);
    if (els.showReference.checked) routeLine(r?.linkState, palette.ls, 7);
    routeLine(r?.topological, palette.topo, 3.5);
    const stations = els.showStations.checked
      ? d.groundStations.map((_, i) => i)
      : [Number(els.source.value), Number(els.target.value)];
    viewer.entities.suspendEvents();
    for (const i of new Set(stations)) {
      const g = d.groundStations[i];
      if (!g) continue;
      const position = Cesium.Cartesian3.fromDegrees(
        g.longitude,
        g.latitude,
        g.elevation_m,
      );
      const selected =
        i === Number(els.source.value) || i === Number(els.target.value);
      viewer.entities.add({
        position,
        point: {
          pixelSize: selected ? 10 : 5,
          color: color(palette.ls),
          outlineColor: color("#07111b"),
          outlineWidth: 2,
        },
        label: selected
          ? {
              text: g.name,
              font: "14px sans-serif",
              fillColor: Cesium.Color.WHITE,
              outlineColor: color("#07111b"),
              outlineWidth: 3,
              style: Cesium.LabelStyle.FILL_AND_OUTLINE,
              pixelOffset: new Cesium.Cartesian2(0, -20),
            }
          : undefined,
      });
      for (const sat of f.attachments[i])
        state.lines.add({
          positions: [position, positions[sat]],
          width: selected ? 2 : 1,
          material: Cesium.Material.fromType("Color", {
            color: color(palette.ls, selected ? 0.8 : 0.2),
          }),
        });
    }
    viewer.entities.resumeEvents();
    viewer.scene.requestRender();
  }
  document.addEventListener("DOMContentLoaded", init);
})();
