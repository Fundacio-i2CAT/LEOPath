/* Exercise the viewer and compare rendered decisions with exported simulator data. */
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const path = require("node:path");
const { chromium } = require("playwright");
(async () => {
  const url = process.env.LEOPATH_VIEWER_URL || "http://127.0.0.1:8765/";
  const output =
    process.env.LEOPATH_VIEWER_OUTPUT || path.resolve("viewer-recordings");
  await fs.mkdir(output, { recursive: true });
  const browser = await chromium.launch({
    executablePath: process.env.LEOPATH_CHROMIUM || "/usr/bin/chromium",
    headless: true,
    args: [
      "--no-sandbox",
      "--use-angle=swiftshader",
      "--enable-unsafe-swiftshader",
      "--disable-dev-shm-usage",
    ],
  });
  const page = await browser.newPage({
    viewport: { width: 1920, height: 1080 },
  });
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto(new URL("replay.html", url).href, { waitUntil: "networkidle" });
  await page.waitForFunction(() => window.leopathReplay?.ready, null, {
    timeout: 60000,
  });
  assert.ok(await page.locator(".brand-logo").evaluate(img => img.complete && img.naturalWidth > 0), "Replay logo must load");
  const manifest = await page
    .locator("#scenario option")
    .evaluateAll((options) => options.map((o) => o.value));
  let phases = 0;
  for (const id of manifest) {
    await page.selectOption("#scenario", id);
    await page.waitForFunction((id) => window.leopathReplay.data.id === id, id);
    const count = await page.evaluate(
      () => window.leopathReplay.data.frames.length,
    );
    for (let i = 0; i < count; i++) {
      await page.evaluate((i) => window.leopathReplay.setFrame(i), i);
      const model = await page.evaluate(() => {
        const a = window.leopathReplay;
        const f = a.data.frames[a.frame];
        const pair = `${document.querySelector("#source").value}:${document.querySelector("#target").value}`;
        return {
          reason: f.routes[pair].reason,
          entries: f.stats.regionEntries,
        };
      });
      assert.equal(
        await page.locator("#exceptionCount").innerText(),
        model.entries.toLocaleString(),
      );
      if (model.reason === "partition")
        assert.equal(
          await page.locator("#delivery").innerText(),
          "Partitioned",
        );
      if (model.reason === "delivered")
        assert.equal(await page.locator("#delivery").innerText(), "Delivered");
      phases++;
    }
    console.log(`PASS ${id}: ${count} phases`);
  }
  await page.selectOption("#scenario", "telesat-grid");
  await page.waitForFunction(
    () => window.leopathReplay.data.id === "telesat-grid",
  );
  await page.evaluate(() => window.leopathReplay.setFrame(1));
  assert.equal(await page.locator("#decisionTag").innerText(), "EXCEPTION");
  assert.equal(
    await page.locator(".neighbor-table .chosen td:last-child").innerText(),
    "Entry ✓",
  );
  await page.locator("#expandGrid").click();
  await page.keyboard.press("Escape");
  assert.equal(await page.locator("#expandGrid").getAttribute("aria-expanded"), "false");
  await page.locator("#expandGrid").click();
  assert.equal(
    await page.locator("#expandGrid").getAttribute("aria-expanded"),
    "true",
  );
  await page.locator("#expandGrid").click();
  const sat = await page.evaluate(
    () => window.leopathReplay.data.focusSatellite,
  );
  await page.locator(`[data-satellite="${sat}"]`).focus();
  await page.keyboard.press("ArrowRight");
  assert.equal(await page.evaluate(() => window.leopathReplay.frame), 1,
    "Grid navigation must not also advance the replay phase");
  assert.notEqual(
    await page.locator("#inspectorTitle").innerText(),
    `Satellite ${sat} · (${Math.floor(sat / 13)}, ${sat % 13})`,
  );
  await page.evaluate((sat) => window.leopathReplay.selectSatellite(sat), sat);
  await page.locator("#showMesh").uncheck();
  await page.locator("#showMesh").check();
  await page.locator("#showReference").uncheck();
  await page.locator("#showReference").check();
  await page.locator("#showPrevious").check();
  await page.locator("#showPrevious").uncheck();
  await page.locator("#showStations").check();
  await page.locator("#showStations").uncheck();
  await page.locator("#stepHop").click();
  await page.locator("#swap").click();
  await page.locator("#swap").click();
  const bad = await page.evaluate(() => {
    const data = structuredClone(window.leopathReplay.data);
    data.positions[0][0] = "bad";
    try {
      window.leopathReplay.validateReplay(data);
      return false;
    } catch {
      return true;
    }
  });
  assert.ok(bad);
  const invalidAddress = await page.evaluate(() => {
    const data = structuredClone(window.leopathReplay.data);
    Object.values(data.frames[0].routes)[0].targetSatellite = "<img src=x>";
    try { window.leopathReplay.validateReplay(data); return false; } catch { return true; }
  });
  assert.ok(invalidAddress);
  await page.locator("#importReplay").setInputFiles({name:"invalid.json",mimeType:"application/json",buffer:Buffer.from('{"schemaVersion":0}')});
  await page.waitForFunction(() => document.querySelector("#loadStatus").classList.contains("error"));
  assert.equal(await page.evaluate(() => window.leopathReplay.data.id), "telesat-grid");
  const validReplay = await page.evaluate(() => JSON.stringify(window.leopathReplay.data));
  await page.locator("#importReplay").setInputFiles({name:"valid.json",mimeType:"application/json",buffer:Buffer.from(validReplay)});
  await page.waitForFunction(() => document.querySelector("#loadStatus").textContent === "Loaded valid.json");
  assert.equal(await page.evaluate(() => window.leopathReplay.frame), 0);
  await page.locator("#pace").selectOption("4");
  await page.locator("#play").click();
  await page.waitForFunction(() => window.leopathReplay.frame === 1, null, {timeout:10000});
  await page.locator("#play").click();
  await page.selectOption(
    "#target",
    await page.locator("#source").inputValue(),
  );
  assert.equal(await page.locator("#delivery").innerText(), "Select endpoints");
  await page.evaluate(() => {
    const [a, b] = window.leopathReplay.data.defaultPair.split(":").map(Number);
    window.leopathReplay.setPair(a, b);
    window.leopathReplay.setFrame(1);
    window.leopathReplay.selectSatellite(
      window.leopathReplay.data.focusSatellite,
    );
  });
  await page.waitForTimeout(1200);
  await page.screenshot({ path: path.join(output, "viewer-failure.png") });
  await page.evaluate(() => window.leopathReplay.setFrame(0));
  await page.screenshot({ path: path.join(output, "viewer-normal.png") });
  await page.locator("#presentation").click();
  await page.evaluate(() => window.leopathReplay.setFrame(1));
  await page.waitForTimeout(800);
  await page.screenshot({ path: path.join(output, "viewer-presentation.png") });
  const bounds = await page.locator(".reachability-card").boundingBox();
  assert.ok(
    bounds.y + bounds.height <= 1000,
    "Reachability card should be visible at 1080p",
  );
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator("#presentation").click();
  await page.screenshot({
    path: path.join(output, "viewer-mobile.png"),
    fullPage: true,
  });
  assert.equal(
    await page.evaluate(
      () => document.documentElement.scrollWidth > window.innerWidth,
    ),
    false,
    "Mobile page should not overflow horizontally",
  );
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto(new URL("index.html", url).href, { waitUntil: "networkidle" });
  await page.waitForFunction(() => window.leopathLive?.ready, null, { timeout: 60000 });
  await page.waitForFunction(() => window.leopathLive.viewer.imageryLayers.length > 0);
  const liveState = () => page.evaluate(() => {
    const v = window.leopathLive.viewer;
    const c = v.clock;
    const entity = v.entities.values.find(e => e.id.startsWith("sat-"));
    const p = entity.position.getValue(c.currentTime);
    return {
      seconds: Cesium.JulianDate.secondsDifference(c.currentTime, c.startTime),
      position: [p.x, p.y, p.z],
      playing: c.shouldAnimate,
      multiplier: c.multiplier,
    };
  });
  assert.ok(await page.locator(".brand-logo").evaluate(img => img.complete && img.naturalWidth > 0), "Live logo must load");
  assert.equal(await page.locator("#routeSource").isVisible(), true);
  assert.equal(await page.locator("#routeSource").inputValue(), "New York");
  assert.equal(await page.locator("#routeTarget").inputValue(), "Perth");
  const routeState = () => page.evaluate(() => window.leopathLive.route);
  const initialRoute = await routeState();
  assert.ok(initialRoute.topo.length > 1 && initialRoute.ls.length > 1, "Live routes must be visible by default");
  assert.equal(initialRoute.topo[0], initialRoute.source);
  assert.equal(initialRoute.topo.at(-1), initialRoute.destination);
  assert.equal(new Set(initialRoute.topo).size, initialRoute.topo.length, "Live topological routes must be loop-free");
  assert.equal(initialRoute.ls.at(-1), initialRoute.destination);
  const before = await liveState();
  await page.waitForTimeout(1600);
  const moving = await liveState();
  assert.ok(moving.seconds - before.seconds > 60, "Live clock must advance at the selected speed");
  assert.ok(Math.hypot(...moving.position.map((x, i) => x - before.position[i])) > 1000, "Satellite positions must move with the clock");
  assert.ok((await routeState()).updatedAt > initialRoute.updatedAt, "Routes must recompute as orbital time advances");
  await page.locator("#livePlay").click();
  assert.equal(await page.locator("#livePlay").innerText(), "▶ Play");
  const paused = await liveState();
  const pausedRoute = await routeState();
  await page.waitForTimeout(700);
  assert.deepEqual(await liveState(), paused, "Pause must freeze both clock and satellites");
  assert.deepEqual(await routeState(), pausedRoute, "Paused routes must stay fixed");
  await page.locator("#showLinkStateRoute").uncheck();
  assert.equal(await page.evaluate(() => Boolean(window.leopathLive.viewer.entities.getById("linkstate-route"))), false);
  await page.locator("#showLinkStateRoute").check();
  await page.locator("#swapRoute").click();
  assert.equal(await page.locator("#routeSource").inputValue(), "Perth");
  await page.locator("#swapRoute").click();
  await page.selectOption("#routeTarget", "London");
  assert.equal(await page.locator("#routePair").innerText(), "New York → London");
  await page.selectOption("#routeTarget", "Perth");
  await page.selectOption("#islTopology", "ring");
  const ringRoute = await routeState();
  assert.ok(ringRoute.topo.every(id => Math.floor(id / 13) === Math.floor(ringRoute.source / 13)), "Ring routes must stay in one orbital plane");
  await page.locator("#showGsl").uncheck();
  await page.locator("#showGround").uncheck();
  await page.locator("#showGround").check();
  await page.locator("#showGsl").check();
  assert.deepEqual(await liveState(), paused, "Layer changes must preserve time and pause state");
  await page.locator("#speedSlider").evaluate(input => {
    input.value = "240";
    input.dispatchEvent(new Event("input", {bubbles:true}));
  });
  await page.locator("#livePlay").click();
  await page.waitForTimeout(1000);
  const resumed = await liveState();
  assert.ok(resumed.seconds - paused.seconds > 100, "Play must resume orbital motion");
  assert.equal(resumed.multiplier, 240);
  await page.locator("#livePlay").click();
  await page.locator("#restartClock").click();
  assert.equal((await liveState()).seconds, 0);
  assert.equal((await liveState()).playing, false);
  const timeline = page.locator(".cesium-timeline-bar");
  const timelineBox = await timeline.boundingBox();
  await timeline.click({position:{x:Math.floor(timelineBox.width * .55),y:8}});
  assert.ok((await liveState()).seconds > 1000, "Seeking must change orbital time");
  await page.locator("#livePlay").click();
  const seekStart = (await liveState()).seconds;
  await page.waitForTimeout(500);
  assert.ok((await liveState()).seconds > seekStart, "Play must work after timeline seeking");
  await page.locator("#livePlay").click();
  await page.locator("body").click({position:{x:700,y:100}});
  await page.keyboard.press("Space");
  assert.equal((await liveState()).playing, true);
  await page.keyboard.press("Space");
  assert.equal((await liveState()).playing, false);
  await page.selectOption("#constellationSelect", "oneweb");
  await page.waitForFunction(() => window.leopathLive?.ready && document.querySelector("#shellLabel").textContent.startsWith("12 × 49"));
  assert.equal((await liveState()).playing, false, "Switching shells must preserve pause state");
  assert.equal(await page.locator("#replayLink").getAttribute("href"), "replay.html?replay=oneweb-grid");
  assert.equal(await page.locator("#routeSource").isVisible(), true);
  await page.selectOption("#islTopology", "grid");
  const seamRoute = await routeState();
  for (const ids of [seamRoute.topo, seamRoute.ls]) {
    for (let i=1;i<ids.length;i++) {
      const p = Math.floor(ids[i-1]/49), q = Math.floor(ids[i]/49);
      assert.ok(!(p===0 && q===11 || p===11 && q===0), "OneWeb routes must respect the open seam");
    }
  }
  await page.selectOption("#routeTarget", "New York");
  assert.equal(await page.locator("#routeDelivery").innerText(), "Choose two different endpoints");
  await page.selectOption("#routeTarget", "Perth");
  await page.selectOption("#islTopology", "none");
  assert.equal(await page.locator("#routeDelivery").innerText(), "Choose an ISL topology to show routes");
  await page.selectOption("#islTopology", "grid");
  await page.locator("#hidePanel").click();
  assert.equal(await page.locator(".panel").isVisible(), false);
  await page.locator("#showPanel").click();
  await page.screenshot({path:path.join(output,"viewer-live.png")});
  await page.setViewportSize({width:390,height:844});
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
  await page.screenshot({path:path.join(output,"viewer-live-mobile.png")});
  await page.goto(new URL("orbit.html", url).href, {waitUntil:"networkidle"});
  await page.waitForFunction(() => window.leopathLive?.ready);
  assert.ok(page.url().endsWith("index.html"), "Existing live-view links must resolve to the default viewer");
  console.log("PASS live routes, dynamic recomputation, route controls, open seam, logos, orbital movement, pause, resume, speed, restart, preserved clock, shell selection, mode links and mobile layout");
  assert.deepEqual(errors, []);
  console.log(
    `PASS ${phases} replay phases, layer controls, route stepping, keyboard selection, mobile layout, invalid input, and orbital explorer; no JavaScript exceptions`,
  );
  await browser.close();
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
