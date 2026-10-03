/* Record the verified viewer. Requires a localhost server and ffmpeg on PATH. */
const fs = require("node:fs/promises");
const path = require("node:path");
const { spawnSync } = require("node:child_process");
const { chromium } = require("playwright");

(async () => {
  const output = path.resolve(
    process.env.LEOPATH_VIEWER_OUTPUT || "viewer-recordings",
  );
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
  const context = await browser.newContext({
    viewport: { width: 1920, height: 1080 },
    deviceScaleFactor: 1,
    recordVideo: { dir: output, size: { width: 1920, height: 1080 } },
  });
  const start = Date.now();
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  const url = process.env.LEOPATH_VIEWER_URL || "http://127.0.0.1:8765/";
  await page.goto(new URL("index.html", url).href, {waitUntil:"networkidle"});
  await page.waitForFunction(() => window.leopathLive?.ready, null, {timeout:60000});
  await page.waitForFunction(() => window.leopathLive.viewer.imageryLayers.length > 0);
  await page.waitForTimeout(2500);
  const trimStart = (Date.now() - start) / 1000;
  const chapters = [];
  const stage = async (title, seconds, action, caption) => {
    await page.evaluate(() => (window.leopathReplay || window.leopathLive).caption(""));
    const chapterStart = (Date.now() - start) / 1000 - trimStart;
    if (action) await action();
    if (caption)
      await page.evaluate(
        (message) => (window.leopathReplay || window.leopathLive).caption(message),
        caption,
      );
    await page.waitForTimeout(seconds * 1000);
    chapters.push({
      title,
      start: chapterStart,
      end: (Date.now() - start) / 1000 - trimStart,
      caption: caption || "",
    });
    console.log(`Recorded: ${title}`);
  };
  await stage(
    "Live routes in Cesium", 10, null,
    "Live routes follow the moving constellation. Yellow is topological forwarding; green is the shortest-path reference."
  );
  await stage(
    "Select another live flow", 6,
    async () => { await page.selectOption("#routeTarget", "London"); },
    "Choose source and destination stations. The route and its propagation-delay preview update on the globe."
  );
  await stage(
    "Pause and inspect", 4,
    async () => { await page.locator("#livePlay").click(); },
    "Pause freezes simulation time and satellite positions. Layer changes preserve the clock and camera."
  );
  await stage(
    "Resume orbital motion", 4,
    async () => { await page.locator("#livePlay").click(); await page.selectOption("#routeTarget", "Perth"); },
    "Play resumes continuous orbital motion. Switch to failure replay to inspect exported routing decisions."
  );
  await stage(
    "A data-backed view of routing",
    8,
    async () => {
      await page.goto(new URL("replay.html?replay=telesat-grid", url).href, {waitUntil:"networkidle"});
      await page.waitForFunction(() => window.leopathReplay?.ready, null, {timeout:60000});
    },
    "Explore the constellation and its logical grid side by side. Routes and next-hop decisions are exported by LEOPath.",
  );
  await stage(
    "Normal forwarding",
    9,
    async () => {
      await page.evaluate(() => window.leopathReplay.presentation(true));
      await page.waitForTimeout(300);
    },
    "The selected flow follows the guarded pivot rule. In this snapshot, its propagation delay matches the shortest-path reference.",
  );
  await stage(
    "One ISL fails",
    12,
    async () => {
      await page.evaluate(() => window.leopathReplay.setFrame(1));
    },
    "The red dashed link is unavailable. Three region entries across the shell support exception forwarding; this flow takes a detour.",
  );
  await stage(
    "Inspect the exception",
    10,
    async () => {
      await page.locator("#expandGrid").click();
    },
    "In plane × slot coordinates, the detour is explicit. The inspector identifies the installed exception and the blocked default rule.",
  );
  await stage(
    "The link recovers",
    9,
    async () => {
      await page.locator("#expandGrid").click();
      await page.evaluate(() => window.leopathReplay.setFrame(2));
    },
    "After recovery, the exception entries are removed and the default route resumes.",
  );
  await stage(
    "A satellite goes offline",
    10,
    async () => {
      await page.evaluate(() => window.leopathReplay.setFrame(3));
    },
    "Satellite loss removes every incident ISL. Attachments and forwarding are recomputed over the live graph.",
  );
  await stage(
    "Satellite recovery",
    6,
    async () => {
      await page.evaluate(() => window.leopathReplay.setFrame(4));
    },
    "The satellite returns and the failure exceptions disappear.",
  );
  await stage(
    "A partition limits reachability",
    12,
    async () => {
      await page.evaluate(() => window.leopathReplay.setFrame(5));
    },
    "The selected attachment addresses are now in separate components. Neither routing family has a path between them.",
  );
  await stage(
    "Restore connectivity",
    10,
    async () => {
      await page.evaluate(() => window.leopathReplay.setFrame(6));
    },
    "All 552 station pairs are deliverable again. These are converged routing snapshots; delay excludes queueing and processing.",
  );
  await stage(
    "Explore and reproduce",
    7,
    async () => {
      await page.evaluate(() => {
        window.leopathReplay.presentation(false);
        window.leopathReplay.setFrame(0);
      });
    },
    "Choose another shell, inspect a satellite, step along a route, or load a reproducible replay.",
  );
  await page.evaluate(() => (window.leopathReplay || window.leopathLive).caption(""));
  const duration = (Date.now() - start) / 1000 - trimStart;
  const video = page.video();
  await context.close();
  const raw = await video.path();
  await browser.close();
  if (errors.length) throw new Error(errors.join("\n"));
  const mp4 = path.join(output, "LEOPath_live_routes_demo.mp4");
  const converted = spawnSync(
    "ffmpeg",
    [
      "-hide_banner",
      "-loglevel",
      "error",
      "-y",
      "-ss",
      trimStart.toFixed(3),
      "-i",
      raw,
      "-t",
      duration.toFixed(3),
      "-vf",
      "fps=30",
      "-c:v",
      "libx264",
      "-preset",
      "medium",
      "-crf",
      "18",
      "-pix_fmt",
      "yuv420p",
      "-movflags",
      "+faststart",
      mp4,
    ],
    { stdio: "inherit" },
  );
  if (converted.status !== 0) throw new Error("ffmpeg conversion failed");
  const srtTime = (t) => {
    const ms = Math.round(t * 1000);
    return `${String(Math.floor(ms / 3600000)).padStart(2, "0")}:${String(Math.floor(ms / 60000) % 60).padStart(2, "0")}:${String(Math.floor(ms / 1000) % 60).padStart(2, "0")},${String(ms % 1000).padStart(3, "0")}`;
  };
  await fs.writeFile(
    path.join(output, "live_routes_captions.srt"),
    chapters
      .map(
        (c, i) =>
          `${i + 1}\n${srtTime(c.start)} --> ${srtTime(c.end)}\n${c.caption}\n`,
      )
      .join("\n"),
  );
  await fs.writeFile(
    path.join(output, "live_routes_chapters.json"),
    JSON.stringify({ duration, replay: "telesat-grid", chapters }, null, 2),
  );
  console.log(`Saved ${mp4} (${duration.toFixed(1)} s)`);
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
