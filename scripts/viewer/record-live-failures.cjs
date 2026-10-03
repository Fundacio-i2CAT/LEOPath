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
  await page.selectOption("#constellationSelect", "kuiper");
  await page.waitForFunction(() => window.leopathLive?.ready && document.querySelector("#shellLabel").textContent.startsWith("34 × 34"));
  await page.selectOption("#islTopology", "brick_a");
  await page.waitForTimeout(1200);
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
  await stage("Brick A and live routes", 8, null,
    "Brick A has two intra-plane links and one staggered cross-plane link per satellite. Both live routes follow this wiring.");
  await stage("Pause before failure", 4,
    async () => { await page.locator("#livePlay").click(); },
    "Pause holds orbital time while we introduce a failure on the selected route.");
  await stage("Fail an ISL", 8,
    async () => { await page.locator("#failRouteLink").click(); await page.locator("#focusLiveFailure").click(); },
    "The dashed red ISL is unavailable. Both routes avoid it; an orange segment marks a preview repair if the pivot rule blocks.");
  await stage("Motion with a failed link", 7,
    async () => { await page.locator("#livePlay").click(); },
    "The orbit continues with the ISL still down. Failures persist as route choices and station attachments update.");
  await stage("Restore the ISL", 5,
    async () => { await page.locator("#restoreFailures").click(); await page.locator("#resetButton").click(); },
    "Restore repairs the live network without restarting the clock or changing playback.");
  await stage("Satellite outage", 8,
    async () => { await page.locator("#livePlay").click(); await page.locator("#failRouteSatellite").click(); await page.locator("#focusLiveFailure").click(); },
    "An offline satellite appears red. Its incident ISLs and ground attachments are removed from both routing policies.");
  await stage("Satellite recovery", 5,
    async () => { await page.locator("#restoreFailures").click(); await page.locator("#resetButton").click(); },
    "The satellite returns. Routing recalculates over the restored network at the same orbital time.");
  await stage("Isolate the ingress", 8,
    async () => {
      await page.locator("#isolateIngress").click();
      if (await page.evaluate(() => window.leopathLive.route.reason) !== "partition") throw new Error("Expected partition in recorded scene");
    },
    "Cutting the ingress satellite's ISLs partitions this attachment from the destination. Neither policy can deliver the selected flow.");
  await stage("Restore connectivity", 5,
    async () => { await page.locator("#restoreFailures").click(); await page.locator("#livePlay").click(); },
    "Restoration reconnects the flow, and Play resumes the moving constellation.");
  await stage("Brick B", 8,
    async () => { await page.selectOption("#islTopology", "brick_b"); },
    "Brick B swaps the terminal split: one intra-plane link and two cross-plane links. The globe and routes use the same graph.");
  await stage("Explore and reproduce", 6, null,
    "Select an exact satellite or ISL, inject multiple failures, or compare repair enabled and disabled. These are immediate live routing previews.");
  await page.evaluate(() => (window.leopathReplay || window.leopathLive).caption(""));
  const duration = (Date.now() - start) / 1000 - trimStart;
  const video = page.video();
  await context.close();
  const raw = await video.path();
  await browser.close();
  if (errors.length) throw new Error(errors.join("\n"));
  const mp4 = path.join(output, "LEOPath_live_failures_brick_demo.mp4");
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
    path.join(output, "live_failures_captions.srt"),
    chapters
      .map(
        (c, i) =>
          `${i + 1}\n${srtTime(c.start)} --> ${srtTime(c.end)}\n${c.caption}\n`,
      )
      .join("\n"),
  );
  await fs.writeFile(
    path.join(output, "live_failures_chapters.json"),
    JSON.stringify({ duration, shell: "kuiper", mode: "live failures", chapters }, null, 2),
  );
  console.log(`Saved ${mp4} (${duration.toFixed(1)} s)`);
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
