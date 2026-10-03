/* Check displayed provenance, portable downloads, imports and missing release TLEs. */
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const {chromium} = require("playwright");
(async () => {
  const base=process.env.LEOPATH_VIEWER_URL || "http://127.0.0.1:8765/";
  const browser=await chromium.launch({executablePath:"/usr/bin/chromium",headless:true,args:["--no-sandbox","--use-angle=swiftshader","--enable-unsafe-swiftshader","--disable-dev-shm-usage"]});
  try {
    const page=await browser.newPage({viewport:{width:1920,height:1080}});
    const errors=[];
    page.on("pageerror",e=>errors.push(String(e)));
    const response=await page.request.get(new URL("dataset.json",base).href);
    assert.ok(response.ok(),"Run against a dataset-backed build");
    const provenance=await response.json();
    for (const mode of ["index.html","replay.html"]) {
      await page.goto(new URL(mode,base).href,{waitUntil:"networkidle"});
      await page.waitForFunction(mode=>mode==="index.html"?window.leopathLive?.ready:window.leopathReplay?.ready,mode,{timeout:60000});
      assert.match(await page.locator("#datasetVersion").innerText(),new RegExp(provenance.commit.slice(0,7)));
      assert.equal(await page.locator("#datasetReleases").isVisible(),true);
    }
    // Test a released version without publishing a synthetic test release.
    const released={...provenance,tag:"v9.9.9",preview:false};
    await page.route("**/dataset.json",route=>route.fulfill({json:released}));
    await page.goto(new URL("replay.html",base).href,{waitUntil:"networkidle"});
    await page.waitForFunction(()=>window.leopathReplay?.ready,null,{timeout:60000});
    assert.match(await page.locator("#datasetVersion").innerText(),/Dataset v9\.9\.9/);
    const downloadPromise=page.waitForEvent("download");
    await page.locator("#downloadReplay").click();
    const download=await downloadPromise;
    const file=JSON.parse(await fs.readFile(await download.path(),"utf8"));
    assert.equal(file.dataset.tag,"v9.9.9");
    await page.locator("#importReplay").setInputFiles({name:"versioned.json",mimeType:"application/json",buffer:Buffer.from(JSON.stringify(file))});
    assert.match(await page.locator("#datasetVersion").innerText(),/Dataset v9\.9\.9/);
    delete file.dataset;
    await page.locator("#importReplay").setInputFiles({name:"unversioned.json",mimeType:"application/json",buffer:Buffer.from(JSON.stringify(file))});
    assert.equal(await page.locator("#datasetVersion").innerText(),"Imported replay · unversioned");
    await page.selectOption("#scenario","kuiper-grid");
    await page.waitForFunction(()=>window.leopathReplay.data.id==="kuiper-grid");
    assert.match(await page.locator("#datasetVersion").innerText(),/Dataset v9\.9\.9/);
    const remote=[];
    page.on("request",request=>{if(request.url().includes("raw.githubusercontent.com"))remote.push(request.url())});
    await page.route("**/data/*.txt",route=>route.fulfill({status:404,body:"missing"}));
    await page.goto(new URL("index.html",base).href,{waitUntil:"networkidle"});
    await page.waitForFunction(()=>document.querySelector("#status").textContent.startsWith("Failed to load constellation"));
    assert.equal(await page.evaluate(()=>window.leopathLive.ready),false);
    assert.deepEqual(remote,[],"Released data must not use an unversioned remote fallback");
    assert.ok(await page.locator("#livePlay").isDisabled());
    assert.deepEqual(errors,[]);
    console.log("PASS dataset versions, portable downloads, imported provenance, restored source and missing-TLE failure");
  } finally { await browser.close(); }
})().catch(error=>{console.error(error);process.exit(1)});
