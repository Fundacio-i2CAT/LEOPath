/* Verify live failure transitions and brick routes against the actual active graph. */
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const {chromium} = require('playwright');
(async()=>{
  const browser = await chromium.launch({executablePath:process.env.LEOPATH_CHROMIUM || '/usr/bin/chromium',headless:true,args:['--no-sandbox','--use-angle=swiftshader','--enable-unsafe-swiftshader','--disable-dev-shm-usage']});
  const page = await browser.newPage({viewport:{width:1440,height:900}});
  const errors=[];page.on('pageerror',e=>errors.push(String(e)));
  const url=process.env.LEOPATH_VIEWER_URL || 'http://127.0.0.1:8765/';
  const output=process.env.LEOPATH_VIEWER_OUTPUT || path.resolve('viewer-recordings');
  await fs.mkdir(output,{recursive:true});
  await page.goto(new URL('index.html',url).href,{waitUntil:'networkidle'});
  await page.waitForFunction(()=>window.leopathLive?.ready);
  await page.locator('#livePlay').click();
  const snapshot=()=>page.evaluate(()=>{
    const a=window.leopathLive,v=a.viewer;
    return {route:a.route,graph:a.graph,failures:a.failures,
      seconds:Cesium.JulianDate.secondsDifference(v.clock.currentTime,v.clock.startTime),playing:v.clock.shouldAnimate};
  });
  let checks=0;
  async function verify(){
    const {route,graph,failures}=await snapshot();
    const edges=new Set(graph.activeEdges.map(([a,b])=>`${Math.min(a,b)}:${Math.max(a,b)}`));
    const failed=new Set(failures.satellites);
    for(const ids of [route.topo,route.ls,route.repair]){
      assert.equal(new Set(ids).size,ids.length,'Paths must be loop-free');
      for(const id of ids)assert.ok(!failed.has(id),'A route cannot traverse an offline satellite');
      for(let i=1;i<ids.length;i++)assert.ok(edges.has(`${Math.min(ids[i-1],ids[i])}:${Math.max(ids[i-1],ids[i])}`),'Every route edge must be live in the selected wiring');
    }
    if(route.reason==='delivered'){
      assert.equal(route.topo[0],route.source);assert.equal(route.topo.at(-1),route.destination);
    }
    if(route.repair.length)assert.deepEqual(route.topo.slice(-route.repair.length),route.repair,'Repair must be a real suffix of the displayed route');
    if(route.source!==undefined){
      assert.ok(!failed.has(route.source)&&!failed.has(route.destination),'Attachments cannot be offline');
      const neighbors=new Map();for(const [a,b]of graph.activeEdges){if(!neighbors.has(a))neighbors.set(a,[]);if(!neighbors.has(b))neighbors.set(b,[]);neighbors.get(a).push(b);neighbors.get(b).push(a);}
      const seen=new Set([route.source]),queue=[route.source];for(let i=0;i<queue.length;i++)for(const next of neighbors.get(queue[i])||[])if(!seen.has(next)){seen.add(next);queue.push(next);}
      const reachable=seen.has(route.destination);
      assert.equal(route.reason==='partition',!reachable,'Partitions must agree with graph reachability');
      if(route.ls.length){assert.equal(route.ls[0],route.source);assert.equal(route.ls.at(-1),route.destination);}
    }
    checks++;return route;
  }
  const baseline=await snapshot();
  assert.equal(await page.locator('#islTopology option[value="brick_a"]').evaluate(option=>option.disabled),true);
  assert.equal(await page.locator('#islTopology option[value="brick_b"]').evaluate(option=>option.disabled),true);
  await page.locator('#failRouteLink').click();
  assert.equal((await snapshot()).failures.links.length,1);
  const failedLink=(await snapshot()).failures.links[0];
  const [a,b]=failedLink.split(':').map(Number);
  assert.equal(await page.evaluate(({a,b})=>window.leopathLive.viewer.entities.getById(`isl-${a}-${b}`).polyline.material.getType(),{a,b}),'PolylineDash');
  await verify();
  assert.equal((await snapshot()).seconds,baseline.seconds,'Failure injection must preserve time');
  assert.equal((await snapshot()).playing,false,'Failure injection must preserve pause state');
  await page.locator('#allowRepair').uncheck();await verify();await page.locator('#allowRepair').check();await verify();
  await page.screenshot({path:path.join(output,'viewer-live-link-failure.png')});
  await page.locator('#failureList button').click();
  assert.deepEqual((await snapshot()).route,baseline.route,'Individual restoration must restore the baseline route');
  await page.locator('#failRouteSatellite').click();
  const node=(await snapshot()).failures.satellites[0];await verify();
  assert.equal(await page.evaluate(id=>window.leopathLive.viewer.entities.getById(`sat-telesat-${id}`).point.pixelSize.getValue(),node),10);
  assert.ok((await snapshot()).graph.activeEdges.every(e=>!e.includes(node)),'Offline satellites must lose every incident ISL');
  await page.screenshot({path:path.join(output,'viewer-live-satellite-failure.png')});
  await page.locator('#restoreFailures').click();
  await page.locator('#isolateIngress').click();
  assert.equal((await verify()).reason,'partition');
  assert.equal(await page.locator('#routeDelivery').innerText(),'No path across the live network');
  await page.screenshot({path:path.join(output,'viewer-live-partition.png')});
  await page.locator('#restoreFailures').click();
  assert.deepEqual((await snapshot()).route,baseline.route);
  await page.evaluate(()=>window.leopathLive.failSatellite(window.leopathLive.route.source));
  assert.notEqual((await verify()).source,baseline.route.source,'Attachments must change when ingress goes offline');
  await page.locator('#livePlay').click();await page.waitForTimeout(700);
  assert.ok((await snapshot()).seconds>baseline.seconds);
  assert.equal((await snapshot()).failures.satellites.length,1,'Failures must persist through orbital motion');
  await verify();await page.locator('#livePlay').click();await page.locator('#restoreFailures').click();
  await page.evaluate(()=>{const v=window.leopathLive.viewer;v.selectedEntity=v.entities.getById(`sat-telesat-${window.leopathLive.route.topo[1]}`);});
  await page.locator('.manual-failure summary').click();
  assert.equal(await page.locator('#failSelectedSatellite').isDisabled(),false);
  await page.locator('#failSelectedSatellite').click();await verify();await page.locator('#restoreFailures').click();
  await page.locator('#linkFailureA').fill('0');await page.locator('#linkFailureB').fill('100');
  await page.locator('#linkFailureForm button').click();
  assert.equal((await snapshot()).failures.links.length,0);
  assert.equal(await page.locator('#failureStatus').innerText(),'Those satellites have no ISL in this topology.');
  for(const [shell,topologies]of [['starlink',['brick_a','brick_b']],['kuiper',['brick_a','brick_b']],['oneweb',['brick_a']]]){
    await page.selectOption('#constellationSelect',shell);
    await page.waitForFunction(shell=>window.leopathLive?.ready&&window.leopathLive.viewer.entities.values.some(e=>e.id.startsWith(`sat-${shell}-`)),shell);
    for(const topology of topologies){
      await page.selectOption('#islTopology',topology);
      const healthy=await snapshot();
      assert.ok(healthy.graph.edges.length>0);assert.equal(healthy.graph.topology,topology);
      for(const seconds of [0,900,3600]){
        await page.evaluate(seconds=>{const v=window.leopathLive.viewer;v.clock.currentTime=Cesium.JulianDate.addSeconds(v.clock.startTime,seconds,new Cesium.JulianDate());},seconds);
        assert.equal((await verify()).reason,'delivered',`${shell}/${topology} must route its default flow`);
      }
      await page.locator('#failRouteLink').click();await verify();
      await page.locator('#failRouteSatellite').click();await verify();
      await page.locator('#restoreFailures').click();await verify();
      await page.locator('#isolateIngress').click();assert.equal((await verify()).reason,'partition');
      await page.locator('#restoreFailures').click();
      console.log(`PASS ${shell}/${topology}: live paths, link/satellite failures, partition and restoration`);
    }
    if(shell==='oneweb')assert.equal(await page.locator('#islTopology option[value="brick_b"]').evaluate(option=>option.disabled),true);
  }
  await page.selectOption('#constellationSelect','kuiper');await page.waitForFunction(()=>window.leopathLive?.ready&&document.querySelector('#shellLabel').textContent.startsWith('34 × 34'));
  await page.selectOption('#islTopology','brick_a');await page.locator('#failRouteLink').click();
  await page.screenshot({path:path.join(output,'viewer-live-brick-failure.png')});
  await page.setViewportSize({width:390,height:844});await page.locator('#hidePanel').click();
  assert.equal(await page.locator('.failure-card').isVisible(),true);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
  await page.screenshot({path:path.join(output,'viewer-live-failure-mobile.png')});
  await page.locator('#restoreFailures').click();
  await page.setViewportSize({width:1440,height:900});
  await page.locator("#showPanel").click();
  await page.selectOption('#islTopology','grid');
  assert.deepEqual((await snapshot()).failures,{links:[],satellites:[]},'Wiring changes reset failures');
  assert.deepEqual(errors,[]);
  console.log(`PASS ${checks} live network/route checks, controls, restoration, playback persistence, invalid input and mobile failure panel; no JavaScript exceptions`);
  await browser.close();
})().catch(error=>{console.error(error);process.exit(1)});
