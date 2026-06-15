#!/usr/bin/env bash
set -euo pipefail

zensical build --clean

mkdir -p site/cesium/data
cp -R docs/cesium/. site/cesium/

cp tles_starlink_550_sgp.txt.tmp site/cesium/data/tles_starlink_550_sgp.txt
cp tles_kuiper_synth.txt site/cesium/data/tles_kuiper_synth.txt
cp tles_oneweb_synth.txt site/cesium/data/tles_oneweb_synth.txt
cp tles_telesat_synth.txt site/cesium/data/tles_telesat_synth.txt
cp tles_dense_leo_synth.txt site/cesium/data/tles_dense_leo_synth.txt
