#!/usr/bin/env bash
set -euo pipefail
viewer_root="$(cd "$(dirname "$0")/../.." && pwd)"
viewer_tools="$viewer_root/scripts/viewer"
npm ci --prefix "$viewer_tools" --ignore-scripts
mkdir -p "$viewer_root/docs/cesium/brand"
cp "$viewer_root/docs/assets/brand/"*.png "$viewer_root/docs/cesium/brand/"
mkdir -p "$viewer_root/docs/cesium/vendor"
cp -R "$viewer_tools/node_modules/cesium/Build/Cesium" "$viewer_root/docs/cesium/vendor/"
cp "$viewer_tools/node_modules/satellite.js/dist/satellite.min.js" "$viewer_root/docs/cesium/vendor/satellite.min.js"
cp "$viewer_tools/node_modules/cesium/LICENSE.md" "$viewer_root/docs/cesium/vendor/Cesium-LICENSE.md"
cp "$viewer_tools/node_modules/satellite.js/LICENSE.md" "$viewer_root/docs/cesium/vendor/satellite-LICENSE.md"
mkdir -p "$viewer_root/docs/cesium/data"
for shell_name in telesat starlink kuiper oneweb; do
  case "$shell_name" in
    starlink) tle_name="tles_starlink_550_sgp.txt" ;;
    *) tle_name="tles_${shell_name}_synth.txt" ;;
  esac
  if test -f "$viewer_root/docs/cesium/replays/${shell_name}-grid.tles.txt"; then
    cp "$viewer_root/docs/cesium/replays/${shell_name}-grid.tles.txt" "$viewer_root/docs/cesium/data/$tle_name"
  else
    cp "$viewer_root/$tle_name" "$viewer_root/docs/cesium/data/$tle_name"
  fi
done
cp "$viewer_root/tles_dense_leo_synth.txt" "$viewer_root/docs/cesium/data/tles_dense_leo_synth.txt"
