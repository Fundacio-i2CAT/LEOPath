#!/usr/bin/env bash
set -euo pipefail

bash scripts/viewer/build-assets.sh
zensical build --clean
mkdir -p site/cesium
cp -R docs/cesium/. site/cesium/

# Replace only build output; source examples and tracked configurations stay intact.
if [[ "${LEOPATH_DATASET_RELEASE:-}" == "1" ]]; then
  python scripts/fetch_viewer_release.py \
    --tag "${LEOPATH_DATASET_TAG:-}" --commit "${LEOPATH_DATASET_COMMIT:-}" \
    --target site/cesium
  python scripts/verify_viewer_replays.py site/cesium/replays
elif [[ -n "${LEOPATH_DATASET_DIR:-}" ]]; then
  python scripts/viewer_dataset.py install "$LEOPATH_DATASET_DIR/viewer" --target site/cesium --preview
  python scripts/verify_viewer_replays.py site/cesium/replays
fi
