#!/usr/bin/env bash
set -euo pipefail

direct-overlap DATA_DIR OPENMX_PAO_DIR \
  --basis-code openmx \
  --openmx-basis Au=Au7.0-s2p2d2f1 \
  --openmx-basis Mo=Mo7.0-s3p2d2f1 \
  --openmx-basis S=S7.0-s3p3d2f1 \
  --output-dir OUT_DIR \
  --ecut 100
