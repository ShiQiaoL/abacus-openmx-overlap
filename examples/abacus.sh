#!/usr/bin/env bash
set -euo pipefail

direct-overlap DATA_DIR ABACUS_ORB_DIR \
  --basis-code abacus \
  --output-dir OUT_DIR \
  --ecut 100
