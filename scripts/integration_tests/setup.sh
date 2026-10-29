#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
# Install into this checkout only. This script does not run calculations.
uv sync --locked --extra cuda12 --group dev
mkdir -p integration_test_results
