#!/usr/bin/env bash
# Branch-specific recipe; all resources are resolved through ../base.json.
# YAML_FILE=configs/vla/robotwin/<recipe>.yaml bash train_multinode.sh
# --dry-run / --check-models / --check perform no GPU training.
set -euo pipefail
SUBMIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 -B "$SUBMIT_ROOT/tools/train.py" "$@"
