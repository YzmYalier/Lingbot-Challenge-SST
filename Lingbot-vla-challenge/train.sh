#!/usr/bin/env bash
# Compatibility entrypoint for train.sh tasks/vla/train_lingbotvla.py recipe.yaml.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == *train_lingbotvla.py ]]; then shift; fi
if [[ "${1:-}" == *.yaml || "${1:-}" == *.yml ]]; then
    export YAML_FILE="$1"
    shift
fi
exec bash "$SCRIPT_DIR/train_multinode.sh" "$@"
