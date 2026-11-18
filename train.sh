#!/usr/bin/env bash
# Train the datafix recipe with wrist augmentation disabled.
# Resources, recipe and output directory are configured in base.json.
# Supports --dry-run, --check-models, --check and --train.* / --data.* overrides.
# Configure the launch topology for the available devices before training.
set -euo pipefail

_bundle="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$_bundle"
_project="$(python3 -B -c 'from submit_paths import path; print(path("vla_project"))')"
_recipe="$(python3 -B -c 'from submit_paths import path; print(path("training_recipe"))')"

exec bash "$_project/train_multinode.sh" --config "$_recipe" "$@"
