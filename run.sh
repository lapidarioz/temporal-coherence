#!/usr/bin/env bash

set -euo pipefail

repository_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

: "${TEMPORAL_COHERENCE_DATA_DIR:?Set TEMPORAL_COHERENCE_DATA_DIR to your private dataset directory}"

cd -- "$repository_dir"
docker compose --file compose.yaml up
