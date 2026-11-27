#!/usr/bin/env bash
# Full temporal-overlap mirror, never the legacy cross-client-only joint suite.
set -euo pipefail
cd "$(dirname "$0")"
export TASK_IL_SUITE=joint
exec bash run_all_task_il.sh "$@"
