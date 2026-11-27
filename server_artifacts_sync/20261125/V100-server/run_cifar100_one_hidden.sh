#!/usr/bin/env bash
# Same CIFAR-100 experiment as ./run.sh cifar100, excluding one random client
# from local training, model upload, and global-model broadcast each round.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run.sh" cifar100 --hide_one_client -go cifar100_one_hidden "$@"
