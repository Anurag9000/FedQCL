#!/usr/bin/env bash
# CIFAR-100 setup with two random clients hidden in both directions each round.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run.sh" cifar100 --hidden_clients_per_round 2 -go cifar100_two_hidden "$@"
