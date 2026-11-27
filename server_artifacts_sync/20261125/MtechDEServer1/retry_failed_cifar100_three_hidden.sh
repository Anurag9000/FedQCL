#!/usr/bin/env bash
# Retry the two failed three-client-hidden epoch ablations in isolated workdirs.
set -euo pipefail
REPO_DIR=/home/anurag24092/Projects/FedQCL_Task
PARENT_ROOT=/home/anurag24092/Projects/FedQCL_Task/campaigns/cifar100_12_local_epoch_runs_20261122_012754
STAMP="$(date +%Y%m%d_%H%M%S)"
RUNS_ROOT="$PARENT_ROOT/retries_$STAMP"
DATA_PATH="$REPO_DIR/dataset/CIFAR100_NONIID_alpha10.0_clients5_tasks5"
PYTHON_BIN=/home/anurag24092/Projects/Correlation-Aware-FCL/.venv/bin/python
"$PYTHON_BIN" -c 'import torch, torchvision, h5py, wandb; assert torch.cuda.is_available(); print("CUDA preflight:", torch.cuda.get_device_name(0))'
mkdir -p "$RUNS_ROOT/logs"
printf 'created_at=%s\nparent_campaign=%s\nretry_scope=3_off_epochs_1,3_off_epochs_2\n' "$(date --iso-8601=seconds)" "$PARENT_ROOT" > "$RUNS_ROOT/manifest.txt"
names=()
pids=()
launch_one() {
  local epochs="$1"
  local name="3_off_epochs_${epochs}_retry1"
  local goal="cifar100_three_hidden_e${epochs}_retry1"
  local run_dir="$RUNS_ROOT/$name"
  mkdir -p "$run_dir/system" "$run_dir/results/home/anurag24092/Projects/FedQCL_Task/dataset"
  (
    cd "$run_dir/system"
    export PYTHONPYCACHEPREFIX="$run_dir/pycache"
    exec "$PYTHON_BIN" "$REPO_DIR/system/main.py" \
      -data "$DATA_PATH" -m resnet18 -algo FedAvg \
      -tasks 5 -nb 20 -nc 5 -jr 1.0 \
      -gr 49 -lbs 13 -lr 0.005 -ls "$epochs" \
      -V 200 -delta 1 -memory_size 500 -memory_batch_size 13 \
      -seed 42 -t 1 -go "$goal" --hidden_clients_per_round 3
  ) > "$RUNS_ROOT/logs/$name.log" 2>&1 &
  names+=("$name")
  pids+=("$!")
  printf '%s\t%s\n' "$name" "${pids[-1]}" >> "$RUNS_ROOT/manifest.txt"
  echo "Launched $name (PID ${pids[-1]})"
}
launch_one 1
launch_one 2
failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    printf '%s\tcompleted\n' "${names[$i]}" >> "$RUNS_ROOT/status.tsv"
    echo "Completed ${names[$i]}"
  else
    rc=$?
    printf '%s\tfailed\texit=%s\n' "${names[$i]}" "$rc" >> "$RUNS_ROOT/status.tsv"
    echo "Failed ${names[$i]} (exit $rc)" >&2
    failed=1
  fi
done
printf 'finished_at=%s\n' "$(date --iso-8601=seconds)" >> "$RUNS_ROOT/manifest.txt"
exit "$failed"
