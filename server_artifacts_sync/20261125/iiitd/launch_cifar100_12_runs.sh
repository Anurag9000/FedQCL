#!/usr/bin/env bash
# Launch the 12 CIFAR-100 FedQCL configurations: four visibility modes x three local epoch counts.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUNS_ROOT="${RUNS_ROOT:-$REPO_DIR/campaigns/cifar100_12_local_epoch_runs_$STAMP}"
DATA_PATH="../dataset/CIFAR100_NONIID_alpha10.0_clients5_tasks5"
# The host's default `python` is Python 2. Use the verified Python 3.12 CUDA environment.
PYTHON_BIN="${PYTHON_BIN:-/home/anurag24092/Projects/Correlation-Aware-FCL/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN (set PYTHON_BIN to the project's CUDA Python)" >&2
  exit 2
fi
if ! "$PYTHON_BIN" -c 'import sys, h5py, numpy, torchvision, wandb, torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print("Python", sys.version.split()[0], "| PyTorch", torch.__version__, "| CUDA", torch.version.cuda, "| GPU", torch.cuda.get_device_name(0))'; then
  echo "Python preflight failed; no runs were started." >&2
  exit 2
fi

if [[ -e "$RUNS_ROOT" ]]; then
  echo "Refusing to overwrite existing output directory: $RUNS_ROOT" >&2
  exit 2
fi
mkdir -p "$RUNS_ROOT/logs"
printf 'campaign_root=%s\nstarted_at=%s\n' "$RUNS_ROOT" "$(date --iso-8601=seconds)" > "$RUNS_ROOT/manifest.txt"
printf 'dataset=%s\nclients=5\ntasks=5\nclasses_per_task=20\nrounds_per_task=10\nglobal_rounds=49\nlocal_epochs=1,2,10\nseed=42\nbackbone=resnet18\naggregator=FedAvg\n' >> "$RUNS_ROOT/manifest.txt"

names=()
pids=()
launch_one() {
  local visibility="$1" epochs="$2" goal="$3" wave="$4"
  local name="${visibility}_epochs_${epochs}"
  local run_dir="$RUNS_ROOT/$name"
  mkdir -p "$run_dir/system" "$run_dir/results"
  ln -s "$REPO_DIR/dataset" "$run_dir/dataset"
  local -a visibility_args=()
  case "$visibility" in
    full) ;;
    1_off) visibility_args=(--hide_one_client) ;;
    2_off) visibility_args=(--hidden_clients_per_round 2) ;;
    3_off) visibility_args=(--hidden_clients_per_round 3) ;;
    *) echo "Unknown visibility mode: $visibility" >&2; return 2 ;;
  esac
  (
    cd "$run_dir/system"
    export PYTHONPYCACHEPREFIX="$run_dir/pycache"
    exec "$PYTHON_BIN" "$REPO_DIR/system/main.py" \
      -data "$DATA_PATH" -m resnet18 -algo FedAvg \
      -tasks 5 -nb 20 -nc 5 -jr 1.0 \
      -gr 49 -lbs 13 -lr 0.005 -ls "$epochs" \
      -V 200 -delta 1 -memory_size 500 -memory_batch_size 13 \
      -seed 42 -t 1 -go "$goal" "${visibility_args[@]}"
  ) > "$RUNS_ROOT/logs/$name.log" 2>&1 &
  local pid=$!
  names+=("$name")
  pids+=("$pid")
  printf '%s\t%s\t%s\t%s\n' "wave_${wave}" "$name" "$pid" "$epochs" >> "$RUNS_ROOT/manifest.txt"
  echo "Launched $name (PID $pid)"
}

overall_failed=0
wait_wave() {
  local wave="$1"
  echo "Waiting for all six jobs in wave $wave to finish"
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      printf 'wave_%s\t%s\tcompleted\n' "$wave" "${names[$i]}" >> "$RUNS_ROOT/status.tsv"
      echo "Completed ${names[$i]}"
    else
      rc=$?
      printf 'wave_%s\t%s\tfailed\texit=%s\n' "$wave" "${names[$i]}" "$rc" >> "$RUNS_ROOT/status.tsv"
      echo "Failed ${names[$i]} (exit $rc)" >&2
      overall_failed=1
    fi
  done
  names=()
  pids=()
  printf 'wave_%s_finished_at=%s\n' "$wave" "$(date --iso-8601=seconds)" >> "$RUNS_ROOT/manifest.txt"
}

# Wave 1: six runs, balanced client participation workload (full + 3 hidden).
for epochs in 1 2 10; do
  launch_one full "$epochs" "cifar100_full_e${epochs}" 1
  launch_one 3_off "$epochs" "cifar100_three_hidden_e${epochs}" 1
done
printf 'wave_1_started_at=%s\n' "$(date --iso-8601=seconds)" >> "$RUNS_ROOT/manifest.txt"
wait_wave 1

# Wave 2 starts only after all six wave-1 processes have exited.
for epochs in 1 2 10; do
  launch_one 1_off "$epochs" "cifar100_one_hidden_e${epochs}" 2
  launch_one 2_off "$epochs" "cifar100_two_hidden_e${epochs}" 2
done
printf 'wave_2_started_at=%s\n' "$(date --iso-8601=seconds)" >> "$RUNS_ROOT/manifest.txt"
wait_wave 2
printf 'finished_at=%s\n' "$(date --iso-8601=seconds)" >> "$RUNS_ROOT/manifest.txt"
printf 'RUNS_ROOT=%s\n' "$RUNS_ROOT" > "$RUNS_ROOT/launcher.env"
exit "$overall_failed"
