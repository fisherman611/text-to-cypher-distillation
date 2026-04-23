#! /bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_ROOT="${ROOT_DIR}/updated_span_scripts/qwen"

MODE="parallel"
GPU_LIST="0,1,2,3,4,5,6,7"
GPUS_PER_JOB=2
FILTER=""
DRY_RUN=0
CONTINUE_ON_ERROR=0

usage() {
  cat <<'EOF'
Usage: ./running.sh [options]

Options:
  --mode <parallel|sequential>   Run jobs in parallel batches or one by one. Default: parallel
  --gpus <list>                  Comma-separated GPU ids to use. Default: 0,1,2,3,4,5,6,7
  --gpus-per-job <n>             Number of GPUs assigned to each script. Default: 2
  --filter <pattern>             Only run scripts whose path contains this substring.
  --dry-run                      Print commands without launching them.
  --continue-on-error            Continue with next batch/script even if one fails.
  -h, --help                     Show this message.

Examples:
  ./running.sh
  ./running.sh --mode sequential --gpus 0,1 --filter fdd
  ./running.sh --mode parallel --gpus 0,1,2,3,4,5,6,7 --gpus-per-job 2
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --gpus)
      GPU_LIST="$2"
      shift 2
      ;;
    --gpus-per-job)
      GPUS_PER_JOB="$2"
      shift 2
      ;;
    --filter)
      FILTER="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --continue-on-error)
      CONTINUE_ON_ERROR=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "${MODE}" != "parallel" && "${MODE}" != "sequential" ]]; then
  echo "--mode must be 'parallel' or 'sequential'" >&2
  exit 1
fi

if ! [[ "${GPUS_PER_JOB}" =~ ^[0-9]+$ ]] || [[ "${GPUS_PER_JOB}" -le 0 ]]; then
  echo "--gpus-per-job must be a positive integer" >&2
  exit 1
fi

IFS=',' read -r -a ALL_GPUS <<< "${GPU_LIST}"
GPU_COUNT="${#ALL_GPUS[@]}"
if [[ "${GPU_COUNT}" -lt "${GPUS_PER_JOB}" ]]; then
  echo "Need at least ${GPUS_PER_JOB} GPUs, but got ${GPU_COUNT}: ${GPU_LIST}" >&2
  exit 1
fi

mapfile -t SCRIPTS < <(find "${SCRIPT_ROOT}" -type f -name "*.sh" | sort)
if [[ -n "${FILTER}" ]]; then
  FILTERED=()
  for script in "${SCRIPTS[@]}"; do
    if [[ "${script}" == *"${FILTER}"* ]]; then
      FILTERED+=("${script}")
    fi
  done
  SCRIPTS=("${FILTERED[@]}")
fi

if [[ "${#SCRIPTS[@]}" -eq 0 ]]; then
  echo "No scripts found under ${SCRIPT_ROOT} matching filter '${FILTER}'" >&2
  exit 1
fi

chunks=()
chunk_size="${GPUS_PER_JOB}"
for ((i=0; i<GPU_COUNT; i+=chunk_size)); do
  if (( i + chunk_size <= GPU_COUNT )); then
    chunk="$(IFS=,; echo "${ALL_GPUS[*]:i:chunk_size}")"
    chunks+=("${chunk}")
  fi
done

if [[ "${#chunks[@]}" -eq 0 ]]; then
  echo "Failed to form GPU chunks from ${GPU_LIST}" >&2
  exit 1
fi

launch_job() {
  local script="$1"
  local gpu_chunk="$2"
  local port="$3"
  local rel_script="${script#${ROOT_DIR}/}"
  local cmd="RUN_GPUS=${gpu_chunk} RUN_MASTER_PORT=${port} bash ${script}"

  echo "[launch] ${rel_script}"
  echo "         GPUs: ${gpu_chunk} | port: ${port}"
  echo "         cmd : ${cmd}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi

  RUN_GPUS="${gpu_chunk}" RUN_MASTER_PORT="${port}" bash "${script}"
}

failures=0
base_port=29600

if [[ "${MODE}" == "sequential" ]]; then
  gpu_chunk="${chunks[0]}"
  for idx in "${!SCRIPTS[@]}"; do
    port=$((base_port + idx))
    if ! launch_job "${SCRIPTS[$idx]}" "${gpu_chunk}" "${port}"; then
      failures=$((failures + 1))
      if [[ "${CONTINUE_ON_ERROR}" -ne 1 ]]; then
        echo "Stopping after failure." >&2
        exit 1
      fi
    fi
  done
else
  batch_size="${#chunks[@]}"
  for ((start=0; start<${#SCRIPTS[@]}; start+=batch_size)); do
    pids=()
    batch_scripts=("${SCRIPTS[@]:start:batch_size}")
    for offset in "${!batch_scripts[@]}"; do
      script="${batch_scripts[$offset]}"
      gpu_chunk="${chunks[$offset]}"
      port=$((base_port + start + offset))

      if [[ "${DRY_RUN}" -eq 1 ]]; then
        launch_job "${script}" "${gpu_chunk}" "${port}"
        continue
      fi

      (
        launch_job "${script}" "${gpu_chunk}" "${port}"
      ) &
      pids+=("$!")
    done

    if [[ "${DRY_RUN}" -eq 1 ]]; then
      continue
    fi

    batch_failed=0
    for pid in "${pids[@]}"; do
      if ! wait "${pid}"; then
        batch_failed=1
      fi
    done

    if [[ "${batch_failed}" -eq 1 ]]; then
      failures=$((failures + 1))
      if [[ "${CONTINUE_ON_ERROR}" -ne 1 ]]; then
        echo "A batch failed. Stopping." >&2
        exit 1
      fi
    fi
  done
fi

if [[ "${failures}" -gt 0 ]]; then
  echo "Finished with ${failures} failed run group(s)." >&2
  exit 1
fi

echo "All requested scripts finished successfully."
