#! /bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

MODE="parallel"
GPU_LIST="0,1"
GPUS_PER_JOB=1
MAX_RETRIES=0
CONTINUE_ON_ERROR=0
DRY_RUN=0
LOG_DIR=""

SCRIPTS=(
  "scripts/qwen/sft/sft_qwen3.5_2B_test.sh"
  "scripts/qwen/sft/sft_qwen3.5_4B_test.sh"
)

SCRIPT_ARGS=()

usage() {
  cat <<'EOF'
Usage: ./test_run_finetuning_qwen3.5.sh [options] [-- extra-args-for-child-scripts]

Run the fixed Qwen3.5 SFT jobs with the same queue style as running.sh.

Options:
  --mode <parallel|sequential>   Queue jobs by GPU chunks or run one-by-one. Default: parallel
  --gpus <list>                  Comma-separated GPU ids. Default: 0,1
  --gpus-per-job <n>             GPUs used by each script. Default: 1
  --max-retries <n>              Retry failed jobs up to n times. Default: 0
  --log-dir <path>               Log directory. Default: ./run_logs/qwen3.5_<timestamp>
  --continue-on-error            Keep scheduling after retries are exhausted.
  --dry-run                      Print what would run without launching jobs.
  -h, --help                     Show this message.

Examples:
  ./run_finetuning_qwen3.5.sh
  ./run_finetuning_qwen3.5.sh --gpus 2,3
  ./run_finetuning_qwen3.5.sh --mode sequential --gpus 0
  ./run_finetuning_qwen3.5.sh -- --epochs 3
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
    --max-retries)
      MAX_RETRIES="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --continue-on-error)
      CONTINUE_ON_ERROR=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      SCRIPT_ARGS=("$@")
      break
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

if ! [[ "${MAX_RETRIES}" =~ ^[0-9]+$ ]]; then
  echo "--max-retries must be a non-negative integer" >&2
  exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
if [[ -z "${LOG_DIR}" ]]; then
  LOG_DIR="${ROOT_DIR}/run_logs/qwen3.5_${timestamp}"
fi

JOB_LOG_DIR="${LOG_DIR}/jobs"
mkdir -p "${JOB_LOG_DIR}"

RUN_LOG="${LOG_DIR}/run.log"
SUCCESS_LOG="${LOG_DIR}/success.log"
FAIL_LOG="${LOG_DIR}/failed.log"
RETRY_LOG="${LOG_DIR}/retry.log"

: > "${RUN_LOG}"
: > "${SUCCESS_LOG}"
: > "${FAIL_LOG}"
: > "${RETRY_LOG}"

append_log() {
  local log_file="$1"
  local message="$2"
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${message}" >> "${log_file}"
}

script_id() {
  local rel="$1"
  rel="${rel//\//__}"
  rel="${rel//\\/__}"
  printf '%s' "${rel%.sh}"
}

IFS=',' read -r -a ALL_GPUS <<< "${GPU_LIST}"
GPU_COUNT="${#ALL_GPUS[@]}"
if [[ "${GPU_COUNT}" -lt "${GPUS_PER_JOB}" ]]; then
  echo "Need at least ${GPUS_PER_JOB} GPUs, but got ${GPU_COUNT}: ${GPU_LIST}" >&2
  exit 1
fi

chunks=()
for ((i=0; i<GPU_COUNT; i+=GPUS_PER_JOB)); do
  if (( i + GPUS_PER_JOB <= GPU_COUNT )); then
    chunk="$(IFS=,; echo "${ALL_GPUS[*]:i:GPUS_PER_JOB}")"
    chunks+=("${chunk}")
  fi
done

if [[ "${#chunks[@]}" -eq 0 ]]; then
  echo "Failed to form GPU chunks from ${GPU_LIST}" >&2
  exit 1
fi

if (( GPU_COUNT % GPUS_PER_JOB != 0 )); then
  leftover=$((GPU_COUNT % GPUS_PER_JOB))
  echo "Warning: ${leftover} GPU(s) will be idle because ${GPU_COUNT} is not divisible by ${GPUS_PER_JOB}." >&2
  append_log "${RUN_LOG}" "Warning: ${leftover} GPU(s) idle because ${GPU_COUNT} is not divisible by ${GPUS_PER_JOB}."
fi

append_log "${RUN_LOG}" "Started run: mode=${MODE}, gpus=${GPU_LIST}, gpus_per_job=${GPUS_PER_JOB}, max_retries=${MAX_RETRIES}, continue_on_error=${CONTINUE_ON_ERROR}"
append_log "${RUN_LOG}" "Logs directory: ${LOG_DIR}"
append_log "${RUN_LOG}" "Scheduled scripts: ${#SCRIPTS[@]}"

launch_counter=0
failures=0
base_port=29600

run_script_once() {
  local script="$1"
  local gpu_chunk="$2"
  local attempt="$3"
  local port=$((base_port + launch_counter))
  local job_id
  local job_log

  job_id="$(script_id "${script}")"
  job_log="${JOB_LOG_DIR}/${job_id}.attempt${attempt}.log"
  launch_counter=$((launch_counter + 1))

  echo "[launch] ${script}"
  echo "         GPUs: ${gpu_chunk} | port: ${port} | attempt: ${attempt}"
  echo "         log : ${job_log}"
  append_log "${RUN_LOG}" "DISPATCH attempt=${attempt} gpus=${gpu_chunk} port=${port} script=${script} log=${job_log}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi

  RUN_GPUS="${gpu_chunk}" RUN_MASTER_PORT="${port}" bash "${script}" "${SCRIPT_ARGS[@]}" > "${job_log}" 2>&1
}

if [[ "${MODE}" == "sequential" ]]; then
  gpu_chunk="${chunks[0]}"
  for script in "${SCRIPTS[@]}"; do
    attempt=1
    while true; do
      if run_script_once "${script}" "${gpu_chunk}" "${attempt}"; then
        append_log "${RUN_LOG}" "DONE attempt=${attempt} gpus=${gpu_chunk} script=${script}"
        append_log "${SUCCESS_LOG}" "attempt=${attempt} gpus=${gpu_chunk} script=${script}"
        break
      fi

      append_log "${RUN_LOG}" "FAIL attempt=${attempt} gpus=${gpu_chunk} script=${script}"
      if [[ "${attempt}" -le "${MAX_RETRIES}" ]]; then
        attempt=$((attempt + 1))
        append_log "${RETRY_LOG}" "next_attempt=${attempt} script=${script}"
        continue
      fi

      failures=$((failures + 1))
      append_log "${FAIL_LOG}" "attempt=${attempt} gpus=${gpu_chunk} script=${script}"
      if [[ "${CONTINUE_ON_ERROR}" -ne 1 ]]; then
        append_log "${RUN_LOG}" "Stopped sequential mode after exhausted retries for ${script}."
        echo "Stopping after failure: ${script}" >&2
        exit 1
      fi
      break
    done
  done
else
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    for idx in "${!SCRIPTS[@]}"; do
      chunk_idx=$((idx % ${#chunks[@]}))
      run_script_once "${SCRIPTS[$idx]}" "${chunks[$chunk_idx]}" 1
    done
  else
    declare -a JOB_ATTEMPTS=()
    declare -a PENDING_RUNS=()
    declare -a ACTIVE_PIDS=()
    declare -a CHUNK_BUSY=()
    declare -A PID_TO_RUN_IDX=()
    declare -A PID_TO_CHUNK_IDX=()
    declare -A PID_TO_ATTEMPT=()
    declare -A PID_TO_LOG=()

    for idx in "${!SCRIPTS[@]}"; do
      JOB_ATTEMPTS[idx]=0
      PENDING_RUNS+=("${idx}")
    done

    for idx in "${!chunks[@]}"; do
      CHUNK_BUSY[idx]=0
    done

    queue_head=0
    stop_scheduling=0

    remove_active_pid() {
      local target_pid="$1"
      local next_active=()
      local active_pid

      for active_pid in "${ACTIVE_PIDS[@]}"; do
        if [[ "${active_pid}" != "${target_pid}" ]]; then
          next_active+=("${active_pid}")
        fi
      done
      ACTIVE_PIDS=("${next_active[@]}")
    }

    start_job_on_chunk() {
      local run_idx="$1"
      local chunk_idx="$2"
      local script="${SCRIPTS[$run_idx]}"
      local attempt=$((JOB_ATTEMPTS[run_idx] + 1))
      local port=$((base_port + launch_counter))
      local job_id
      local job_log
      local pid

      JOB_ATTEMPTS[run_idx]="${attempt}"
      launch_counter=$((launch_counter + 1))
      job_id="$(script_id "${script}")"
      job_log="${JOB_LOG_DIR}/${job_id}.attempt${attempt}.log"

      echo "[queue] dispatch ${script}"
      echo "        GPUs: ${chunks[$chunk_idx]} | port: ${port} | attempt: ${attempt}"
      echo "        log : ${job_log}"
      append_log "${RUN_LOG}" "DISPATCH attempt=${attempt} gpus=${chunks[$chunk_idx]} port=${port} script=${script} log=${job_log}"

      (
        RUN_GPUS="${chunks[$chunk_idx]}" RUN_MASTER_PORT="${port}" bash "${script}" "${SCRIPT_ARGS[@]}" > "${job_log}" 2>&1
      ) &
      pid="$!"

      ACTIVE_PIDS+=("${pid}")
      CHUNK_BUSY[chunk_idx]=1
      PID_TO_RUN_IDX["${pid}"]="${run_idx}"
      PID_TO_CHUNK_IDX["${pid}"]="${chunk_idx}"
      PID_TO_ATTEMPT["${pid}"]="${attempt}"
      PID_TO_LOG["${pid}"]="${job_log}"
    }

    schedule_ready_jobs() {
      local chunk_idx
      while [[ "${stop_scheduling}" -eq 0 ]]; do
        chunk_idx=""
        for idx in "${!chunks[@]}"; do
          if [[ "${CHUNK_BUSY[idx]}" -eq 0 ]]; then
            chunk_idx="${idx}"
            break
          fi
        done

        if [[ -z "${chunk_idx}" || "${queue_head}" -ge "${#PENDING_RUNS[@]}" ]]; then
          break
        fi

        start_job_on_chunk "${PENDING_RUNS[$queue_head]}" "${chunk_idx}"
        queue_head=$((queue_head + 1))
      done
    }

    schedule_ready_jobs

    while [[ "${#ACTIVE_PIDS[@]}" -gt 0 ]]; do
      finished_pid=""
      if wait -n -p finished_pid; then
        wait_status=0
      else
        wait_status=$?
      fi

      run_idx="${PID_TO_RUN_IDX[${finished_pid}]}"
      chunk_idx="${PID_TO_CHUNK_IDX[${finished_pid}]}"
      attempt="${PID_TO_ATTEMPT[${finished_pid}]}"
      script="${SCRIPTS[$run_idx]}"
      job_log="${PID_TO_LOG[${finished_pid}]}"

      CHUNK_BUSY[chunk_idx]=0
      remove_active_pid "${finished_pid}"
      unset "PID_TO_RUN_IDX[$finished_pid]"
      unset "PID_TO_CHUNK_IDX[$finished_pid]"
      unset "PID_TO_ATTEMPT[$finished_pid]"
      unset "PID_TO_LOG[$finished_pid]"

      if [[ "${wait_status}" -eq 0 ]]; then
        echo "[done] ${script}"
        append_log "${RUN_LOG}" "DONE attempt=${attempt} gpus=${chunks[$chunk_idx]} script=${script} log=${job_log}"
        append_log "${SUCCESS_LOG}" "attempt=${attempt} gpus=${chunks[$chunk_idx]} script=${script} log=${job_log}"
      else
        echo "[fail] ${script} exited with ${wait_status}" >&2
        append_log "${RUN_LOG}" "FAIL attempt=${attempt} exit_code=${wait_status} gpus=${chunks[$chunk_idx]} script=${script} log=${job_log}"
        if [[ "${attempt}" -le "${MAX_RETRIES}" ]]; then
          next_attempt=$((attempt + 1))
          append_log "${RETRY_LOG}" "next_attempt=${next_attempt} script=${script}"
          PENDING_RUNS+=("${run_idx}")
        else
          failures=$((failures + 1))
          append_log "${FAIL_LOG}" "attempt=${attempt} exit_code=${wait_status} script=${script} log=${job_log}"
          if [[ "${CONTINUE_ON_ERROR}" -ne 1 ]]; then
            stop_scheduling=1
            append_log "${RUN_LOG}" "Stopped scheduling after exhausted retries for ${script}."
          fi
        fi
      fi

      schedule_ready_jobs
    done

    if [[ "${stop_scheduling}" -eq 1 && "${queue_head}" -lt "${#PENDING_RUNS[@]}" ]]; then
      skipped_jobs=$(( ${#PENDING_RUNS[@]} - queue_head ))
      append_log "${RUN_LOG}" "Stopped with ${skipped_jobs} queued script(s) not started."
      echo "Stopped with ${skipped_jobs} queued script(s) not started." >&2
    fi
  fi
fi

if [[ "${failures}" -gt 0 ]]; then
  append_log "${RUN_LOG}" "Finished with ${failures} failed script(s)."
  echo "Finished with ${failures} failed script(s)." >&2
  exit 1
fi

append_log "${RUN_LOG}" "All requested scripts finished successfully."
echo "All requested scripts finished successfully."
