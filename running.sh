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
MAX_RETRIES=1
LOG_DIR=""
W_REL_LOSS_VALUES="0.5,0.6,0.7,0.8,0.9,1.0"
KD_RATIO_VALUES="0.7"
FDD_WEIGHT_VALUES="0.1,0.2,0.3,0.4"

usage() {
  cat <<'EOF'
Usage: ./running.sh [options]

Options:
  --mode <parallel|sequential>   Run jobs in parallel batches or one by one. Default: parallel
  --gpus <list>                  Comma-separated GPU ids to use. Default: 0,1,2,3,4,5,6,7
  --gpus-per-job <n>             Number of GPUs assigned to each script. Default: 2
  --filter <pattern>             Only run scripts whose path contains this substring.
  --max-retries <n>              Retry a failed job up to n times in parallel mode. Default: 1
  --log-dir <path>               Directory for run logs. Default: ./run_logs/<timestamp>
  --w-rel-loss-values <list>     Comma-separated sweep values for --w-rel-loss. Default: 0.5,0.6,0.7,0.8,0.9,1.0
  --kd-ratio-values <list>       Comma-separated sweep values for --kd-ratio.
  --fdd-weight-values <list>     Comma-separated sweep values for --fdd-weight.
  --dry-run                      Print commands without launching them.
  --continue-on-error            Continue with next batch/script even if one fails.
  -h, --help                     Show this message.

Examples:
  ./running.sh
  ./running.sh --mode sequential --gpus 0,1 --filter fdd
  ./running.sh --mode parallel --gpus 0,1,2,3,4,5,6,7 --gpus-per-job 2
  ./running.sh --mode parallel --gpus 0,1,2,3,4,5,6,7 --gpus-per-job 2 --max-retries 2
  ./running.sh --mode parallel --gpus 0,1,2,3,4,5,6,7 --gpus-per-job 2 --log-dir ./run_logs/qwen_sweep
  ./running.sh --filter kd
  ./running.sh --filter kd --w-rel-loss-values 0.1,0.3,1.0
  ./running.sh --filter fdd --w-rel-loss-values 0.5,1.0 --fdd-weight-values 0.1,0.2 --kd-ratio-values 0.4,0.5
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
    --max-retries)
      MAX_RETRIES="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --w-rel-loss-values)
      W_REL_LOSS_VALUES="$2"
      shift 2
      ;;
    --kd-ratio-values)
      KD_RATIO_VALUES="$2"
      shift 2
      ;;
    --fdd-weight-values)
      FDD_WEIGHT_VALUES="$2"
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

if ! [[ "${MAX_RETRIES}" =~ ^[0-9]+$ ]]; then
  echo "--max-retries must be a non-negative integer" >&2
  exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
if [[ -z "${LOG_DIR}" ]]; then
  LOG_DIR="${ROOT_DIR}/run_logs/${timestamp}"
fi
mkdir -p "${LOG_DIR}"

RUN_LOG="${LOG_DIR}/run.log"
SUCCESS_LOG="${LOG_DIR}/success.log"
FAIL_LOG="${LOG_DIR}/failed.log"
RETRY_LOG="${LOG_DIR}/retry.log"

: > "${RUN_LOG}"
: > "${SUCCESS_LOG}"
: > "${FAIL_LOG}"
: > "${RETRY_LOG}"

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

W_REL_VALUES=("")
if [[ -n "${W_REL_LOSS_VALUES}" ]]; then
  IFS=',' read -r -a W_REL_VALUES <<< "${W_REL_LOSS_VALUES}"
fi

KD_RATIO_SWEEP=("")
if [[ -n "${KD_RATIO_VALUES}" ]]; then
  IFS=',' read -r -a KD_RATIO_SWEEP <<< "${KD_RATIO_VALUES}"
fi

FDD_WEIGHT_SWEEP=("")
if [[ -n "${FDD_WEIGHT_VALUES}" ]]; then
  IFS=',' read -r -a FDD_WEIGHT_SWEEP <<< "${FDD_WEIGHT_VALUES}"
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
  local spec="${4:-}"
  local rel_script="${script#${ROOT_DIR}/}"
  local save_suffix=""
  local extra_args=()
  local cmd=""
  local pretty_overrides=()

  IFS=';' read -r -a kv_pairs <<< "${spec}"
  for kv in "${kv_pairs[@]}"; do
    [[ -z "${kv}" ]] && continue
    local key="${kv%%=*}"
    local value="${kv#*=}"
    local safe_value="${value//./p}"
    case "${key}" in
      w_rel)
        save_suffix+="_wrel${safe_value}"
        extra_args+=(--w-rel-loss "${value}")
        pretty_overrides+=("w_rel_loss=${value}")
        ;;
      kd_ratio)
        save_suffix+="_kd${safe_value}"
        extra_args+=(--kd-ratio "${value}")
        pretty_overrides+=("kd_ratio=${value}")
        ;;
      fdd_weight)
        save_suffix+="_fdd${safe_value}"
        extra_args+=(--fdd-weight "${value}")
        pretty_overrides+=("fdd_weight=${value}")
        ;;
      w_attn)
        save_suffix+="_wattn${safe_value}"
        extra_args+=(--w-attn-loss "${value}")
        pretty_overrides+=("w_attn_loss=${value}")
        ;;
      w_query)
        save_suffix+="_wquery${safe_value}"
        extra_args+=(--w-query-loss "${value}")
        pretty_overrides+=("w_query_loss=${value}")
        ;;
      w_relational)
        save_suffix+="_wrelat${safe_value}"
        extra_args+=(--w-relational-loss "${value}")
        pretty_overrides+=("w_relational_loss=${value}")
        ;;
    esac
  done

  cmd="RUN_GPUS=${gpu_chunk} RUN_MASTER_PORT=${port}"
  if [[ -n "${save_suffix}" ]]; then
    cmd+=" RUN_SAVE_SUFFIX=${save_suffix}"
  fi
  cmd+=" bash ${script}"
  if [[ "${#extra_args[@]}" -gt 0 ]]; then
    cmd+=" ${extra_args[*]}"
  fi

  echo "[launch] ${rel_script}"
  echo "         GPUs: ${gpu_chunk} | port: ${port}"
  if [[ "${#pretty_overrides[@]}" -gt 0 ]]; then
    echo "         overrides: ${pretty_overrides[*]}"
  fi
  echo "         cmd : ${cmd}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi

  RUN_GPUS="${gpu_chunk}" RUN_MASTER_PORT="${port}" RUN_SAVE_SUFFIX="${save_suffix}" bash "${script}" "${extra_args[@]}"
}

append_log() {
  local log_file="$1"
  local message="$2"
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${message}" >> "${log_file}"
}

describe_run() {
  local run_idx="$1"
  local script spec rel_script

  IFS='|' read -r script spec <<< "${RUN_SPECS[$run_idx]}"
  rel_script="${script#${ROOT_DIR}/}"
  if [[ -n "${spec}" ]]; then
    printf '%s | %s' "${rel_script}" "${spec}"
  else
    printf '%s' "${rel_script}"
  fi
}

failures=0
base_port=29600
launch_counter=0

RUN_SPECS=()
for script in "${SCRIPTS[@]}"; do
  for w_rel in "${W_REL_VALUES[@]}"; do
    for kd_ratio in "${KD_RATIO_SWEEP[@]}"; do
      for fdd_weight in "${FDD_WEIGHT_SWEEP[@]}"; do
        spec=""
        [[ -n "${w_rel}" ]] && spec+="w_rel=${w_rel};"
        [[ -n "${kd_ratio}" ]] && spec+="kd_ratio=${kd_ratio};"
        [[ -n "${fdd_weight}" ]] && spec+="fdd_weight=${fdd_weight};"
        RUN_SPECS+=("${script}|${spec}")
      done
    done
  done
done

if [[ "${#RUN_SPECS[@]}" -eq 0 ]]; then
  echo "No run specs were generated. Check your sweep values and filters." >&2
  append_log "${RUN_LOG}" "No run specs were generated. Exiting with error."
  exit 1
fi

append_log "${RUN_LOG}" "Started run: mode=${MODE}, gpus=${GPU_LIST}, gpus_per_job=${GPUS_PER_JOB}, max_retries=${MAX_RETRIES}, continue_on_error=${CONTINUE_ON_ERROR}, filter=${FILTER:-<none>}"
append_log "${RUN_LOG}" "Logs directory: ${LOG_DIR}"
append_log "${RUN_LOG}" "Total scheduled runs: ${#RUN_SPECS[@]}"

if [[ "${MODE}" == "sequential" ]]; then
  gpu_chunk="${chunks[0]}"
  for idx in "${!RUN_SPECS[@]}"; do
    port=$((base_port + launch_counter))
    launch_counter=$((launch_counter + 1))
    IFS='|' read -r script spec <<< "${RUN_SPECS[$idx]}"
    run_desc="$(describe_run "${idx}")"
    append_log "${RUN_LOG}" "DISPATCH sequential run=$((idx + 1)) gpus=${gpu_chunk} port=${port} ${run_desc}"
    if ! launch_job "${script}" "${gpu_chunk}" "${port}" "${spec}"; then
      failures=$((failures + 1))
      append_log "${RUN_LOG}" "FAIL sequential run=$((idx + 1)) port=${port} ${run_desc}"
      append_log "${FAIL_LOG}" "run=$((idx + 1)) port=${port} ${run_desc}"
      if [[ "${CONTINUE_ON_ERROR}" -ne 1 ]]; then
        echo "Stopping after failure." >&2
        append_log "${RUN_LOG}" "STOP sequential mode after first failure."
        exit 1
      fi
    else
      append_log "${RUN_LOG}" "DONE sequential run=$((idx + 1)) port=${port} ${run_desc}"
      append_log "${SUCCESS_LOG}" "run=$((idx + 1)) port=${port} ${run_desc}"
    fi
  done
else
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    for idx in "${!RUN_SPECS[@]}"; do
      IFS='|' read -r script spec <<< "${RUN_SPECS[$idx]}"
      chunk_idx=$((idx % ${#chunks[@]}))
      gpu_chunk="${chunks[$chunk_idx]}"
      port=$((base_port + launch_counter))
      launch_counter=$((launch_counter + 1))
      launch_job "${script}" "${gpu_chunk}" "${port}" "${spec}"
    done
  else
    declare -a JOB_ATTEMPTS=()
    declare -a PENDING_RUNS=()
    declare -a ACTIVE_PIDS=()
    declare -a CHUNK_BUSY=()
    declare -A PID_TO_RUN_IDX=()
    declare -A PID_TO_CHUNK_IDX=()
    declare -A PID_TO_ATTEMPT=()

    for idx in "${!RUN_SPECS[@]}"; do
      JOB_ATTEMPTS[idx]=0
      PENDING_RUNS+=("${idx}")
    done

    for idx in "${!chunks[@]}"; do
      CHUNK_BUSY[idx]=0
    done

    queue_head=0
    stop_scheduling=0

    start_job_on_chunk() {
      local run_idx="$1"
      local chunk_idx="$2"
      local attempt=$((JOB_ATTEMPTS[run_idx] + 1))
      local port=$((base_port + launch_counter))
      local script spec gpu_chunk pid run_desc

      JOB_ATTEMPTS[run_idx]="${attempt}"
      launch_counter=$((launch_counter + 1))
      IFS='|' read -r script spec <<< "${RUN_SPECS[$run_idx]}"
      gpu_chunk="${chunks[$chunk_idx]}"
      run_desc="$(describe_run "${run_idx}")"

      echo "[queue] dispatch run #$((run_idx + 1)) on GPUs ${gpu_chunk} (attempt ${attempt}/$((MAX_RETRIES + 1)))"
      append_log "${RUN_LOG}" "DISPATCH parallel run=$((run_idx + 1)) attempt=${attempt} gpus=${gpu_chunk} port=${port} ${run_desc}"
      (
        launch_job "${script}" "${gpu_chunk}" "${port}" "${spec}"
      ) &
      pid="$!"

      ACTIVE_PIDS+=("${pid}")
      CHUNK_BUSY[chunk_idx]=1
      PID_TO_RUN_IDX["${pid}"]="${run_idx}"
      PID_TO_CHUNK_IDX["${pid}"]="${chunk_idx}"
      PID_TO_ATTEMPT["${pid}"]="${attempt}"
    }

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
      CHUNK_BUSY[chunk_idx]=0
      remove_active_pid "${finished_pid}"
      unset "PID_TO_RUN_IDX[$finished_pid]"
      unset "PID_TO_CHUNK_IDX[$finished_pid]"
      unset "PID_TO_ATTEMPT[$finished_pid]"

      IFS='|' read -r script spec <<< "${RUN_SPECS[$run_idx]}"
      rel_script="${script#${ROOT_DIR}/}"
      run_desc="$(describe_run "${run_idx}")"

      if [[ "${wait_status}" -eq 0 ]]; then
        echo "[done] ${rel_script} succeeded on attempt ${attempt}"
        append_log "${RUN_LOG}" "DONE parallel run=$((run_idx + 1)) attempt=${attempt} gpus=${chunks[$chunk_idx]} ${run_desc}"
        append_log "${SUCCESS_LOG}" "run=$((run_idx + 1)) attempt=${attempt} gpus=${chunks[$chunk_idx]} ${run_desc}"
      else
        echo "[fail] ${rel_script} failed on attempt ${attempt} with exit code ${wait_status}" >&2
        append_log "${RUN_LOG}" "FAIL parallel run=$((run_idx + 1)) attempt=${attempt} exit_code=${wait_status} gpus=${chunks[$chunk_idx]} ${run_desc}"
        if [[ "${attempt}" -le "${MAX_RETRIES}" ]]; then
          echo "[retry] Re-queueing ${rel_script} for another attempt." >&2
          append_log "${RETRY_LOG}" "run=$((run_idx + 1)) next_attempt=$((attempt + 1)) previous_exit_code=${wait_status} ${run_desc}"
          PENDING_RUNS+=("${run_idx}")
        else
          failures=$((failures + 1))
          append_log "${FAIL_LOG}" "run=$((run_idx + 1)) attempt=${attempt} exit_code=${wait_status} ${run_desc}"
          if [[ "${CONTINUE_ON_ERROR}" -ne 1 ]]; then
            stop_scheduling=1
            echo "A job failed after ${attempt} attempt(s). Waiting for running jobs to finish before stopping." >&2
            append_log "${RUN_LOG}" "STOP parallel scheduling after exhausted retries for run=$((run_idx + 1))."
          fi
        fi
      fi

      schedule_ready_jobs
    done

    if [[ "${stop_scheduling}" -eq 1 && "${queue_head}" -lt "${#PENDING_RUNS[@]}" ]]; then
      skipped_jobs=$(( ${#PENDING_RUNS[@]} - queue_head ))
      echo "Stopped with ${skipped_jobs} queued job(s) not started." >&2
      append_log "${RUN_LOG}" "Stopped with ${skipped_jobs} queued job(s) not started."
    fi
  fi
fi

if [[ "${failures}" -gt 0 ]]; then
  append_log "${RUN_LOG}" "Finished with ${failures} failed job(s)."
  echo "Finished with ${failures} failed job(s)." >&2
  exit 1
fi

append_log "${RUN_LOG}" "All requested scripts finished successfully."
echo "All requested scripts finished successfully."
