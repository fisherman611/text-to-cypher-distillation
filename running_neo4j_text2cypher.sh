#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
BENCHMARK="${BENCHMARK:-Neo4j_Text2Cypher}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
MODEL_TAG="${MODEL##*/}"
MODEL_SCORE_TAG="${MODEL_TAG//-/_}"
BASE_RESULTS_DIR="${BASE_RESULTS_DIR:-results/Neo4j_Text2Cypher}"

MODE="parallel"
GPU_LIST="0,1"
GPUS_PER_JOB=1
EXPERIMENTS="rkl,distillm,csd,sfkl,fkl,adaptive_srkl_kd0.6_wrel0.5"
MAX_RETRIES=0
CONTINUE_ON_ERROR=0
DRY_RUN=0
LOG_DIR=""

BATCH_SIZE="${BATCH_SIZE:-32}"
TEMPERATURE="${TEMPERATURE:-0.5}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-0}"
DEVICE="${DEVICE:-cuda}"
MAX_LENGTH="${MAX_LENGTH:-3092}"

SUBSETS=(
  bluesky
  buzzoverflow
  companies
  neoflix
  fincen
  gameofthrones
  grandstack
  movies
  network
  northwind
  offshoreleaks
  recommendations
  stackoverflow2
  twitch
  twitter
)

usage() {
  cat <<'EOF'
Usage: ./running_neo4j_text2cypher.sh [options]

Run infer + score for Neo4j_Text2Cypher across multiple checkpoint presets.
Each experiment means: infer full set once, then evaluate all 15 subsets.

Options:
  --mode <parallel|sequential>   Queue jobs by GPU chunks or run one-by-one. Default: parallel
  --gpus <list>                  Comma-separated GPU ids. Default: 0,1
  --gpus-per-job <n>             GPUs used by each experiment. Default: 1
  --experiments <list>           Comma-separated presets: rkl,distillm,csd,sfkl,fkl,adaptive_srkl_kd0.6_wrel0.5
  --max-retries <n>              Retry failed experiments up to n times. Default: 0
  --log-dir <path>               Log directory. Default: ./run_logs/neo4j_text2cypher_<timestamp>
  --continue-on-error            Keep scheduling after retries are exhausted.
  --dry-run                      Print what would run without launching jobs.
  -h, --help                     Show this message.

Environment overrides:
  PYTHON_BIN, BENCHMARK, MODEL, BASE_RESULTS_DIR,
  BATCH_SIZE, TEMPERATURE, TOP_P, TOP_K, DEVICE, MAX_LENGTH

Examples:
  ./running_neo4j_text2cypher.sh
  ./running_neo4j_text2cypher.sh --mode sequential --gpus 0
  ./running_neo4j_text2cypher.sh --gpus 0,1,2,3 --gpus-per-job 2
  ./running_neo4j_text2cypher.sh --experiments distillm
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
    --experiments)
      EXPERIMENTS="$2"
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

experiment_tag() {
  local exp="$1"
  case "${exp}" in
    rkl)
      printf '%s' "distill_rkl"
      ;;
    distillm)
      printf '%s' "distill_distillm"
      ;;
    csd)
      printf '%s' "distill_csd"
      ;;
    sfkl)
      printf '%s' "distill_sfkl"
      ;;
    fkl)
      printf '%s' "distill_fkl"
      ;;
    adaptive_srkl_kd0.6_wrel0.5)
      printf '%s' "distillm_adaptive_srkl_kd0.6_wrel0.5"
      ;;
    *)
      return 1
      ;;
  esac
}

experiment_ckpt() {
  local exp="$1"
  case "${exp}" in
    rkl)
      printf '%s' "https://huggingface.co/fisherman611/text-to-cypher-baselines/tree/main/distillm_0.6B_4B_Cypherbench_rkl/1065"
      ;;
    distillm)
      printf '%s' "https://huggingface.co/fisherman611/text-to-cypher-baselines/distillm_0.6B_4B_Cypherbench_distillm/1065"
      ;;
    csd)
      printf '%s' "https://huggingface.co/fisherman611/text-to-cypher-baselines/distillm_0.6B_4B_Cypherbench_csd/1065"
      ;;
    sfkl)
      printf '%s' "https://huggingface.co/fisherman611/text-to-cypher-baselines/distillm_0.6B_4B_Cypherbench_sfkl/1065"
      ;;
    fkl)
      printf '%s' "https://huggingface.co/fisherman611/text-to-cypher-baselines/distillm_0.6B_4B_Cypherbench_fkl/1065"
      ;;
    adaptive_srkl_kd0.6_wrel0.5)
      printf '%s' "https://huggingface.co/fisherman611/text-to-cypher-models/distillm_new_train_0.6B_4B_adaptive_srkl_kd0.6_wrel0.5/2130"
      ;;
    *)
      return 1
      ;;
  esac
}

IFS=',' read -r -a EXP_RAW <<< "${EXPERIMENTS}"
EXP_LIST=()
for exp in "${EXP_RAW[@]}"; do
  exp_trimmed="$(echo "${exp}" | xargs)"
  [[ -z "${exp_trimmed}" ]] && continue

  if ! experiment_tag "${exp_trimmed}" > /dev/null; then
    echo "Unsupported experiment: ${exp_trimmed}. Supported: rkl, distillm, csd, sfkl, fkl, adaptive_srkl_kd0.6_wrel0.5" >&2
    exit 1
  fi
  EXP_LIST+=("${exp_trimmed}")
done

if [[ "${#EXP_LIST[@]}" -eq 0 ]]; then
  echo "No valid experiments selected." >&2
  exit 1
fi

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
  echo "Warning: $((GPU_COUNT % GPUS_PER_JOB)) GPU(s) will be idle." >&2
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
if [[ -z "${LOG_DIR}" ]]; then
  LOG_DIR="${ROOT_DIR}/run_logs/neo4j_text2cypher_${timestamp}"
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

exp_id() {
  local exp="$1"
  printf '%s' "neo4j__${exp}"
}

run_experiment_once() {
  local exp="$1"
  local gpu_chunk="$2"
  local attempt="$3"
  local exp_tag
  local ckpt_path
  local output_path
  local scores_output_dir
  local exp_name
  local infer_log
  local eval_log

  exp_tag="$(experiment_tag "${exp}")"
  ckpt_path="$(experiment_ckpt "${exp}")"
  output_path="${BASE_RESULTS_DIR}/full_cyphers_result_${MODEL_TAG}_${exp_tag}.json"
  scores_output_dir="${BASE_RESULTS_DIR}/calculated_scores_${MODEL_SCORE_TAG}_${exp_tag}"
  exp_name="$(exp_id "${exp}")"

  infer_log="${JOB_LOG_DIR}/${exp_name}.attempt${attempt}.infer.log"
  eval_log="${JOB_LOG_DIR}/${exp_name}.attempt${attempt}.eval.log"

  mkdir -p "${BASE_RESULTS_DIR}" "${scores_output_dir}"
  : > "${infer_log}"
  : > "${eval_log}"

  echo "[launch] ${exp}"
  echo "         GPUs: ${gpu_chunk} | attempt: ${attempt}"
  echo "         infer_log: ${infer_log}"
  echo "         eval_log : ${eval_log}"
  append_log "${RUN_LOG}" "DISPATCH exp=${exp} attempt=${attempt} gpus=${gpu_chunk}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi

  CUDA_VISIBLE_DEVICES="${gpu_chunk}" "${PYTHON_BIN}" infer.py \
    --benchmark "${BENCHMARK}" \
    --model "${MODEL}" \
    --ckpt_path "${ckpt_path}" \
    --output_path "${output_path}" \
    --batch-size "${BATCH_SIZE}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --top-k "${TOP_K}" \
    --device "${DEVICE}" \
    --max-length "${MAX_LENGTH}" \
    >> "${infer_log}" 2>&1

  for subset in "${SUBSETS[@]}"; do
    "${PYTHON_BIN}" src/calculate_scores_neo4j_text2cypher.py \
      --input "${output_path}" \
      --output_dir "${scores_output_dir}" \
      --subset "${subset}" \
      >> "${eval_log}" 2>&1
  done
}

append_log "${RUN_LOG}" "Started run: mode=${MODE}, gpus=${GPU_LIST}, gpus_per_job=${GPUS_PER_JOB}, max_retries=${MAX_RETRIES}, continue_on_error=${CONTINUE_ON_ERROR}, experiments=${EXPERIMENTS}"
append_log "${RUN_LOG}" "Logs directory: ${LOG_DIR}"
append_log "${RUN_LOG}" "Scheduled experiments: ${#EXP_LIST[@]}"

failures=0

if [[ "${MODE}" == "sequential" ]]; then
  gpu_chunk="${chunks[0]}"
  for exp in "${EXP_LIST[@]}"; do
    attempt=1
    while true; do
      if run_experiment_once "${exp}" "${gpu_chunk}" "${attempt}"; then
        append_log "${RUN_LOG}" "DONE exp=${exp} attempt=${attempt} gpus=${gpu_chunk}"
        append_log "${SUCCESS_LOG}" "exp=${exp} attempt=${attempt} gpus=${gpu_chunk}"
        break
      fi

      append_log "${RUN_LOG}" "FAIL exp=${exp} attempt=${attempt} gpus=${gpu_chunk}"
      if [[ "${attempt}" -le "${MAX_RETRIES}" ]]; then
        attempt=$((attempt + 1))
        append_log "${RETRY_LOG}" "exp=${exp} next_attempt=${attempt}"
        continue
      fi

      failures=$((failures + 1))
      append_log "${FAIL_LOG}" "exp=${exp} attempt=${attempt} gpus=${gpu_chunk}"
      if [[ "${CONTINUE_ON_ERROR}" -ne 1 ]]; then
        append_log "${RUN_LOG}" "Stopped sequential mode after exhausted retries for exp=${exp}"
        echo "Stopping after failure: ${exp}" >&2
        exit 1
      fi
      break
    done
  done
else
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    for idx in "${!EXP_LIST[@]}"; do
      chunk_idx=$((idx % ${#chunks[@]}))
      run_experiment_once "${EXP_LIST[$idx]}" "${chunks[$chunk_idx]}" 1
    done
  else
    declare -a JOB_ATTEMPTS=()
    declare -a PENDING_RUNS=()
    declare -a ACTIVE_PIDS=()
    declare -a CHUNK_BUSY=()
    declare -A PID_TO_RUN_IDX=()
    declare -A PID_TO_CHUNK_IDX=()
    declare -A PID_TO_ATTEMPT=()

    for idx in "${!EXP_LIST[@]}"; do
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
      local exp="${EXP_LIST[$run_idx]}"
      local attempt=$((JOB_ATTEMPTS[run_idx] + 1))
      local pid

      JOB_ATTEMPTS[run_idx]="${attempt}"

      (
        run_experiment_once "${exp}" "${chunks[$chunk_idx]}" "${attempt}"
      ) &
      pid="$!"

      ACTIVE_PIDS+=("${pid}")
      CHUNK_BUSY[chunk_idx]=1
      PID_TO_RUN_IDX["${pid}"]="${run_idx}"
      PID_TO_CHUNK_IDX["${pid}"]="${chunk_idx}"
      PID_TO_ATTEMPT["${pid}"]="${attempt}"
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
      exp="${EXP_LIST[$run_idx]}"

      CHUNK_BUSY[chunk_idx]=0
      remove_active_pid "${finished_pid}"
      unset "PID_TO_RUN_IDX[$finished_pid]"
      unset "PID_TO_CHUNK_IDX[$finished_pid]"
      unset "PID_TO_ATTEMPT[$finished_pid]"

      if [[ "${wait_status}" -eq 0 ]]; then
        echo "[done] ${exp}"
        append_log "${RUN_LOG}" "DONE exp=${exp} attempt=${attempt} gpus=${chunks[$chunk_idx]}"
        append_log "${SUCCESS_LOG}" "exp=${exp} attempt=${attempt} gpus=${chunks[$chunk_idx]}"
      else
        echo "[fail] ${exp} exited with ${wait_status}" >&2
        append_log "${RUN_LOG}" "FAIL exp=${exp} attempt=${attempt} exit_code=${wait_status} gpus=${chunks[$chunk_idx]}"
        if [[ "${attempt}" -le "${MAX_RETRIES}" ]]; then
          next_attempt=$((attempt + 1))
          append_log "${RETRY_LOG}" "exp=${exp} next_attempt=${next_attempt}"
          PENDING_RUNS+=("${run_idx}")
        else
          failures=$((failures + 1))
          append_log "${FAIL_LOG}" "exp=${exp} attempt=${attempt} exit_code=${wait_status} gpus=${chunks[$chunk_idx]}"
          if [[ "${CONTINUE_ON_ERROR}" -ne 1 ]]; then
            stop_scheduling=1
            append_log "${RUN_LOG}" "Stopped scheduling after exhausted retries for exp=${exp}"
          fi
        fi
      fi

      schedule_ready_jobs
    done

    if [[ "${stop_scheduling}" -eq 1 && "${queue_head}" -lt "${#PENDING_RUNS[@]}" ]]; then
      skipped_jobs=$(( ${#PENDING_RUNS[@]} - queue_head ))
      append_log "${RUN_LOG}" "Stopped with ${skipped_jobs} queued experiment(s) not started."
      echo "Stopped with ${skipped_jobs} queued experiment(s) not started." >&2
    fi
  fi
fi

if [[ "${failures}" -gt 0 ]]; then
  append_log "${RUN_LOG}" "Finished with ${failures} failed experiment(s)."
  echo "Finished with ${failures} failed experiment(s)." >&2
  exit 1
fi

append_log "${RUN_LOG}" "All requested experiments finished successfully."
echo "All requested experiments finished successfully."
