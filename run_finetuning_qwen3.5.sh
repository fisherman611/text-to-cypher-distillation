#! /bin/bash
set -euo pipefail

BASE_PATH="."
LOG_DIR="${BASE_PATH}/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "${LOG_DIR}"

LOG_2B="${LOG_DIR}/finetune_qwen3.5_2B_${TIMESTAMP}.log"
LOG_4B="${LOG_DIR}/finetune_qwen3.5_4B_${TIMESTAMP}.log"

run_job() {
  local job_name="$1"
  local script_path="$2"
  local log_path="$3"

  echo "=================================================="
  echo "[$(date '+%F %T')] Start ${job_name}"
  echo "Script: ${script_path}"
  echo "Log   : ${log_path}"
  echo "=================================================="

  bash "${script_path}" 2>&1 | tee "${log_path}"

  echo "[$(date '+%F %T')] Finished ${job_name}"
  echo
}

run_job "Qwen3.5 2B" "scripts/qwen/sft/sft_qwen3.5_2B.sh" "${LOG_2B}"
run_job "Qwen3.5 4B" "scripts/qwen/sft/sft_qwen3.5_4B.sh" "${LOG_4B}"

echo "Done. Logs:"
echo "- ${LOG_2B}"
echo "- ${LOG_4B}"
