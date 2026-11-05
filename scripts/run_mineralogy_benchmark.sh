#!/bin/bash
# Serve each text model on one GPU, score data/cls/test.csv, then write the report.
#
# Usage (from the cemcat repo root):
#   bash scripts/run_mineralogy_benchmark.sh
#   bash scripts/run_mineralogy_benchmark.sh qwen25-7b gemma3-4b ministral-8b
#   bash scripts/run_mineralogy_benchmark.sh llamat-3-chat llamat-2-chat

set -uo pipefail

# Remember the submit environment, then each model prepends its own env lib.
BASE_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
unset LD_LIBRARY_PATH

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PORT="${VLLM_PORT:-8000}"
LOG_DIR="${REPO_ROOT}/results/mineralogy_cls/logs"
mkdir -p "${LOG_DIR}"

if [[ $# -gt 0 ]]; then
  MODELS=("$@")
else
  MODELS=(qwen25-0.5b gemma3-270m ministral-3b)
fi

# Emit and the final report only need the standard library. Prefer an explicit
# interpreter, then the qwen env, then python3.
if [[ -z "${REGISTRY_PY:-}" ]]; then
  if [[ -n "${CONDA_ENV_ROOT:-}" && -x "${CONDA_ENV_ROOT}/qwen_vllm/bin/python" ]]; then
    REGISTRY_PY="${CONDA_ENV_ROOT}/qwen_vllm/bin/python"
  elif [[ -n "${SCRATCH_ENV_ROOT:-}" && -x "${SCRATCH_ENV_ROOT}/qwen_vllm/bin/python" ]]; then
    REGISTRY_PY="${SCRATCH_ENV_ROOT}/qwen_vllm/bin/python"
  else
    REGISTRY_PY="python3"
  fi
fi

stop_server() {
  local key="$1"
  local pid_file="${LOG_DIR}/serve_${key}.pid"
  if [[ -f "${pid_file}" ]]; then
    local pid
    pid="$(cat "${pid_file}")"
    if kill -0 "${pid}" 2>/dev/null; then
      echo "Stopping server PID ${pid}"
      kill "${pid}" 2>/dev/null || true
      for _ in $(seq 1 60); do
        kill -0 "${pid}" 2>/dev/null || break
        sleep 2
      done
      kill -0 "${pid}" 2>/dev/null && kill -9 "${pid}" 2>/dev/null || true
    fi
    rm -f "${pid_file}"
  fi
  for _ in $(seq 1 30); do
    curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1 || break
    sleep 2
  done
}

wait_for_ready() {
  local pid="$1"
  local log_file="$2"
  local deadline=$(( SECONDS + ${VLLM_STARTUP_TIMEOUT:-1800} ))
  while (( SECONDS < deadline )); do
    if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "Server ready on port ${PORT}"
      return 0
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "Server exited during startup. Tail of ${log_file}:" >&2
      tail -n 40 "${log_file}" >&2
      return 1
    fi
    sleep 5
  done
  echo "Timed out waiting for server on port ${PORT}" >&2
  return 1
}

if [[ -z "${HF_HUB_OFFLINE:-}" ]] && ! curl -sf --max-time 8 https://huggingface.co/api/models/gpt2 >/dev/null 2>&1; then
  echo "No hub connectivity detected; serving in offline mode from the shared cache."
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

if [[ -z "${HF_HOME:-}" ]]; then
  if [[ -n "${CACHE_ROOT:-}" ]]; then
    export HF_HOME="${CACHE_ROOT}/huggingface"
  elif [[ -n "${XDG_CACHE_HOME:-}" ]]; then
    export HF_HOME="${XDG_CACHE_HOME}/huggingface"
  else
    export HF_HOME="${HOME}/.cache/huggingface"
  fi
fi
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
mkdir -p "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

wait_for_weights() {
  local model_id="$1"
  local repo="models--${model_id//\//--}"
  local hub="${HF_HUB_CACHE}/${repo}"
  local deadline=$(( SECONDS + ${WEIGHT_WAIT_TIMEOUT:-3600} ))
  while (( SECONDS < deadline )); do
    if [[ -d "${hub}" ]] && find "${hub}" -name '*.incomplete' -print -quit | grep -q .; then
      echo "Waiting for ${model_id} download to finish in ${hub}"
      sleep 20
      continue
    fi
    if [[ -d "${hub}" ]] && find "${hub}" -name '*.safetensors' -print -quit | grep -q .; then
      return 0
    fi
    echo "Waiting for ${model_id} weights in ${hub}"
    sleep 20
  done
  echo "Timed out waiting for ${model_id} weights" >&2
  return 1
}

FAILED=()
for model in "${MODELS[@]}"; do
  echo
  echo "=============================================================="
  echo "MODEL: ${model}"
  echo "=============================================================="
  stop_server "${model}"

  EXTRA_SERVE_ARGS=()
  MAX_MODEL_LEN=16384
  MAX_TOKENS=16
  OVERWRITE=0
  eval "$("${REGISTRY_PY}" "${SCRIPT_DIR}/benchmark_mineralogy_cls.py" --model "${model}" --emit shell)"
  if [[ ! -x "${VLLM_BIN}" ]]; then
    echo "vLLM binary missing for ${model}: ${VLLM_BIN}" >&2
    FAILED+=("${model}:env")
    continue
  fi

  SERVE_LOG="${LOG_DIR}/serve_${model}.log"
  echo "Checkpoint: ${MODEL_ID}"
  echo "Env:        ${ENV_PREFIX}"
  echo "Log:        ${SERVE_LOG}"

  if ! wait_for_weights "${MODEL_ID}"; then
    FAILED+=("${model}:weights")
    continue
  fi

  export LD_LIBRARY_PATH="${ENV_PREFIX}/lib${BASE_LD_LIBRARY_PATH:+:${BASE_LD_LIBRARY_PATH}}"

  "${VLLM_BIN}" serve "${MODEL_ID}" \
    --port "${PORT}" \
    --tensor-parallel-size 1 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.90 \
    "${EXTRA_SERVE_ARGS[@]}" \
    > "${SERVE_LOG}" 2>&1 &
  SERVER_PID=$!
  echo "${SERVER_PID}" > "${LOG_DIR}/serve_${model}.pid"

  if ! wait_for_ready "${SERVER_PID}" "${SERVE_LOG}"; then
    FAILED+=("${model}:serve")
    stop_server "${model}"
    continue
  fi

  SCORE_ARGS=(--max-tokens "${MAX_TOKENS}")
  if [[ "${OVERWRITE}" == "1" ]]; then
    SCORE_ARGS+=(--overwrite)
  fi
  if ! "${PYTHON_BIN}" "${SCRIPT_DIR}/benchmark_mineralogy_cls.py" \
      --model "${model}" \
      --base-url "http://localhost:${PORT}/v1" \
      --output-dir "${REPO_ROOT}/results/mineralogy_cls" \
      "${SCORE_ARGS[@]}"; then
    FAILED+=("${model}:infer")
  fi
  stop_server "${model}"
done

"${REGISTRY_PY}" "${SCRIPT_DIR}/benchmark_mineralogy_cls.py" \
  --report-only \
  --output-dir "${REPO_ROOT}/results/mineralogy_cls" || FAILED+=("report")

echo
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "Failed: ${FAILED[*]}"
  exit 1
fi
echo "Benchmark finished. Report: ${REPO_ROOT}/results/mineralogy_cls/report.md"
