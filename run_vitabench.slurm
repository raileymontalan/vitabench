#!/bin/bash
#
# vitabench SLURM evaluation script
#
# Runs an agent model + GPT OSS 120B (judge + user-simulator)
# via two local vLLM servers. Evaluates all four domains.
#
# Usage:
#   sbatch run_vitabench.slurm
#   ./submit_vitabench.sh [MODEL]    # preferred: auto-computes GPU count
#
# Override variables at submit time:
#   MODEL=Qwen/Qwen3.6-27B sbatch --gres=gpu:2 run_vitabench.slurm
#
# ── Variable reference ────────────────────────────────────────────────────────
#
# MODEL
#   HuggingFace model ID for the agent under evaluation.
#   Default: eval.default_model in config_vllm.yaml
#
# MODEL_TP
#   Tensor parallel size for the agent model.
#   Default: from config_vllm.yaml models.<MODEL>.tp
#
# MODEL_PORT
#   Port for the agent model vLLM server.
#   Default: auto-assigned free port
#
# JUDGE_MODEL
#   HuggingFace model ID for judge + user-simulator.
#   Default: eval.judge_model in config_vllm.yaml
#
# JUDGE_TP
#   Tensor parallel size for the judge model.
#   Default: eval.judge_tp in config_vllm.yaml
#
# JUDGE_PORT
#   Port for the judge model vLLM server.
#   Default: auto-assigned free port
#
# DOMAIN
#   Domain(s) to evaluate. Comma-separated for multi-domain.
#   Options: delivery, instore, ota, cross_domain, or "delivery,instore,ota"
#   Default: eval.domain in config_vllm.yaml
#
# NUM_TRIALS
#   Number of independent trials per task.
#   Default: eval.num_trials in config_vllm.yaml
#
# MAX_CONCURRENCY
#   Number of parallel simulation workers.
#   Default: eval.max_concurrency in config_vllm.yaml
#
# MAX_STEPS
#   Maximum steps per simulation.
#   Default: eval.max_steps in config_vllm.yaml
#
# SAVE_TO
#   Output file name (saved to data/simulations/<SAVE_TO>). All domains in one file.
#   View results with: vita view
#   Default: <model_basename>
#
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=vitabench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --nodelist=smc-pod-[14,8,5]
#SBATCH --output=logs/vitabench_%j.out
#SBATCH --error=logs/vitabench_%j.out

set -euo pipefail
cd $SLURM_SUBMIT_DIR

module use "${MODULEFILES_PATH}"
module swap cuda/13.0.3

source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"

SITE_PACKAGES=$(.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')
export LD_LIBRARY_PATH="${SITE_PACKAGES}/nvidia/cudnn/lib:${SITE_PACKAGES}/nvidia/cusparselt/lib:${SITE_PACKAGES}/nvidia/nccl/lib:${SITE_PACKAGES}/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_USE_DEEP_GEMM=0

mkdir -p logs

free_port() { python3 -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)"; }

# Resolve MODEL (env override > config default)
MODEL="${MODEL:-}"
if [ -z "${MODEL}" ]; then
    MODEL=$(python3 -c "
import yaml, pathlib
print(yaml.safe_load(pathlib.Path('config_vllm.yaml').read_text()).get('eval', {}).get('default_model', ''))
")
fi
if [ -z "${MODEL}" ]; then
    echo "ERROR: MODEL not set and no default_model in config_vllm.yaml"
    exit 1
fi

# Load model-specific settings
_model_vars=$(python3 - "$MODEL" <<'PYEOF'
import yaml, pathlib, shlex, sys
cfg = yaml.safe_load(pathlib.Path('config_vllm.yaml').read_text())
model = sys.argv[1]
models = cfg.get('models', {})
if model not in models:
    raise SystemExit(f"ERROR: Model '{model}' not found in config_vllm.yaml.\nAvailable: {list(models.keys())}")
mcfg = models[model]
print(f'_MODEL_TP={shlex.quote(str(mcfg.get("tp", 1)))}')
thinking = mcfg.get("enable_thinking", False)
print(f'_MODEL_THINKING={"true" if thinking else "false"}')
print(f'_TOOL_CALL_PARSER={shlex.quote(str(mcfg.get("tool_call_parser", "")))}')
print(f'_REASONING_PARSER={shlex.quote(str(mcfg.get("reasoning_parser", "")))}')
PYEOF
) || exit 1
eval "$_model_vars"

MODEL_TP=${MODEL_TP:-${_MODEL_TP:-"1"}}
MODEL_PORT=${MODEL_PORT:-$(free_port)}
ENABLE_THINKING=${ENABLE_THINKING:-${_MODEL_THINKING:-"false"}}
TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-${_TOOL_CALL_PARSER:-""}}
REASONING_PARSER=${REASONING_PARSER:-${_REASONING_PARSER:-""}}

# Load eval defaults from config
eval "$(python3 - <<'PYEOF'
import yaml, pathlib, shlex
cfg = yaml.safe_load(pathlib.Path('config_vllm.yaml').read_text())
ecfg = cfg.get('eval', {})
print(f'_JUDGE_MODEL={shlex.quote(str(ecfg.get("judge_model", "openai/gpt-oss-120b")))}')
print(f'_JUDGE_TP={shlex.quote(str(ecfg.get("judge_tp", 1)))}')
print(f'_DOMAIN={shlex.quote(str(ecfg.get("domain", "delivery,instore,ota")))}')
print(f'_LANGUAGE={shlex.quote(str(ecfg.get("language", "english")))}')
print(f'_NUM_TRIALS={shlex.quote(str(ecfg.get("num_trials", 1)))}')
print(f'_MAX_CONCURRENCY={shlex.quote(str(ecfg.get("max_concurrency", 4)))}')
print(f'_MAX_STEPS={shlex.quote(str(ecfg.get("max_steps", 300)))}')
PYEOF
)"

JUDGE_MODEL=${JUDGE_MODEL:-${_JUDGE_MODEL:-"openai/gpt-oss-120b"}}
JUDGE_TP=${JUDGE_TP:-${_JUDGE_TP:-"1"}}
JUDGE_PORT=${JUDGE_PORT:-$(free_port)}

DOMAIN=${DOMAIN:-${_DOMAIN:-"delivery,instore,ota"}}
LANGUAGE=${LANGUAGE:-${_LANGUAGE:-"english"}}
NUM_TRIALS=${NUM_TRIALS:-${_NUM_TRIALS:-"1"}}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-${_MAX_CONCURRENCY:-"4"}}
MAX_STEPS=${MAX_STEPS:-${_MAX_STEPS:-"300"}}

MODEL_BASENAME="$(echo ${MODEL} | awk -F/ '{print $NF}')"
SAVE_TO=${SAVE_TO:-"${MODEL_BASENAME}"}

# Judge model name without vendor prefix (e.g., gpt-oss-120b from openai/gpt-oss-120b)
JUDGE_NAME="$(echo "${JUDGE_MODEL}" | sed 's|openai/||')"

# GPU UUID → index mapping
UUIDS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n')
declare -a VISIBLE_DEVICES=()
if [ ! -z "$UUIDS" ]; then
    for UUID in $UUIDS; do
        if [[ $UUID == GPU-* ]] || [[ $UUID == MIG-* ]]; then
            ID=$(nvidia-smi --id=$UUID --query-gpu=index --format=csv,noheader)
            if [ ! -z "$ID" ]; then
                VISIBLE_DEVICES+=($ID)
            else
                VISIBLE_DEVICES+=($UUID)
            fi
        else
            VISIBLE_DEVICES+=($UUID)
        fi
    done
    VISIBLE_DEVICES_STR=$( IFS=$','; echo "${VISIBLE_DEVICES[*]}" )
    export CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICES_STR
fi

if [ ${#VISIBLE_DEVICES[@]} -eq 0 ] && [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -ra VISIBLE_DEVICES <<< "$CUDA_VISIBLE_DEVICES"
fi

# Partition GPUs: first MODEL_TP for agent, next JUDGE_TP for judge
MODEL_GPUS=$(IFS=','; echo "${VISIBLE_DEVICES[*]:0:${MODEL_TP}}")
MODEL_GPUS=${MODEL_GPUS:-0}
JUDGE_GPUS=$(IFS=','; echo "${VISIBLE_DEVICES[*]:${MODEL_TP}:${JUDGE_TP}}")
JUDGE_GPUS=${JUDGE_GPUS:-1}

THINKING_FLAG=""
if [[ "$(echo "$ENABLE_THINKING" | tr '[:upper:]' '[:lower:]')" == "false" ]]; then
    THINKING_FLAG="--chat-template-kwargs {\"enable_thinking\": false}"
fi

REASONING_PARSER_FLAG=""
if [ -n "${REASONING_PARSER}" ]; then
    REASONING_PARSER_FLAG="--reasoning-parser ${REASONING_PARSER}"
fi

# Start agent model vLLM server
echo "Starting agent vLLM server: ${MODEL} on port ${MODEL_PORT} (TP=${MODEL_TP})..."
CUDA_VISIBLE_DEVICES=${MODEL_GPUS} .venv/bin/vllm serve "${MODEL}" \
    --port "${MODEL_PORT}" \
    --tensor-parallel-size "${MODEL_TP}" \
    --served-model-name "${MODEL_BASENAME}" \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser "${TOOL_CALL_PARSER}" \
    ${REASONING_PARSER_FLAG} \
    ${THINKING_FLAG} &
MODEL_PID=$!

echo "Waiting for agent model vLLM server to be ready..."
until curl -sf "http://localhost:${MODEL_PORT}/health" > /dev/null 2>&1 && kill -0 "${MODEL_PID}" 2>/dev/null; do
    if ! kill -0 "${MODEL_PID}" 2>/dev/null; then
        echo "ERROR: Agent vLLM server (PID ${MODEL_PID}) died. Exiting."
        exit 1
    fi
    sleep 5
done
echo "Agent model vLLM server ready."

# Start judge/user vLLM server
echo "Starting judge vLLM server: ${JUDGE_MODEL} on port ${JUDGE_PORT} (TP=${JUDGE_TP})..."
CUDA_VISIBLE_DEVICES=${JUDGE_GPUS} .venv/bin/vllm serve "${JUDGE_MODEL}" \
    --port "${JUDGE_PORT}" \
    --tensor-parallel-size "${JUDGE_TP}" \
    --served-model-name "${JUDGE_NAME}" \
    --dtype bfloat16 \
    --enable-prefix-caching &
JUDGE_PID=$!

echo "Waiting for judge model vLLM server to be ready..."
until curl -sf "http://localhost:${JUDGE_PORT}/health" > /dev/null 2>&1 && kill -0 "${JUDGE_PID}" 2>/dev/null; do
    if ! kill -0 "${JUDGE_PID}" 2>/dev/null; then
        echo "ERROR: Judge vLLM server (PID ${JUDGE_PID}) died. Exiting."
        exit 1
    fi
    sleep 5
done
echo "Judge model vLLM server ready."

# Generate per-job models.yaml with actual ports
JOB_MODELS_YAML="${SLURM_TMPDIR:-/tmp}/vita_models_${SLURM_JOB_ID}.yaml"
cat > "${JOB_MODELS_YAML}" << YAML
default:
  base_url: http://localhost:${MODEL_PORT}/v1/chat/completions
  temperature: 0.0
  max_input_tokens: 32768
  headers:
    Authorization: "Bearer dummy"
    Content-Type: "application/json"

models:
  - name: ${MODEL_BASENAME}
    max_tokens: 8192
    max_input_tokens: 32768

  - name: ${JUDGE_NAME}
    base_url: http://localhost:${JUDGE_PORT}/v1/chat/completions
    max_tokens: 8192
    max_input_tokens: 32768
YAML

export VITA_MODEL_CONFIG_PATH="${JOB_MODELS_YAML}"

echo ""
echo "Model:       ${MODEL} @ http://localhost:${MODEL_PORT}/v1"
echo "Judge:       ${JUDGE_NAME} @ http://localhost:${JUDGE_PORT}/v1"
echo "Domain:      ${DOMAIN}"
echo "Language:    ${LANGUAGE}"
echo "Trials:      ${NUM_TRIALS}"
echo "Concurrency: ${MAX_CONCURRENCY}"
echo "Max steps:   ${MAX_STEPS}"
echo "Save to:     ${SAVE_TO}"
echo ""

THINK_FLAG=""
if [[ "$(echo "$ENABLE_THINKING" | tr '[:upper:]' '[:lower:]')" == "true" ]]; then
    THINK_FLAG="--enable-think"
fi

vita run \
    --domain "${DOMAIN}" \
    --agent-llm "${MODEL_BASENAME}" \
    --user-llm "${JUDGE_NAME}" \
    --evaluator-llm "${JUDGE_NAME}" \
    --language "${LANGUAGE}" \
    --num-trials "${NUM_TRIALS}" \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --max-steps "${MAX_STEPS}" \
    --save-to "${SAVE_TO}" \
    --auto-resume \
    ${THINK_FLAG}

EXIT_CODE=$?

# Shut down vLLM servers
echo "Shutting down vLLM servers..."
kill "${MODEL_PID}" "${JUDGE_PID}" 2>/dev/null || true
wait "${MODEL_PID}" "${JUDGE_PID}" 2>/dev/null || true

rm -f "${JOB_MODELS_YAML}"
unset VITA_MODEL_CONFIG_PATH

# Print and save results summary
echo ""
echo "=== Results Summary ==="
uv run python3 score_summary.py data/simulations/ 2>&1 | tee "data/simulations/score_summary.txt"
echo "Summary saved to data/simulations/score_summary.txt"

exit $EXIT_CODE
