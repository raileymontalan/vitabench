# VitaBench: Benchmarking LLM Agents with Versatile Interactive Tasks (AISG Internal Fork)

> **This is an internal AISG fork of [meituan-longcat/vitabench](https://github.com/meituan-longcat/vitabench) for evaluating locally-served models. Results are not submitted to the official leaderboard.**
>
> Default model under evaluation: [`Qwen/Qwen3.6-27B`](https://huggingface.co/Qwen/Qwen3.6-27B), served locally via vLLM. Judge + user-simulator: [`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b), also served locally.

VitaBench evaluates agents on versatile interactive tasks across food delivery, in-store consumption, and online travel (OTA) domains — 66 tools, 100 cross-scenario tasks, 300 single-scenario tasks.

Default eval runs the **100 cross-scenario tasks** (`--domain delivery,instore,ota` selects tasks whose domain spans all three). Per-domain single-scenario splits (100 tasks each) can be run separately with `--domain delivery`, `--domain instore`, or `--domain ota`.

## Quick Start

### 1. Install (one-time, GPU node required)

```bash
sbatch setup_env.slurm
```

Check `logs/setup_<jobid>.out` for completion. Creates `.venv/` with vitabench + vLLM. A GPU node is required — vLLM compiles CUDA kernels on first install.

### 2. Add models to evaluate

Edit `config_vllm.yaml`:

```yaml
eval:
  default_model: Qwen/Qwen3.6-27B   # ← change this to switch default

models:
  Qwen/Qwen3.6-27B:
    tp: 1
    enable_thinking: true
    reasoning_parser: qwen3
    tool_call_parser: qwen3_coder

  # Example: add a new model
  Your/Model-Name:
    tp: 1
    enable_thinking: true
    tool_call_parser: qwen3_coder   # check vLLM docs for your model
    reasoning_parser: qwen3
```

`tp` = tensor parallel size (number of GPUs). The submit script reads this to request the right GPU count automatically.

### 3. Submit

```bash
# Default model from config_vllm.yaml
./submit_vitabench.sh

# Specific model
./submit_vitabench.sh Qwen/Qwen3.6-27B

# Override eval settings
DOMAIN=cross_domain NUM_TRIALS=3 ./submit_vitabench.sh Qwen/Qwen3.6-27B
```

Logs: `logs/vitabench_<jobid>.out`

| Variable          | Default                        | Source                              | Description                                                  |
| ----------------- | ------------------------------ | ----------------------------------- | ------------------------------------------------------------ |
| `MODEL`           | `Qwen/Qwen3.6-27B`             | `config_vllm.yaml` → submit arg     | HuggingFace model ID under evaluation                        |
| `MODEL_TP`        | `1`                            | `config_vllm.yaml models[MODEL].tp` | Tensor parallel size — set per model in config               |
| `JUDGE_MODEL`     | `openai/gpt-oss-120b`          | `config_vllm.yaml eval.judge_model` | Judge + user-simulator model                                 |
| `JUDGE_TP`        | `1`                            | `config_vllm.yaml eval.judge_tp`    | Tensor parallel size for judge                               |
| `DOMAIN`          | `delivery,instore,ota`         | `config_vllm.yaml eval.domain`      | Comma-separated domains; `cross_domain` available separately |
| `NUM_TRIALS`      | `3`                            | `config_vllm.yaml eval.num_trials`  | Independent trials per task (Pass^3 metric)                  |
| `MAX_CONCURRENCY` | `4`                            | `config_vllm.yaml eval.max_concurrency` | Concurrent simulation workers                           |
| `MAX_STEPS`       | `300`                          | `config_vllm.yaml eval.max_steps`   | Max steps per simulation before abandoning                   |
| `SAVE_TO`         | `<model basename>`             | SLURM script                        | Output filename under `data/simulations/`                    |

## Results

Results saved to `data/simulations/<SAVE_TO>` (single file, all domains). A `_summary.txt` is written alongside after each run.

View individual trajectories interactively:

```bash
source .venv/bin/activate
vita view
```

Re-score all models from saved files:

```bash
python score_summary.py data/simulations/
```

Or for a single model file:

```bash
python score_summary.py data/simulations/Qwen3.6-27B
```

### AISG evaluation results (100 cross-scenario tasks, delivery + instore + ota)

Models ordered alphabetically.

| Model | Tasks | Sims | Avg Reward | Pass@1 | Pass^1 | Pass^2 | Pass^3 |
|-------|:-----:|:----:|:----------:|:------:|:------:|:------:|:------:|
| aisingapore/gemma4_e2b_cand1 | 100 | 300 | 0.000 | 0.0% | 0.000 | 0.000 | 0.000 |
| aisingapore/gemma4_e2b_cand2 | 100 | 300 | 0.000 | 0.0% | 0.000 | 0.000 | 0.000 |
| aisingapore/Qwen-SEA-LION-v4.5-27B | 100 | 300 | 0.040 | 4.0% | 0.040 | 0.007 | 0.000 |
| google/gemma-4-31B-it | 100 | 300 | 0.087 | 8.7% | 0.087 | **0.027** | **0.010** |
| google/gemma-4-E2B-it | 100 | 300 | 0.000 | 0.0% | 0.000 | 0.000 | 0.000 |
| google/gemma-4-E4B-it | 100 | 300 | 0.007 | 0.7% | 0.007 | 0.000 | 0.000 |
| Qwen/Qwen3.5-27B | 100 | 300 | **0.093** | **9.2%** | **0.092** | 0.020 | 0.005 |
| Qwen/Qwen3.6-27B | 100 | 300 | 0.067 | 6.7% | 0.067 | 0.013 | 0.000 |

### Thinking-off (nothink) variants

All models have completed 3-trial runs (300 simulations).

| Model | Tasks | Sims | Avg Reward | Pass@1 | Pass^1 | Pass^2 | Pass^3 |
|-------|:-----:|:----:|:----------:|:------:|:------:|:------:|:------:|
| aisingapore/gemma4_e2b_cand1_nothink | 100 | 300 | 0.000 | 0.0% | 0.000 | 0.000 | 0.000 |
| aisingapore/gemma4_e2b_cand2_nothink | 100 | 300 | 0.000 | 0.0% | 0.000 | 0.000 | 0.000 |
| aisingapore/Qwen-SEA-LION-v4.5-27B_nothink | 100 | 300 | 0.067 | 6.7% | 0.067 | 0.017 | 0.010 |
| google/gemma-4-31B-it_nothink | 100 | 300 | **0.107** | **10.7%** | **0.107** | **0.040** | 0.010 |
| google/gemma-4-E2B-it_nothink | 100 | 300 | 0.000 | 0.0% | 0.000 | 0.000 | 0.000 |
| google/gemma-4-E4B-it_nothink | 100 | 300 | 0.000 | 0.0% | 0.000 | 0.000 | 0.000 |
| Qwen/Qwen3.5-27B_nothink | 100 | 300 | 0.103 | 10.3% | 0.103 | 0.033 | **0.020** |
| Qwen/Qwen3.6-27B_nothink | 100 | 300 | 0.093 | 9.3% | 0.093 | 0.030 | 0.010 |

Metric definitions:
- **Avg Reward** — mean task reward across all trials (0–1); vitabench uses a strict sliding-window rubric evaluator, so scores are near-binary (tasks either fully succeed or largely fail)
- **Pass@1** — tasks where ≥1 trial passed (optimistic; upper bound on capability; equals Pass^1 for 1-trial runs)
- **Pass^1** — per-trial pass rate: fraction of all task-trial pairs that passed (differs from Pass@1 when n_trials > 1)
- **Pass^2** — tasks where ≥2 trials passed
- **Pass^3** — tasks where all 3 trials passed (strict reliability; primary metric)

---

<details>
<summary>Upstream documentation</summary>

## Introduction

In this paper, we introduce **VitaBench**, a challenging benchmark that evaluates agents on **v**ersatile **i**nteractive **ta**sks grounded in real-world settings. Drawing from daily applications in food delivery, in-store consumption, and online travel services, VitaBench presents agents with the most complex life-serving simulation environment to date, comprising **66 tools**. Through a framework that eliminates domain-specific policies, we enable flexible composition of these scenarios and tools, yielding **100 cross-scenario tasks (main results) and 300 single-scenario tasks**. Each task is derived from multiple real user requests and requires agents to reason across temporal and spatial dimensions, utilize complex tool sets, proactively clarify ambiguous instructions, and track shifting user intent throughout multi-turn conversations.

Moreover, we propose a rubric-based sliding window evaluator, enabling robust assessment of diverse solution pathways in complex environments and stochastic interactions. Our comprehensive evaluation reveals that even the most advanced models achieve only 32.5% success rate on cross-scenario tasks, and less than 62% success rate on others.

> *The name "Vita" derives from the Latin word for "Life", reflecting our focus on life-serving applications.*

![overall_performance](assets/overall_performance.png)

## Benchmark Details

|                                | Cross-Scenarios | Delivery | In-store |  OTA  |
| :----------------------------- | :-------------: | :------: | :------: | :---: |
| **Databases**                  |                 |          |          |       |
| &nbsp;&nbsp; Service Providers |      1,324      |   409    |   611    | 1,437 |
| &nbsp;&nbsp; Products          |      6,942      |   784    |  3,277   | 9,693 |
| &nbsp;&nbsp; Transactions      |       334       |    48    |    36    |  154  |
| **API Tools**                  |                 |          |          |       |
| &nbsp;&nbsp; Write             |       27        |    4     |    9     |  14   |
| &nbsp;&nbsp; Read              |       33        |    10    |    10    |  19   |
| &nbsp;&nbsp; General           |        6        |    6     |    5     |   5   |
| **Tasks**                      |       100       |   100    |   100    |  100  |

## Re-evaluation

Re-evaluate saved simulations with a different evaluator:

```bash
vita run \
  --re-evaluate-file <simulation file path> \
  --evaluation-type <evaluation type> \
  --evaluator-llm <evaluation model> \
  --save-to <new simulation file path>
```

## Citation

```bibtex
@article{he2025vitabench,
      title={VitaBench: Benchmarking LLM Agents with Versatile Interactive Tasks in Real-world Applications},
      author={He, Wei and Sun, Yueqing and Hao, Hongyan and Hao, Xueyuan and Xia, Zhikang and Gu, Qi and Han, Chengcheng and Zhao, Dengchang and Su, Hui and Zhang, Kefeng and Gao, Man and Su, Xi and Cai, Xiaodong and Cai, Xunliang and Yang, Yu and Zhao, Yunke},
      journal={arXiv preprint arXiv:2509.26490},
      year={2025}
}
```

## Acknowledgement

We adapted part of the [tau2-bench](https://github.com/sierra-research/tau2-bench)'s codebase in building our evaluation framework, and we greatly appreciate their contributions to the agent community.

## License

MIT — see [LICENSE](./LICENSE).

</details>
