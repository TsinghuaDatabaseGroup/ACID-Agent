# Agentic Transaction: Towards ACID-Compliant Agent Systems

This repository contains the implementation and benchmark integration for **Agentic Transaction: Towards ACID-Compliant Agent Systems**. The system applies database-inspired ACID principles to agent execution: each step is treated as an agentic transaction that is explored, checked, retried when evidence is weak, and committed only when it passes review.

<div align="center">
<img src="docs/img/example.png" width="400px">
</div>

This repository includes:

- **ACID-Agent**, the transaction-oriented agent implementation used in the paper.
- **KramaBench integration**, for running and evaluating end-to-end data-science tasks.
- **Baseline harnesses** for Claude Code and Codex-style agents.
- **Trace and cost logging**, including per-task token usage, runtime, code execution count, and execution traces.

## Repository Structure

```text
.
├── run_kramabench.py              # Main entry point for ACID-Agent and baselines
├── requirements.txt               # Python dependencies for ACID-Agent and KramaBench
├── da_agent/
│   ├── agent/                     # ACID-Agent, Claude Code, Codex wrappers
│   ├── configs/                   # Shared environment/task configuration helpers
│   ├── controllers/               # Execution controllers
│   ├── envs/                      # Docker-backed execution environments
│   ├── images/                    # Docker images for execution sandboxes
│   ├── review/                    # Candidate, retry, and evidence review modules
│   └── utils/                     # LLM, cost metric, observation, runtime config utilities
└── Kramabench/
    ├── benchmark/                 # Evaluation harness and metrics
    ├── workload/                  # Task definitions
    ├── solutions/                 # Reference solutions
    ├── systems/                   # System adapters
    ├── tests/                     # Benchmark tests
    ├── utils/                     # Benchmark utilities
    ├── evaluate.py                # KramaBench evaluator
    └── Kramabench_modification_summary.md
```

## Setup

### 1. Clone

```bash
git clone git@github.com:<your-org-or-user>/ACID-Agent.git
cd ACID-Agent
```

### 2. Create Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install KramaBench and runtime dependencies:

```bash
pip install -e Kramabench

pip install -r requirements.txt
```

### 3. Build Docker Images

ACID-Agent executes task code inside Docker containers.

Build the default KramaBench execution image:

```bash
docker build -t idatascience-kramabench da_agent/images/idatascience-kramabench
```

Optional baseline images:

```bash
docker build -t codex da_agent/images/codex
docker build -t claude-code da_agent/images/claude-code
```

Check Docker access:

```bash
docker ps
docker run --rm idatascience-kramabench whoami
```

## Data

KramaBench task data should be placed under:

```text
data/data-agent/kramabench/datasets/<domain>/input
```

By default, ACID-Agent reads from:

```text
./data/data-agent/kramabench/datasets
```

and writes outputs to:

```text
./data/data-agent/kramabench/output
```

You can override paths with environment variables:

```bash
export KRAMABENCH_DATA_BASE=/path/to/kramabench
export KRAMABENCH_DATASET_DIR=/path/to/kramabench/datasets
export KRAMABENCH_OUTPUT_DIR=/path/to/kramabench/output
export ACID_RUN_STORAGE_BASE=/path/to/acid/runs
```

## Local Scorer Model

ACID-Agent uses a small local language model for probability-based confidence checks. By default, it expects:

```text
./models/Qwen3-0.6B-Base
```

The model is not included in this repository. Download it separately, for example from Hugging Face:

```bash
mkdir -p models

huggingface-cli download Qwen/Qwen3-0.6B-Base \
  --local-dir models/Qwen3-0.6B-Base
```

If your environment uses ModelScope instead:

```bash
mkdir -p models

modelscope download \
  --model Qwen/Qwen3-0.6B-Base \
  --local_dir models/Qwen3-0.6B-Base
```

Then set:

```bash
export ACID_LOCAL_SCORER_MODEL=$(pwd)/models/Qwen3-0.6B-Base
```

By default, this same scorer is used for both probability contrast and evidence surprise. You can override them separately:

```bash
export ACID_PROB_CONTRAST_MODEL=/path/to/probability-contrast-model
export ACID_EVIDENCE_SURPRISE_MODEL=/path/to/evidence-surprise-model
```

## API Configuration

ACID-Agent uses OpenAI-compatible chat APIs.

For OpenAI-compatible providers:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

For DashScope / Alibaba Bailian compatible mode:

```bash
export DASHSCOPE_API_KEY="your-api-key"
export OPENAI_API_KEY="$DASHSCOPE_API_KEY"
export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

## Run ACID-Agent

Run a full domain:

```bash
python3 run_kramabench.py \
  --agent_type acid \
  --domain environment \
  --overwriting \
  --suffix acid-agent-environment-run1 \
  -m qwen3.5-397b-a17b
```

Run selected tasks by zero-based task index:

```bash
python3 run_kramabench.py \
  --agent_type acid \
  --domain environment \
  -i 0,1,2,3,4 \
  --overwriting \
  --suffix acid-agent-environment-part1 \
  -m qwen3.5-397b-a17b
```

Useful options:

```text
--domain              KramaBench domain: archeology, astronomy, biomedical, environment, legal, wildfire
-i, --task_index      Task indices, e.g. all or 0,1,2
-m, --model           Remote LLM model name
--max_steps           Maximum task steps, default 20
--max_memory_length   Maximum compacted history length, default 15
--suffix              Run suffix used in output/result directories
--overwriting         Overwrite existing run outputs
```

## Run Baselines

Claude Code baseline:

```bash
python3 run_kramabench.py \
  --agent_type claude-code \
  --domain environment \
  --overwriting \
  --suffix baseline-claude-code-environment-run1 \
  -m qwen3.5-397b-a17b
```

Codex baseline:

```bash
python3 run_kramabench.py \
  --agent_type codex \
  --domain environment \
  --overwriting \
  --suffix baseline-codex-environment-run1 \
  -m qwen3.5-397b-a17b
```

## Evaluate Results

After a run, locate the generated SUT directory under:

```text
Kramabench/results/<SUT_NAME>/
```

Evaluate with cached system outputs:

```bash
cd Kramabench

export EVAL_LLM_MODEL=qwen-plus

python3 evaluate.py \
  --sut <SUT_NAME> \
  --workload environment \
  --result_directory results \
  --use_system_cache \
  --no_pipeline_eval \
  --num_workers 1
```

The evaluator writes:

```text
Kramabench/results/<SUT_NAME>/<domain>_measures_<timestamp>.csv
```

## Outputs

Each task output directory contains execution records such as:

```text
result.json
acid_trace.jsonl
acid_trace_summary.json
cost_metrics_events.jsonl
```

For Claude/Codex baselines, trace files are named similarly:

```text
claude_trace.jsonl
claude_trace_summary.json
codex_trace.jsonl
codex_trace_summary.json
```

These files are sufficient to inspect the full trajectory, final answer, code executions, runtime, and token usage for each task.

## Benchmark Corrections

This release includes benchmark-level clarifications and corrections used in our experiments, including:

- task wording clarifications,
- reference-solution fixes,
- evaluator updates for very small scientific-notation answers.

See:

```text
Kramabench/Kramabench_modification_summary.md
```

for the full list of changes and rationales.

## Reproducibility Notes

- Docker is required because each task is executed inside an isolated container.
- KramaBench datasets and local scorer weights are not included in this repository; configure them through the paths above before running tasks.
- Experiments use deterministic decoding by default with `temperature=0.0`.
- API pricing is provider-specific. ACID-Agent records token usage and runtime, but does not hard-code dollar costs.

## Citation

If you use this repository, please cite:

```bibtex
@misc{agentictransaction2026,
  title = {Agentic Transaction: Towards ACID-Compliant Agent Systems},
  author = {Zhaoyan Sun and Xiaoxiao Wang and Guoliang Li},
  year = {2026}
}
```

## License

Copyright 2026 Zhaoyan Sun, Xiaoxiao Wang, Guoliang Li.

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE).