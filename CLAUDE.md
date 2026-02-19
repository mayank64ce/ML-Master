# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ML-Master is an AI4AI agent that uses Monte Carlo Tree Search (MCTS) to iteratively draft, improve, and debug Python ML solutions for [MLE-Bench](https://github.com/openai/mle-bench) competitions. It uses two LLMs: a **code model** (DeepSeek-R1 or GPT-5) for generation and a **feedback model** (GPT-4o) for evaluating execution outputs.

## Environment Setup

```bash
conda create -n ml-master python=3.12
conda activate ml-master
# Install MLE-Bench first (see its README)
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env: set CODE_API_KEY, CODE_BASE_URL, FEEDBACK_API_KEY, FEEDBACK_BASE_URL
```

## Running

```bash
# 1. Launch grading server (validates submission.csv format against MLE-Bench rules)
bash launch_server.sh

# 2. Run the agent
bash run.sh
```

Key `run.sh` variables to configure before running:
- `EXP_ID` — competition name (must match a folder under `dataset/full_instructions/` and MLE-Bench data)
- `dataset_dir` — path to prepared MLE-Bench dataset
- `data_dir` — path to local data (use `./dataset/house-price-toy` for the toy example)
- `code_model` / `feedback_model` — LLM model names
- `TIME_LIMIT_SECS` — hard timeout

All config keys can also be overridden as CLI args to `python main_mcts.py`:
```bash
python main_mcts.py data_dir=./dataset/house-price-toy desc_file=./dataset/full_instructions/house-price-toy/full_instructions.txt agent.steps=5 agent.steerable_reasoning=false
```

## Architecture

### Entry Point: `main_mcts.py`
Orchestrates parallel MCTS using `ThreadPoolExecutor`. Each worker calls `agent.step()`, which returns the next node to expand. The journal is protected by a lock when appending nodes.

### `agent/mcts_agent.py` — MCTSAgent
Core agent logic. Three operations on nodes:
- `_draft()` — creates a new root-level solution from scratch
- `_improve(parent)` — incrementally improves a working solution
- `_debug(parent)` — fixes a buggy solution

Selection uses UCT (`uct_value()`) with a decaying exploration constant. After execution, `parse_exec_result()` calls the feedback LLM to judge whether the run succeeded and extract the metric value.

### `search/mcts_node.py` — MCTSNode
Extends `Node` with MCTS-specific fields: `visits`, `total_reward`, `uct_value()`, `stage` (root/draft/improve/debug), `lock` (prevents parallel expansion of same draft), `expected_child_count` (thread-safe expansion tracking with a `threading.Lock`).

### `search/journal.py` — Journal
Ordered list of all nodes. Provides `get_best_node()`, `generate_summary()`, and path-filtering utilities.

### `interpreter/interpreter_parallel.py` — Interpreter
Executes generated code in a subprocess with a timeout. Returns `ExecutionResult` containing stdout/stderr, exec time, and exception info.

### `backend/call.py` — LLM Backends
- `r1_query()` — for open-source models (DeepSeek, Qwen); uses `client.completions.create()` for steerable reasoning (pre-fills `<think>` tokens in the prompt)
- `gpt_query()` — for closed-source models; uses `client.chat.completions.create()`

### `utils/config_mcts.yaml` — Default Config
Loaded via OmegaConf; CLI args merge on top. Key sections:
- `agent.code` / `agent.feedback` — model, temp, base_url, api_key (resolved from env vars via `${oc.env:VAR,default}`)
- `agent.search` — MCTS hyperparams (`num_drafts`, `num_improves`, `num_bugs`, `parallel_search_num`)
- `agent.decay` — exploration constant decay schedule (piecewise by default)
- `exec.timeout` — per-step code execution timeout in seconds

## Key Design Decisions

**Steerable reasoning** (`agent.steerable_reasoning`): When `true`, injects a `<think>` prefix into the raw completion prompt to guide DeepSeek/Qwen reasoning. Only works with open-source models that expose `client.completions.create()`. Must be `false` for GPT, Claude, or Gemini models.

**Workspace layout** (created per experiment):
```
workspaces/{exp_name}/
├── input/        # symlinked or copied from data_dir
├── working/      # temporary files from agent code
├── submission/   # submission_{node_id}.csv files
├── best_solution/solution.py
└── best_submission/submission.csv
```

**Grading server** (`utils/server_utils.py`): When `agent.check_format=true`, the agent POSTs submission files to a local server (launched by `launch_server.sh` / `grading_server.py`) to validate format against MLE-Bench ground truth. Disable with `agent.check_format=false` for custom tasks.

**Logs**: `logs/{exp_name}/ml-master.log` (standard) and `ml-master.verbose.log` (includes full prompts/responses).
