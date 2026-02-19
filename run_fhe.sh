#!/bin/bash
set -x # Print commands and their arguments as they are executed

AGENT_DIR=./
FHE_CHALLENGE_DIR=../fhe_challenge/black_box/challenge_relu
MEMORY_INDEX=0

code_model=gpt-5-mini-2025-08-07
code_temp=0.5

feedback_model=gpt-4o-mini
feedback_temp=0.5

start_cpu=0
CPUS_PER_TASK=36
end_cpu=$((start_cpu + CPUS_PER_TASK - 1))

cd ${AGENT_DIR}
export MEMORY_INDEX
export STEP_LIMIT=500

mkdir -p ${AGENT_DIR}/logs

# use the mirror if needed
# export HF_ENDPOINT=https://hf-mirror.com

# API keys are loaded from .env by main_mcts.py via python-dotenv
# base_url and api_key are read from env vars CODE_BASE_URL, CODE_API_KEY,
# FEEDBACK_BASE_URL, FEEDBACK_API_KEY (see utils/config_mcts.yaml)

CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} python main_mcts.py \
  fhe.challenge_dir="${FHE_CHALLENGE_DIR}" \
  agent.steps=10 \
  agent.steerable_reasoning=false \
  agent.check_format=false \
  agent.search.parallel_search_num=2 \
  agent.code.model=$code_model \
  agent.code.temp=$code_temp \
  agent.feedback.model=$feedback_model \
  agent.feedback.temp=$feedback_temp \
  start_cpu_id="${start_cpu}" \
  cpu_number="${CPUS_PER_TASK}"

