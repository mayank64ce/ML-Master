#!/bin/bash
set -x # Print commands and their arguments as they are executed

AGENT_DIR=./
EXP_ID=house-price-toy
dataset_dir=/path/to/mle-bench   # not used for custom task
MEMORY_INDEX=0

code_model=gpt-5-mini-2025-08-07
code_temp=0.5

feedback_model=gpt-4o-mini
feedback_temp=0.5

start_cpu=0
CPUS_PER_TASK=36
end_cpu=$((start_cpu + CPUS_PER_TASK - 1))

TIME_LIMIT_SECS=600  # 10 minutes

cd ${AGENT_DIR}
export MEMORY_INDEX
format_time() {
  local time_in_sec=$1
  local hours=$((time_in_sec / 3600))
  local minutes=$(((time_in_sec % 3600) / 60))
  local seconds=$((time_in_sec % 60))
  echo "${hours}hrs ${minutes}mins ${seconds}secs"
}
export TIME_LIMIT=$(format_time $TIME_LIMIT_SECS)
export STEP_LIMIT=500

mkdir -p ${AGENT_DIR}/logs

# use the mirror if needed
# export HF_ENDPOINT=https://hf-mirror.com

# API keys are loaded from .env by main_mcts.py via python-dotenv
# base_url and api_key are read from env vars CODE_BASE_URL, CODE_API_KEY,
# FEEDBACK_BASE_URL, FEEDBACK_API_KEY (see utils/config_mcts.yaml)

CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} timeout $TIME_LIMIT_SECS python main_mcts.py \
  dataset_dir="${dataset_dir}" \
  data_dir="./dataset/house-price-toy" \
  desc_file="./dataset/full_instructions/${EXP_ID}/full_instructions.txt" \
  exp_name="${EXP_ID}_test_run" \
  start_cpu_id="${start_cpu}" \
  cpu_number="${CPUS_PER_TASK}" \
  agent.code.model=$code_model \
  agent.code.temp=$code_temp \
  agent.feedback.model=$feedback_model \
  agent.feedback.temp=$feedback_temp \
  agent.steerable_reasoning=false \
  agent.check_format=false \
  agent.steps=20

if [ $? -eq 124 ]; then
  echo "Timed out after $TIME_LIMIT"
fi
