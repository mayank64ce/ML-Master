#!/bin/bash
set -x # Print commands and their arguments as they are executed

AGENT_DIR=./
FHE_CHALLENGE_BASE=../fhe_challenge
MEMORY_INDEX=0

feedback_model=gpt-4o-mini
feedback_temp=0.5
code_temp=0.5

start_cpu=0
CPUS_PER_TASK=36
end_cpu=$((start_cpu + CPUS_PER_TASK - 1))

cd ${AGENT_DIR}
export MEMORY_INDEX
export STEP_LIMIT=500

mkdir -p ${AGENT_DIR}/logs

# API keys are loaded from .env by main_mcts.py via python-dotenv

# ============================================================
# SECTION 1: gpt-4o-mini
# ============================================================

code_model=gpt-4o-mini

# --- black_box ---
# CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} python main_mcts.py \
#   fhe.challenge_dir="${FHE_CHALLENGE_BASE}/black_box/challenge_relu" \
#   agent.steps=10 agent.steerable_reasoning=false agent.check_format=false \
#   agent.search.parallel_search_num=2 \
#   agent.code.model=$code_model agent.code.temp=$code_temp \
#   agent.feedback.model=$feedback_model agent.feedback.temp=$feedback_temp \
#   start_cpu_id="${start_cpu}" cpu_number="${CPUS_PER_TASK}"

# CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} python main_mcts.py \
#   fhe.challenge_dir="${FHE_CHALLENGE_BASE}/black_box/challenge_sigmoid" \
#   agent.steps=10 agent.steerable_reasoning=false agent.check_format=false \
#   agent.search.parallel_search_num=2 \
#   agent.code.model=$code_model agent.code.temp=$code_temp \
#   agent.feedback.model=$feedback_model agent.feedback.temp=$feedback_temp \
#   start_cpu_id="${start_cpu}" cpu_number="${CPUS_PER_TASK}"

# CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} python main_mcts.py \
#   fhe.challenge_dir="${FHE_CHALLENGE_BASE}/black_box/challenge_sign" \
#   agent.steps=10 agent.steerable_reasoning=false agent.check_format=false \
#   agent.search.parallel_search_num=2 \
#   agent.code.model=$code_model agent.code.temp=$code_temp \
#   agent.feedback.model=$feedback_model agent.feedback.temp=$feedback_temp \
#   start_cpu_id="${start_cpu}" cpu_number="${CPUS_PER_TASK}"

# --- white_box/ml_inference ---
# CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} python main_mcts.py \
#   fhe.challenge_dir="${FHE_CHALLENGE_BASE}/white_box/ml_inference/challenge_cifar10" \
#   agent.steps=10 agent.steerable_reasoning=false agent.check_format=false \
#   agent.search.parallel_search_num=2 \
#   agent.code.model=$code_model agent.code.temp=$code_temp \
#   agent.feedback.model=$feedback_model agent.feedback.temp=$feedback_temp \
#   start_cpu_id="${start_cpu}" cpu_number="${CPUS_PER_TASK}"

# CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} python main_mcts.py \
#   fhe.challenge_dir="${FHE_CHALLENGE_BASE}/white_box/ml_inference/challenge_sentiment_analysis" \
#   agent.steps=10 agent.steerable_reasoning=false agent.check_format=false \
#   agent.search.parallel_search_num=2 \
#   agent.code.model=$code_model agent.code.temp=$code_temp \
#   agent.feedback.model=$feedback_model agent.feedback.temp=$feedback_temp \
#   start_cpu_id="${start_cpu}" cpu_number="${CPUS_PER_TASK}"

# CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} python main_mcts.py \
#   fhe.challenge_dir="${FHE_CHALLENGE_BASE}/white_box/ml_inference/challenge_house_prediction" \
#   agent.steps=10 agent.steerable_reasoning=false agent.check_format=false \
#   agent.search.parallel_search_num=2 \
#   agent.code.model=$code_model agent.code.temp=$code_temp \
#   agent.feedback.model=$feedback_model agent.feedback.temp=$feedback_temp \
#   start_cpu_id="${start_cpu}" cpu_number="${CPUS_PER_TASK}"

# CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} python main_mcts.py \
#   fhe.challenge_dir="${FHE_CHALLENGE_BASE}/white_box/ml_inference/challenge_svm" \
#   agent.steps=10 agent.steerable_reasoning=false agent.check_format=false \
#   agent.search.parallel_search_num=2 \
#   agent.code.model=$code_model agent.code.temp=$code_temp \
#   agent.feedback.model=$feedback_model agent.feedback.temp=$feedback_temp \
#   start_cpu_id="${start_cpu}" cpu_number="${CPUS_PER_TASK}"

# --- white_box/openfhe ---
CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} python main_mcts.py \
  fhe.challenge_dir="${FHE_CHALLENGE_BASE}/white_box/openfhe/challenge_gelu" \
  agent.steps=10 agent.steerable_reasoning=false agent.check_format=false \
  agent.search.parallel_search_num=2 \
  agent.code.model=$code_model agent.code.temp=$code_temp \
  agent.feedback.model=$feedback_model agent.feedback.temp=$feedback_temp \
  start_cpu_id="${start_cpu}" cpu_number="${CPUS_PER_TASK}"

CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} python main_mcts.py \
  fhe.challenge_dir="${FHE_CHALLENGE_BASE}/white_box/openfhe/challenge_softmax" \
  agent.steps=10 agent.steerable_reasoning=false agent.check_format=false \
  agent.search.parallel_search_num=2 \
  agent.code.model=$code_model agent.code.temp=$code_temp \
  agent.feedback.model=$feedback_model agent.feedback.temp=$feedback_temp \
  start_cpu_id="${start_cpu}" cpu_number="${CPUS_PER_TASK}"

CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} python main_mcts.py \
  fhe.challenge_dir="${FHE_CHALLENGE_BASE}/white_box/openfhe/challenge_shl" \
  agent.steps=10 agent.steerable_reasoning=false agent.check_format=false \
  agent.search.parallel_search_num=2 \
  agent.code.model=$code_model agent.code.temp=$code_temp \
  agent.feedback.model=$feedback_model agent.feedback.temp=$feedback_temp \
  start_cpu_id="${start_cpu}" cpu_number="${CPUS_PER_TASK}"

CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} python main_mcts.py \
  fhe.challenge_dir="${FHE_CHALLENGE_BASE}/white_box/openfhe/challenge_array_sorting" \
  agent.steps=10 agent.steerable_reasoning=false agent.check_format=false \
  agent.search.parallel_search_num=2 \
  agent.code.model=$code_model agent.code.temp=$code_temp \
  agent.feedback.model=$feedback_model agent.feedback.temp=$feedback_temp \
  start_cpu_id="${start_cpu}" cpu_number="${CPUS_PER_TASK}"

CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} python main_mcts.py \
  fhe.challenge_dir="${FHE_CHALLENGE_BASE}/white_box/openfhe/challenge_lookup_table" \
  agent.steps=10 agent.steerable_reasoning=false agent.check_format=false \
  agent.search.parallel_search_num=2 \
  agent.code.model=$code_model agent.code.temp=$code_temp \
  agent.feedback.model=$feedback_model agent.feedback.temp=$feedback_temp \
  start_cpu_id="${start_cpu}" cpu_number="${CPUS_PER_TASK}"

CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} python main_mcts.py \
  fhe.challenge_dir="${FHE_CHALLENGE_BASE}/white_box/openfhe/challenge_matrix_multiplication" \
  agent.steps=10 agent.steerable_reasoning=false agent.check_format=false \
  agent.search.parallel_search_num=2 \
  agent.code.model=$code_model agent.code.temp=$code_temp \
  agent.feedback.model=$feedback_model agent.feedback.temp=$feedback_temp \
  start_cpu_id="${start_cpu}" cpu_number="${CPUS_PER_TASK}"

CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} python main_mcts.py \
  fhe.challenge_dir="${FHE_CHALLENGE_BASE}/white_box/openfhe/challenge_max" \
  agent.steps=10 agent.steerable_reasoning=false agent.check_format=false \
  agent.search.parallel_search_num=2 \
  agent.code.model=$code_model agent.code.temp=$code_temp \
  agent.feedback.model=$feedback_model agent.feedback.temp=$feedback_temp \
  start_cpu_id="${start_cpu}" cpu_number="${CPUS_PER_TASK}"

CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} python main_mcts.py \
  fhe.challenge_dir="${FHE_CHALLENGE_BASE}/white_box/openfhe/challenge_knn" \
  agent.steps=10 agent.steerable_reasoning=false agent.check_format=false \
  agent.search.parallel_search_num=2 \
  agent.code.model=$code_model agent.code.temp=$code_temp \
  agent.feedback.model=$feedback_model agent.feedback.temp=$feedback_temp \
  start_cpu_id="${start_cpu}" cpu_number="${CPUS_PER_TASK}"

CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} python main_mcts.py \
  fhe.challenge_dir="${FHE_CHALLENGE_BASE}/white_box/openfhe/challenge_parity" \
  agent.steps=10 agent.steerable_reasoning=false agent.check_format=false \
  agent.search.parallel_search_num=2 \
  agent.code.model=$code_model agent.code.temp=$code_temp \
  agent.feedback.model=$feedback_model agent.feedback.temp=$feedback_temp \
  start_cpu_id="${start_cpu}" cpu_number="${CPUS_PER_TASK}"

CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} python main_mcts.py \
  fhe.challenge_dir="${FHE_CHALLENGE_BASE}/white_box/openfhe/challenge_invertible_matrix" \
  agent.steps=10 agent.steerable_reasoning=false agent.check_format=false \
  agent.search.parallel_search_num=2 \
  agent.code.model=$code_model agent.code.temp=$code_temp \
  agent.feedback.model=$feedback_model agent.feedback.temp=$feedback_temp \
  start_cpu_id="${start_cpu}" cpu_number="${CPUS_PER_TASK}"

CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} python main_mcts.py \
  fhe.challenge_dir="${FHE_CHALLENGE_BASE}/white_box/openfhe/challenge_svd" \
  agent.steps=10 agent.steerable_reasoning=false agent.check_format=false \
  agent.search.parallel_search_num=2 \
  agent.code.model=$code_model agent.code.temp=$code_temp \
  agent.feedback.model=$feedback_model agent.feedback.temp=$feedback_temp \
  start_cpu_id="${start_cpu}" cpu_number="${CPUS_PER_TASK}"


# ============================================================
# SECTION 2: gpt-5-mini-2025-08-07
# ============================================================

# code_model=gpt-5-mini-2025-08-07

# # --- black_box ---
# CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} python main_mcts.py \
#   fhe.challenge_dir="${FHE_CHALLENGE_BASE}/black_box/challenge_relu" \
#   agent.steps=10 agent.steerable_reasoning=false agent.check_format=false \
#   agent.search.parallel_search_num=2 \
#   agent.code.model=$code_model agent.code.temp=$code_temp \
#   agent.feedback.model=$feedback_model agent.feedback.temp=$feedback_temp \
#   start_cpu_id="${start_cpu}" cpu_number="${CPUS_PER_TASK}"

# CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} python main_mcts.py \
#   fhe.challenge_dir="${FHE_CHALLENGE_BASE}/black_box/challenge_sigmoid" \
#   agent.steps=10 agent.steerable_reasoning=false agent.check_format=false \
#   agent.search.parallel_search_num=2 \
#   agent.code.model=$code_model agent.code.temp=$code_temp \
#   agent.feedback.model=$feedback_model agent.feedback.temp=$feedback_temp \
#   start_cpu_id="${start_cpu}" cpu_number="${CPUS_PER_TASK}"

# CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} python main_mcts.py \
#   fhe.challenge_dir="${FHE_CHALLENGE_BASE}/black_box/challenge_sign" \
#   agent.steps=10 agent.steerable_reasoning=false agent.check_format=false \
#   agent.search.parallel_search_num=2 \
#   agent.code.model=$code_model agent.code.temp=$code_temp \
#   agent.feedback.model=$feedback_model agent.feedback.temp=$feedback_temp \
#   start_cpu_id="${start_cpu}" cpu_number="${CPUS_PER_TASK}"

# # --- white_box/ml_inference ---
# CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} python main_mcts.py \
#   fhe.challenge_dir="${FHE_CHALLENGE_BASE}/white_box/ml_inference/challenge_cifar10" \
#   agent.steps=10 agent.steerable_reasoning=false agent.check_format=false \
#   agent.search.parallel_search_num=2 \
#   agent.code.model=$code_model agent.code.temp=$code_temp \
#   agent.feedback.model=$feedback_model agent.feedback.temp=$feedback_temp \
#   start_cpu_id="${start_cpu}" cpu_number="${CPUS_PER_TASK}"

# CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} python main_mcts.py \
#   fhe.challenge_dir="${FHE_CHALLENGE_BASE}/white_box/ml_inference/challenge_sentiment_analysis" \
#   agent.steps=10 agent.steerable_reasoning=false agent.check_format=false \
#   agent.search.parallel_search_num=2 \
#   agent.code.model=$code_model agent.code.temp=$code_temp \
#   agent.feedback.model=$feedback_model agent.feedback.temp=$feedback_temp \
#   start_cpu_id="${start_cpu}" cpu_number="${CPUS_PER_TASK}"

# CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} python main_mcts.py \
#   fhe.challenge_dir="${FHE_CHALLENGE_BASE}/white_box/ml_inference/challenge_house_prediction" \
#   agent.steps=10 agent.steerable_reasoning=false agent.check_format=false \
#   agent.search.parallel_search_num=2 \
#   agent.code.model=$code_model agent.code.temp=$code_temp \
#   agent.feedback.model=$feedback_model agent.feedback.temp=$feedback_temp \
#   start_cpu_id="${start_cpu}" cpu_number="${CPUS_PER_TASK}"

# CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} python main_mcts.py \
#   fhe.challenge_dir="${FHE_CHALLENGE_BASE}/white_box/ml_inference/challenge_svm" \
#   agent.steps=10 agent.steerable_reasoning=false agent.check_format=false \
#   agent.search.parallel_search_num=2 \
#   agent.code.model=$code_model agent.code.temp=$code_temp \
#   agent.feedback.model=$feedback_model agent.feedback.temp=$feedback_temp \
#   start_cpu_id="${start_cpu}" cpu_number="${CPUS_PER_TASK}"

# # --- white_box/openfhe ---
# CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} python main_mcts.py \
#   fhe.challenge_dir="${FHE_CHALLENGE_BASE}/white_box/openfhe/challenge_gelu" \
#   agent.steps=10 agent.steerable_reasoning=false agent.check_format=false \
#   agent.search.parallel_search_num=2 \
#   agent.code.model=$code_model agent.code.temp=$code_temp \
#   agent.feedback.model=$feedback_model agent.feedback.temp=$feedback_temp \
#   start_cpu_id="${start_cpu}" cpu_number="${CPUS_PER_TASK}"

# CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} python main_mcts.py \
#   fhe.challenge_dir="${FHE_CHALLENGE_BASE}/white_box/openfhe/challenge_softmax" \
#   agent.steps=10 agent.steerable_reasoning=false agent.check_format=false \
#   agent.search.parallel_search_num=2 \
#   agent.code.model=$code_model agent.code.temp=$code_temp \
#   agent.feedback.model=$feedback_model agent.feedback.temp=$feedback_temp \
#   start_cpu_id="${start_cpu}" cpu_number="${CPUS_PER_TASK}"

# CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} python main_mcts.py \
#   fhe.challenge_dir="${FHE_CHALLENGE_BASE}/white_box/openfhe/challenge_shl" \
#   agent.steps=10 agent.steerable_reasoning=false agent.check_format=false \
#   agent.search.parallel_search_num=2 \
#   agent.code.model=$code_model agent.code.temp=$code_temp \
#   agent.feedback.model=$feedback_model agent.feedback.temp=$feedback_temp \
#   start_cpu_id="${start_cpu}" cpu_number="${CPUS_PER_TASK}"

# CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} python main_mcts.py \
#   fhe.challenge_dir="${FHE_CHALLENGE_BASE}/white_box/openfhe/challenge_array_sorting" \
#   agent.steps=10 agent.steerable_reasoning=false agent.check_format=false \
#   agent.search.parallel_search_num=2 \
#   agent.code.model=$code_model agent.code.temp=$code_temp \
#   agent.feedback.model=$feedback_model agent.feedback.temp=$feedback_temp \
#   start_cpu_id="${start_cpu}" cpu_number="${CPUS_PER_TASK}"

# CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} python main_mcts.py \
#   fhe.challenge_dir="${FHE_CHALLENGE_BASE}/white_box/openfhe/challenge_lookup_table" \
#   agent.steps=10 agent.steerable_reasoning=false agent.check_format=false \
#   agent.search.parallel_search_num=2 \
#   agent.code.model=$code_model agent.code.temp=$code_temp \
#   agent.feedback.model=$feedback_model agent.feedback.temp=$feedback_temp \
#   start_cpu_id="${start_cpu}" cpu_number="${CPUS_PER_TASK}"

# CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} python main_mcts.py \
#   fhe.challenge_dir="${FHE_CHALLENGE_BASE}/white_box/openfhe/challenge_matrix_multiplication" \
#   agent.steps=10 agent.steerable_reasoning=false agent.check_format=false \
#   agent.search.parallel_search_num=2 \
#   agent.code.model=$code_model agent.code.temp=$code_temp \
#   agent.feedback.model=$feedback_model agent.feedback.temp=$feedback_temp \
#   start_cpu_id="${start_cpu}" cpu_number="${CPUS_PER_TASK}"

# CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} python main_mcts.py \
#   fhe.challenge_dir="${FHE_CHALLENGE_BASE}/white_box/openfhe/challenge_max" \
#   agent.steps=10 agent.steerable_reasoning=false agent.check_format=false \
#   agent.search.parallel_search_num=2 \
#   agent.code.model=$code_model agent.code.temp=$code_temp \
#   agent.feedback.model=$feedback_model agent.feedback.temp=$feedback_temp \
#   start_cpu_id="${start_cpu}" cpu_number="${CPUS_PER_TASK}"

# CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} python main_mcts.py \
#   fhe.challenge_dir="${FHE_CHALLENGE_BASE}/white_box/openfhe/challenge_knn" \
#   agent.steps=10 agent.steerable_reasoning=false agent.check_format=false \
#   agent.search.parallel_search_num=2 \
#   agent.code.model=$code_model agent.code.temp=$code_temp \
#   agent.feedback.model=$feedback_model agent.feedback.temp=$feedback_temp \
#   start_cpu_id="${start_cpu}" cpu_number="${CPUS_PER_TASK}"

# CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} python main_mcts.py \
#   fhe.challenge_dir="${FHE_CHALLENGE_BASE}/white_box/openfhe/challenge_parity" \
#   agent.steps=10 agent.steerable_reasoning=false agent.check_format=false \
#   agent.search.parallel_search_num=2 \
#   agent.code.model=$code_model agent.code.temp=$code_temp \
#   agent.feedback.model=$feedback_model agent.feedback.temp=$feedback_temp \
#   start_cpu_id="${start_cpu}" cpu_number="${CPUS_PER_TASK}"

# CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} python main_mcts.py \
#   fhe.challenge_dir="${FHE_CHALLENGE_BASE}/white_box/openfhe/challenge_invertible_matrix" \
#   agent.steps=10 agent.steerable_reasoning=false agent.check_format=false \
#   agent.search.parallel_search_num=2 \
#   agent.code.model=$code_model agent.code.temp=$code_temp \
#   agent.feedback.model=$feedback_model agent.feedback.temp=$feedback_temp \
#   start_cpu_id="${start_cpu}" cpu_number="${CPUS_PER_TASK}"

# CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} python main_mcts.py \
#   fhe.challenge_dir="${FHE_CHALLENGE_BASE}/white_box/openfhe/challenge_svd" \
#   agent.steps=10 agent.steerable_reasoning=false agent.check_format=false \
#   agent.search.parallel_search_num=2 \
#   agent.code.model=$code_model agent.code.temp=$code_temp \
#   agent.feedback.model=$feedback_model agent.feedback.temp=$feedback_temp \
#   start_cpu_id="${start_cpu}" cpu_number="${CPUS_PER_TASK}"

