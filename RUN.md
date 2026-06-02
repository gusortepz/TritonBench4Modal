cd /Users/gustavoortiz/Documents/tec/8/reto-dev/TritonBench4Modal
python3 -m venv ~/venvs/modal
source ~/venvs/modal/bin/activate
python -m pip install -r requirements-local.txt

# Run this now if `modal run ...` says "Token not found".
modal setup

# In LM Studio, click Start Server. "Loaded Models: Ready" is not enough;
# Status must say Running / Server running.
curl http://localhost:1234/v1/models

# Then generate locally and evaluate on Modal.
# PyTorch-first smoke run with prompt-8:
modal run modal_app_lmstudio.py::main --limit 5 --model qwen/qwen3.6-35b-a3b --api native \
  --prompt-file prompt-8.txt --output-subdir results/lmstudio_prompt8 \
  2>&1 | tee latest-run.log

# After the run finishes, download Modal results and archive the local JSONL.
bash save_results.sh results/lmstudio_prompt8 experiments/lmstudio_prompt8_$(date +%Y%m%d-%H%M%S)

# Speed-seeking run with prompt-9-speed:
modal run modal_app_lmstudio.py::main --limit 5 --model qwen/qwen3.6-35b-a3b --api native \
  --prompt-file prompt-9-speed.txt --output-subdir results/lmstudio_prompt9_speed \
  2>&1 | tee latest-run.log
bash save_results.sh results/lmstudio_prompt9_speed experiments/lmstudio_prompt9_speed_$(date +%Y%m%d-%H%M%S)

# Better speed signal: include more elementwise-chain candidates.
modal run modal_app_lmstudio.py::main --limit 20 --model qwen/qwen3.6-35b-a3b --api native \
  --prompt-file prompt-9-speed.txt --output-subdir results/lmstudio_prompt9_speed_limit20 \
  2>&1 | tee latest-run.log
bash save_results.sh results/lmstudio_prompt9_speed_limit20 experiments/lmstudio_prompt9_speed_limit20_$(date +%Y%m%d-%H%M%S)

# Ultraspeed diagnostic run. This relaxes edge-case exactness but still computes
# meaningful approximations rather than dummy outputs.
modal run modal_app_lmstudio.py::main --limit 5 --model qwen/qwen3.6-35b-a3b --api native \
  --prompt-file prompt-9-ultraspeed.txt --output-subdir results/lmstudio_prompt9_ultraspeed \
  2>&1 | tee latest-run.log
bash save_results.sh results/lmstudio_prompt9_ultraspeed experiments/lmstudio_prompt9_ultraspeed_$(date +%Y%m%d-%H%M%S)

# Prompt-10 selective speed run. This tries to keep prompt-9 local speed wins
# while recovering execution accuracy on fragile tasks.
RUN_ID=lmstudio_prompt10_selective_$(date +%Y%m%d-%H%M%S)
modal run modal_app_lmstudio.py::main --limit 20 --model qwen/qwen3.6-35b-a3b --api native \
  --prompt-file prompt-10.txt --output-subdir results/${RUN_ID} \
  2>&1 | tee latest-run.log
bash save_results.sh results/${RUN_ID} experiments/${RUN_ID}

# Prompt-10 full sample after the limit-20 smoke looks healthy.
RUN_ID=lmstudio_prompt10_selective_full_$(date +%Y%m%d-%H%M%S)
modal run modal_app_lmstudio.py::main --model qwen/qwen3.6-35b-a3b --api native \
  --prompt-file prompt-10.txt --output-subdir results/${RUN_ID} \
  2>&1 | tee latest-run.log
bash save_results.sh results/${RUN_ID} experiments/${RUN_ID}

# Prompt-11 router smoke. Goal: fix prompt-10's svd/i0 failures and remove
# hot-path torch.compile from fused_mv_logsoftmax_dropout.
RUN_ID=lmstudio_prompt11_router_$(date +%Y%m%d-%H%M%S)
modal run modal_app_lmstudio.py::main --limit 20 --model qwen/qwen3.6-35b-a3b --api native \
  --prompt-file prompt-11-router.txt --output-subdir results/${RUN_ID} \
  2>&1 | tee latest-run.log
bash save_results.sh results/${RUN_ID} experiments/${RUN_ID}

# Prompt-11 router full sample after the limit-20 smoke is healthy.
RUN_ID=lmstudio_prompt11_router_full_$(date +%Y%m%d-%H%M%S)
modal run modal_app_lmstudio.py::main --model qwen/qwen3.6-35b-a3b --api native \
  --prompt-file prompt-11-router.txt --output-subdir results/${RUN_ID} \
  2>&1 | tee latest-run.log
bash save_results.sh results/${RUN_ID} experiments/${RUN_ID}

# Prompt-12 generalized router. This removes most task-specific recipes and
# emphasizes operator-family classification plus optional-parameter safety.
RUN_ID=lmstudio_prompt12_general_router_$(date +%Y%m%d-%H%M%S)
modal run modal_app_lmstudio.py::main --limit 20 --model qwen/qwen3.6-35b-a3b --api native \
  --prompt-file prompt-12-general-router.txt --output-subdir results/${RUN_ID} \
  2>&1 | tee latest-run.log
bash save_results.sh results/${RUN_ID} experiments/${RUN_ID}

# Prompt-12 full sample after the smoke run.
RUN_ID=lmstudio_prompt12_general_router_full_$(date +%Y%m%d-%H%M%S)
modal run modal_app_lmstudio.py::main --model qwen/qwen3.6-35b-a3b --api native \
  --prompt-file prompt-12-general-router.txt --output-subdir results/${RUN_ID} \
  2>&1 | tee latest-run.log
bash save_results.sh results/${RUN_ID} experiments/${RUN_ID}

# Anthropic Claude Opus 4.7 with adaptive thinking at max effort.
# Requires the Modal secret `tritonbench-llm` to contain ANTHROPIC_API_KEY.
modal run modal_app.py::main --limit 5 --provider anthropic --model claude-opus-4-7 \
  --prompt-file prompt-9-speed.txt --anthropic-thinking adaptive --anthropic-effort max \
  --max-tokens 16000 --concurrency 1 --output-subdir results/opus47_max_prompt9_speed \
  2>&1 | tee latest-run.log
bash save_results.sh results/opus47_max_prompt9_speed experiments/opus47_max_prompt9_speed_$(date +%Y%m%d-%H%M%S)

# --- prompt-6 A/B (fusion vs tiled-matmul) -----------------------------------
# Variant A: fusion-focused
modal run modal_app_lmstudio.py::main --limit 5 --model qwen/qwen3.6-35b-a3b --api native \
  --prompt-file prompt-6a.txt --output-subdir results/lmstudio_6a 2>&1 | tee latest-run.log
bash save_results.sh results/lmstudio_6a experiments/lmstudio_6a_$(date +%Y%m%d-%H%M%S)

# Variant B: tiled matmul + epilogue
modal run modal_app_lmstudio.py::main --limit 5 --model qwen/qwen3.6-35b-a3b --api native \
  --prompt-file prompt-6b.txt --output-subdir results/lmstudio_6b 2>&1 | tee latest-run.log
bash save_results.sh results/lmstudio_6b experiments/lmstudio_6b_$(date +%Y%m%d-%H%M%S)
