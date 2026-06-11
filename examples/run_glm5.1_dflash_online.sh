#!/bin/bash
# GLM-5.1 DFlash (block-diffusion) draft training — ONLINE, SGLang target backend.
#
# WHY ONLINE (not offline like our EAGLE3 driver): scripts/train_dflash.py has NO
# offline / --train-hidden-states-path path. It always co-locates the target via
# --target-model-backend {hf,sglang} and captures hidden states on the fly. The
# capture layers come from the draft config's dflash_config.target_layer_ids, which
# train_dflash feeds to the target generically:
#     target_model.set_capture_layers(draft_model.target_layer_ids)   # train_dflash.py:196
# This IS the GLM capture path — it is model-agnostic, so GlmMoeDsaForCausalLM works
# without a GLM-specific draft class (the draft stays a generic qwen3 backbone fed the
# target's hidden states, exactly like the longcat-flash MoE example).
#
# ⚠️ #1 RISK TO VALIDATE FIRST — co-location OOM. SGLang serves GLM-5.1-FP8 (~90 GB/GPU)
# AND the DFlash draft trains on the SAME 8 GPUs. --sglang-mem-fraction-static carves
# SGLang's share; the rest holds the draft + activations. EAGLE3 SMOKE B found 0.72 is
# the workable band (0.5 → KV-pool init fails, 0.75 → OOM mid-train) at max_length 2048.
# DFlash has no offline fallback, so SMOKE at a short MAX_LEN first, then raise MAX_LEN
# only as far as memory allows (lower MEM_FRAC / batch-size 1 as you go).
#
# Draft = configs/glm5.1-dflash.json (VERIFIED vs zai-org/GLM-5.1-FP8): 8 layers
# (Table-5 net-speedup pick for an MoE target — Qwen3-Coder precedent), block_size 8,
# hidden 6144, target_layer_ids [1,12,22,33,43,54,64,75], mask 154821 (GLM native [MASK]).
#
# Block-8 TRAINING-ARG PITFALL: --loss-decay-gamma must be 4.0 for block_size 8
# (train_dflash.py:81: 7 for 16, 5 for 10, 4 for 8). The shipped qwen/longcat example
# scripts hardcode 7.0 because they are block-16 — do NOT copy that here. Also note
# min_loss_tokens = 2 * --block-size (train_dflash.py:234), so --block-size MUST be 8
# to match the config, or the dataset filter mistunes.
#
# Masking: --chat-template glm-5.1-think uses assistant_pattern_type="glm" (terminates
# the assistant loss span on the next turn header) AND masks the full <think> generation
# — required for GLM's thinking-ON agentic serving distribution. VALIDATE masking on the
# new (bigger) data BEFORE a long run (0 tool-content leaks, full think+answer masked).
#
# Usage:
#   SMOKE:  DATA=/data/.../small.jsonl MAX_LEN=2048 NUM_GPUS=8 MEM_FRAC=0.72 \
#           OUT=/specforge/outputs/glm5.1-dflash-smoke ./examples/run_glm5.1_dflash_online.sh
#   REAL:   DATA=/data/.../glm5_dflash_train.jsonl ./examples/run_glm5.1_dflash_online.sh
#
# MAX_LEN guidance (no offline escape — memory is the ceiling):
#   2048   smoke / OOM-band check
#   3072   the DFlash paper's value; fine for the bulk single-turn code+math data
#   8192   default here — balanced; fits much more agentic context than 3072
#   16384+ agentic-full (EAGLE3 measured GLM R2E-Gym median 16.6K / max 19.5K); only if
#          memory holds — drop MEM_FRAC toward 0.65 and keep batch-size 1.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$ROOT_DIR/cache/compiled_kernels}
export SPECFORGE_DATA_NUM_PROC=${SPECFORGE_DATA_NUM_PROC:-32}
# Let SGLang serve the target at a context length >= our MAX_LEN.
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

NUM_GPUS=${NUM_GPUS:-8}
TP_SIZE=${TP_SIZE:-$NUM_GPUS}
EP_SIZE=${EP_SIZE:-$NUM_GPUS}          # GLM-5.1 is MoE → expert-parallel the target (longcat precedent)
MAX_LEN=${MAX_LEN:-8192}
EPOCHS=${EPOCHS:-6}
BATCH=${BATCH:-1}                      # long context + co-location → keep at 1
LR=${LR:-6e-4}
MEM_FRAC=${MEM_FRAC:-0.72}             # EAGLE3 SMOKE B workable band; lower as MAX_LEN grows
BLOCK_SIZE=${BLOCK_SIZE:-8}            # must equal the config (drives min_loss_tokens=2*block)
GAMMA=${GAMMA:-4.0}                    # 4.0 for block-8 (NOT the examples' 7.0 for block-16)
NUM_ANCHORS=${NUM_ANCHORS:-512}        # paper-confirmed τ lever (Table 9); data augmentation
ATTN_BACKEND=${ATTN_BACKEND:-flex_attention}   # DRAFT backend; proven long-context path on cu130
CHAT_TEMPLATE=${CHAT_TEMPLATE:-glm-5.1-think}  # mask full <think> generation (GLM serves thinking-ON)
BUILD_PROC=${BUILD_DATASET_NUM_PROC:-32}
RESUME=${RESUME:-0}

# GLM target is NSA/DSA — leave SGLang on GLM's native attention backend (do NOT force
# flashinfer; that is for non-NSA targets like longcat). Override only if you must.
SGLANG_ATTN=${SGLANG_ATTN:-}

TARGET=${TARGET_MODEL_PATH:-/models/GLM-5.1-FP8}
DATA=${DATA:-/data/nemotron-swe/glm5_dflash_train.jsonl}
DRAFT_CONFIG=${DRAFT_CONFIG:-$ROOT_DIR/configs/glm5.1-dflash.json}
OUT=${OUT:-/specforge/outputs/glm5.1-dflash-online}

# torchrun console script is not on PATH in the sglang cu130 image; the module form is.
TORCHRUN=${TORCHRUN:-"python -m torch.distributed.run"}

# One checkpoint per epoch is hard to express here (train_dflash counts in steps, and
# the step/epoch count depends on the post-filter dataset size). Default to a frequent
# save so a crash loses little; pair with RESUME=1 to continue from the last checkpoint.
SAVE_INTERVAL=${SAVE_INTERVAL:-500}

RESUME_FLAG=""
if [ "$RESUME" = "1" ]; then RESUME_FLAG="--resume"; fi

SGLANG_ATTN_FLAG=""
if [ -n "$SGLANG_ATTN" ]; then SGLANG_ATTN_FLAG="--sglang-attention-backend $SGLANG_ATTN"; fi

set -x
$TORCHRUN --standalone --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_dflash.py \
    --target-model-path $TARGET \
    --target-model-backend sglang \
    --trust-remote-code \
    --tp-size $TP_SIZE \
    --sglang-ep-size $EP_SIZE \
    --sglang-mem-fraction-static $MEM_FRAC \
    --sglang-context-length $MAX_LEN \
    $SGLANG_ATTN_FLAG \
    --draft-config-path $DRAFT_CONFIG \
    --block-size $BLOCK_SIZE \
    --mask-token-id 154821 \
    --num-anchors $NUM_ANCHORS \
    --loss-decay-gamma $GAMMA \
    --attention-backend $ATTN_BACKEND \
    --embedding-key model.embed_tokens.weight \
    --lm-head-key lm_head.weight \
    --train-data-path $DATA \
    --chat-template $CHAT_TEMPLATE \
    --max-length $MAX_LEN \
    --num-epochs $EPOCHS \
    --batch-size $BATCH \
    --learning-rate $LR \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --output-dir $OUT \
    --save-interval $SAVE_INTERVAL \
    --log-interval 50 \
    $RESUME_FLAG \
    --build-dataset-num-proc $BUILD_PROC \
    --cache-dir $ROOT_DIR/cache \
    --report-to wandb \
    --wandb-offline \
    --wandb-project specforge-glm5.1-dflash \
    --wandb-name glm5.1-dflash-b${BLOCK_SIZE}-${MAX_LEN}
