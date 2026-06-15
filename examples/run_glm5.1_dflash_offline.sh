#!/bin/bash
# GLM-5.1 DFlash draft training — OFFLINE (precomputed hidden states), two stages.
#
# WHY OFFLINE (vs run_glm5.1_dflash_online.sh): DFlash's online path co-locates the
# GLM-5.1-FP8 target in SGLang on the SAME GPUs as the draft, which (a) risks OOM at the
# long agentic context lengths we need (no fallback) and (b) recomputes the 78-layer MoE
# target forward EVERY epoch (6x wasted). Offline computes the target forward ONCE, caches
# the captured hidden states to disk, then trains the draft alone (no target resident) —
# the same pattern we proved for GLM-5.1 EAGLE3 (run_glm5.1_eagle3_offline.sh).
#
# This relies on a small offline path added to SpecForge (this fork): train_dflash.py
# --train-hidden-states-path + OfflineDFlashDataset + DFlashDataCollatorWithPadding.
# ✅ SMOKE-VALIDATED on master-03 8xH200 (2026-06-11): 16 R2E-Gym samples, MAX_LEN=16384,
# STAGE=both → stage-1 dumped aux_hidden_state [1,seq,8*6144], stage-2 trained 2 steps and
# saved a valid 8-layer (~2.9B) checkpoint that reloads clean. NOTE: MAX_LEN must be large
# enough to include assistant content — 2048 front-truncates these ~16K agentic traces to
# system+user only (loss_sum=0, no anchors). Use MAX_LEN>=16384 for agentic data.
#
# KEY INSIGHT making stage-1 free: DFlash's context feature is the concatenation of the
# target hidden states at target_layer_ids — which is EXACTLY the `aux_hidden_state` tensor
# that the existing scripts/prepare_hidden_states.py dumps when --aux-hidden-states-layers
# is set to those ids. Both the offline prepare and the online SGLang DFlash path capture
# via the SAME set_eagle3_layers_to_capture gate, so the cached tensor is identical to what
# generate_dflash_data would return online. So stage-1 REUSES the EAGLE3 prepare script.
#
# Stages:
#   1) prepare_hidden_states.py — serve GLM-5.1 target-only in SGLang (TP=8, no draft
#      co-located → no OOM, SGLang handles long context natively), forward each session
#      ONCE under no_grad, dump {input_ids, loss_mask, aux_hidden_state=[1,seq,8*H]} to disk.
#   2) train_dflash.py --train-hidden-states-path — train the 8-layer DFlash draft from the
#      cache. No target resident; pure data-parallel draft training (tp-size 1).
#
# Draft = configs/glm5.1-dflash.json: 8 layers, block_size 8, hidden 6144,
# target_layer_ids [1,12,22,33,43,54,64,75]. Block-8 needs --loss-decay-gamma 4.0 (NOT 7).
#
# Usage:   ./examples/run_glm5.1_dflash_offline.sh
# Stages:  STAGE=1 (prepare only) | STAGE=2 (train only) | STAGE=both (default)
# Smoke:   NUM_SAMPLES=16 MAX_LEN=16384 EPOCHS=1 STAGE=both ./examples/run_glm5.1_dflash_offline.sh
#          (MAX_LEN=2048 truncates agentic traces before any assistant turn → 0 loss tokens)

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$ROOT_DIR/cache/compiled_kernels}
export SPECFORGE_DATA_NUM_PROC=${SPECFORGE_DATA_NUM_PROC:-32}
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

NUM_GPUS=${NUM_GPUS:-8}
TP_SIZE=${TP_SIZE:-8}                 # stage-1 target TP; stage-2 draft uses tp-size 1
EP_SIZE=${EP_SIZE:-8}                 # GLM-5.1 is MoE → expert-parallel the target in stage-1
MAX_LEN=${MAX_LEN:-20480}             # offline removes co-location → can afford agentic-full
                                      # (GLM R2E-Gym median 16.6K / max 19.5K; 16384 truncates 97%)
NUM_SAMPLES=${NUM_SAMPLES:-}          # empty = all rows in DATA
EPOCHS=${EPOCHS:-6}
BATCH=${BATCH:-1}
LR=${LR:-6e-4}
MEM_FRAC=${MEM_FRAC:-0.70}            # stage-1 captures 8 hidden states/token; at MAX_LEN=40960 the
                                      # ~4GB on-GPU capture buffer OOMs at 0.85. 0.70 frees ~21GB/GPU
                                      # of absolute headroom (no allocator hacks). Lower if MAX_LEN grows.
BLOCK_SIZE=${BLOCK_SIZE:-8}
GAMMA=${GAMMA:-4.0}                   # 4.0 for block-8 (NOT the examples' 7.0 for block-16)
NUM_ANCHORS=${NUM_ANCHORS:-512}
ATTN_BACKEND=${ATTN_BACKEND:-flex_attention}   # draft backend (proven long-context path on cu130)
CHAT_TEMPLATE=${CHAT_TEMPLATE:-glm-5.1-think}
BUILD_PROC=${BUILD_DATASET_NUM_PROC:-32}
STAGE=${STAGE:-both}
RESUME=${RESUME:-0}

# The 8 DFlash capture layers = the draft config's target_layer_ids. Must match
# configs/glm5.1-dflash.json. Passed to prepare_hidden_states as aux-hidden-states-layers.
AUX_LAYERS=${AUX_LAYERS:-1,12,22,33,43,54,64,75}

TARGET=${TARGET_MODEL_PATH:-/models/GLM-5.1-FP8}
DATA=${DATA:-/data/nemotron-swe/glm5_dflash_train.jsonl}
DRAFT_CONFIG=${DRAFT_CONFIG:-$ROOT_DIR/configs/glm5.1-dflash.json}
HS_DIR=${HS_DIR:-/specforge/hidden_states/glm5.1-dflash-${MAX_LEN}}
OUT=${OUT:-/specforge/outputs/glm5.1-dflash-offline}

TORCHRUN=${TORCHRUN:-"python -m torch.distributed.run"}
NUM_SAMPLES_FLAG=""
if [ -n "$NUM_SAMPLES" ]; then NUM_SAMPLES_FLAG="--num-samples $NUM_SAMPLES"; fi

set -x
# ---- Stage 1: dump GLM target hidden states at the 8 DFlash capture layers ----
if [ "$STAGE" = "1" ] || [ "$STAGE" = "both" ]; then
$TORCHRUN --standalone --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/prepare_hidden_states.py \
    --target-model-path $TARGET \
    --trust-remote-code \
    --enable-aux-hidden-states \
    --aux-hidden-states-layers $AUX_LAYERS \
    --data-path $DATA \
    --output-path $HS_DIR \
    --chat-template $CHAT_TEMPLATE \
    --max-length $MAX_LEN \
    --tp-size $TP_SIZE \
    --sglang-ep-size $EP_SIZE \
    --batch-size 1 \
    $NUM_SAMPLES_FLAG \
    --build-dataset-num-proc $BUILD_PROC \
    --sglang-mem-fraction-static $MEM_FRAC
fi

# ---- Stage 2: train the 8-layer DFlash draft from cached hidden states ----
SAVE_INTERVAL=${SAVE_INTERVAL:-500}
RESUME_FLAG=""
if [ "$RESUME" = "1" ]; then RESUME_FLAG="--resume"; fi
if [ "$STAGE" = "2" ] || [ "$STAGE" = "both" ]; then
$TORCHRUN --standalone --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_dflash.py \
    --target-model-path $TARGET \
    --trust-remote-code \
    --draft-config-path $DRAFT_CONFIG \
    --train-hidden-states-path $HS_DIR \
    --block-size $BLOCK_SIZE \
    --mask-token-id 154821 \
    --num-anchors $NUM_ANCHORS \
    --loss-decay-gamma $GAMMA \
    --attention-backend $ATTN_BACKEND \
    --embedding-key model.embed_tokens.weight \
    --lm-head-key lm_head.weight \
    --chat-template $CHAT_TEMPLATE \
    --max-length $MAX_LEN \
    --num-epochs $EPOCHS \
    --batch-size $BATCH \
    --learning-rate $LR \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --tp-size 1 \
    --output-dir $OUT \
    --save-interval $SAVE_INTERVAL \
    --log-interval 50 \
    $RESUME_FLAG \
    --build-dataset-num-proc $BUILD_PROC \
    --cache-dir $ROOT_DIR/cache \
    --report-to wandb \
    --wandb-offline \
    --wandb-project specforge-glm5.1-dflash \
    --wandb-name glm5.1-dflash-offline-b${BLOCK_SIZE}-${MAX_LEN}
fi
