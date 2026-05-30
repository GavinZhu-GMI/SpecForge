#!/bin/bash
# GLM-5.1 EAGLE3 OFFLINE training on agentic (Nemotron-SWE-v1) data, with USP
# (Ulysses) context-parallel so 32K coding-agent sessions fit.
#
# Two stages:
#   1) prepare_hidden_states.py — serve the GLM-5.1 target TP-sharded in SGLang,
#      run each session forward ONCE under no_grad, and dump (input_ids,
#      loss_mask, last_hidden_state, aux_hidden_state) to disk. Target-only;
#      SGLang handles 32K natively (it serves 200K), so there is no co-location
#      OOM here. This stage is the disk writer (~1.4 GB/session at ~30K tokens).
#   2) train_eagle3.py --attention-backend usp — train the 1-layer draft from the
#      cached hidden states. USP shards each 32K sequence across the GPUs, which
#      is what makes the EAGLE3 TTT logits [seq x draft_vocab(32000) x 7 steps]
#      fit — the real OOM driver, not attention. Offline build_target_model loads
#      ONLY the lm_head (TargetHead), so no GLM target is resident in stage 2.
#
# USP is OFFLINE-ONLY: train_eagle3 asserts --train-hidden-states-path under usp,
# and online mode has no context-parallel path. Hence offline for long context.
#
# Masking: --chat-template glm-5.1 uses assistant_pattern_type="glm", which
# terminates the assistant loss span on the next turn header
# (<|observation|>/<|user|>/<|assistant|>) — required for agentic data, where a
# <|user|>-only terminator leaks every tool output into the mask.
#
# Usage:   ./examples/run_glm5.1_eagle3_offline.sh
# Stages:  STAGE=1 (generate only) | STAGE=2 (train only) | STAGE=both (default)
# Smoke:   NUM_SAMPLES=16 HS_DIR=/data/hs_smoke EPOCHS=1 STAGE=both ...
#
# Key env knobs (defaults target the real run on 8xH200):
#   NUM_GPUS=8 TP_SIZE=8 MAX_LEN=32768 NUM_SAMPLES=5000 EPOCHS=10
#   SP_ULYSSES=4 SP_RING=1   -> draft_dp = NUM_GPUS/(SP_ULYSSES*SP_RING) = 2,
#                               per-rank seq = ceil(MAX_LEN/4)+ttt = ~8199
#   MEM_FRAC=0.85 (stage-1 target-only, can be high)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

NUM_GPUS=${NUM_GPUS:-8}
TP_SIZE=${TP_SIZE:-8}
MAX_LEN=${MAX_LEN:-32768}
NUM_SAMPLES=${NUM_SAMPLES:-5000}
EPOCHS=${EPOCHS:-10}
SP_ULYSSES=${SP_ULYSSES:-4}
SP_RING=${SP_RING:-1}
MEM_FRAC=${MEM_FRAC:-0.85}
STAGE=${STAGE:-both}

TARGET=${TARGET_MODEL_PATH:-/models/GLM-5.1-FP8}
DATA=${DATA:-/data/nemotron-swe/nemotron_swe_train.jsonl}
HS_DIR=${HS_DIR:-/specforge/hidden_states/glm5.1-nemotron-swe-${MAX_LEN}}
OUT=${OUT:-/specforge/outputs/glm5.1-eagle3-nemotron-swe-offline}
BUILD_PROC=${BUILD_DATASET_NUM_PROC:-32}
# torchrun console script is not always on PATH in the sglang image; the module form always is.
TORCHRUN=${TORCHRUN:-"python -m torch.distributed.run"}

set -x
if [ "$STAGE" = "1" ] || [ "$STAGE" = "both" ]; then
$TORCHRUN --standalone --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/prepare_hidden_states.py \
    --target-model-path $TARGET \
    --trust-remote-code \
    --enable-aux-hidden-states \
    --aux-hidden-states-layers 1,39,75 \
    --data-path $DATA \
    --output-path $HS_DIR \
    --chat-template glm-5.1 \
    --max-length $MAX_LEN \
    --tp-size $TP_SIZE \
    --batch-size 1 \
    --num-samples $NUM_SAMPLES \
    --build-dataset-num-proc $BUILD_PROC \
    --sglang-mem-fraction-static $MEM_FRAC
fi

# Stage-2 attention backend. In OFFLINE stage-2 no target is resident (only the
# 1-layer draft + lm_head + cached hidden states). The backends:
#   flex_attention - memory-efficient FlexAttention, NO flash_attn dep. Does not
#                    materialize the [heads, seq, seq] score matrix, so 32K fits
#                    per rank with full data-parallel (draft_dp=NUM_GPUS). DEFAULT.
#   sdpa           - plain attention; falls back to the math kernel and
#                    materializes the full score matrix -> ~84 GB single alloc at
#                    32K -> OOM. Do not use for long sequences.
#   fa / usp       - require the flash-attn *v2* interface. The sglang cu130 image
#                    ships flash-attn v4, whose API SpecForge does not import, so
#                    these fail with "NoneType is not callable" until a v2-compatible
#                    flash_attn is built. USP (sequence-parallel) is only worth that
#                    once a single rank can no longer hold the target length.
ATTN_BACKEND=${ATTN_BACKEND:-flex_attention}
if [ "$STAGE" = "2" ] || [ "$STAGE" = "both" ]; then
USP_FLAGS=""
if [ "$ATTN_BACKEND" = "usp" ]; then
    USP_FLAGS="--sp-ulysses-size $SP_ULYSSES --sp-ring-size $SP_RING"
fi
$TORCHRUN --standalone --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path $TARGET \
    --trust-remote-code \
    --draft-model-config $ROOT_DIR/configs/glm5.1-eagle3.json \
    --train-data-path $DATA \
    --train-hidden-states-path $HS_DIR \
    --output-dir $OUT \
    --num-epochs $EPOCHS \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length $MAX_LEN \
    --chat-template glm-5.1 \
    --embedding-key model.embed_tokens.weight \
    --lm-head-key lm_head.weight \
    --tp-size 1 \
    --attention-backend $ATTN_BACKEND \
    $USP_FLAGS \
    --build-dataset-num-proc $BUILD_PROC \
    --cache-dir $ROOT_DIR/cache
fi
