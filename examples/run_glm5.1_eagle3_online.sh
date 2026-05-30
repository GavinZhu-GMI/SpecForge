#!/bin/bash
# GLM-5.1 EAGLE3 online training (non-thinking draft).
#
# Target: zai-org/GLM-5.1-FP8  (GlmMoeDsaForCausalLM, 78 layers, MoE, DSA attn).
# Hidden states are extracted live via the SGLang backend, which exposes the
# EAGLE3 aux-hidden-state capture hook for this model class (verified).
#
# Draft:  configs/glm5.1-eagle3.json  (LlamaForCausalLMEagle3, 1 layer,
#         aux layers [1,39,75], draft_vocab 32000).
# Data:   perfect-blend (the curated mix used by the larger MoE EAGLE3 configs,
#         e.g. deepseek-v3-671b / qwen MoE) — better matched than raw ShareGPT.
#         Template `glm-5.1` anchors loss on </think>.
#
# ---------------------------------------------------------------------------
# Step 0 (once): prepare the dataset
#   python scripts/prepare_data.py --dataset perfectblend \
#       --output-path cache/dataset/perfect-blend.jsonl
#
# Step 0b (RECOMMENDED, biggest acceptance lever): regenerate the answers with
# GLM-5.1 itself so the draft learns the target's true output distribution.
# Point --server-address at your live GLM-5.1 SGLang endpoint:
#   python scripts/regenerate_train_data.py \
#       --model zai-org/GLM-5.1-FP8 \
#       --server-address http://<glm5.1-host>:<port> \
#       --input-file-path  cache/dataset/perfect-blend.jsonl \
#       --output-file-path cache/dataset/perfect-blend-glm5.1-regen.jsonl
#   # then set TRAIN_DATA below to the regen file.
# ---------------------------------------------------------------------------
#
# Usage: ./examples/run_glm5.1_eagle3_online.sh [NUM_GPUS] [TP_SIZE]
#   GLM-5.1 is large — TP_SIZE must be big enough to hold the target for
#   hidden-state extraction (start with 8).

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

NUM_GPUS=${1:-8}
TP_SIZE=${2:-8}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

# Point at a local copy if you have one staged (e.g. /models/GLM-5.1-FP8),
# otherwise the HF id below will be downloaded by the SGLang backend.
TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-zai-org/GLM-5.1-FP8}
# Switch to the regen file once Step 0b is done for a higher acceptance rate.
TRAIN_DATA=${TRAIN_DATA:-$ROOT_DIR/cache/dataset/perfect-blend.jsonl}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path $TARGET_MODEL_PATH \
    --trust-remote-code \
    --draft-model-config $ROOT_DIR/configs/glm5.1-eagle3.json \
    --train-data-path $TRAIN_DATA \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/glm5.1-eagle3-perfect-blend-online \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --chat-template glm-5.1 \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --tp-size $TP_SIZE \
    --target-model-backend sglang \
    --dist-timeout 60 \
    --sglang-mem-fraction-static 0.5  # leave HBM for draft training; target+draft share GPUs
