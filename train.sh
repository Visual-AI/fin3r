#!/usr/bin/env bash
# Run once on each of the four training nodes, with a distinct NODE_RANK.
set -euo pipefail
repo=$(cd "$(dirname "$0")" && pwd)
: "${FIN3R_DATA:?prepared corpus root}" "${FIN3R_LABELS:?MoGe label cache}"
: "${FIN3R_WEIGHTS:?directory containing vggt.pt and moge.pt}" "${FIN3R_RUN:?output directory}"
: "${MASTER_ADDR:?address of the first node}"
rank=${NODE_RANK:-${RANK:?set NODE_RANK to 0, 1, 2 or 3}}
export FIN3R_SCHEDULE="$repo/schedule"
export PYTHONPATH="$repo:$repo/training:$repo/third_party/moge:$repo/third_party/utils3d${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset RANK WORLD_SIZE LOCAL_WORLD_SIZE LOCAL_RANK GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE
exec "${PYTHON:-python}" -m torch.distributed.run --nnodes=4 --nproc_per_node=8 \
    --node_rank="$rank" --master_addr="$MASTER_ADDR" --master_port="${MASTER_PORT:-29685}" "$repo/training/train.py"
