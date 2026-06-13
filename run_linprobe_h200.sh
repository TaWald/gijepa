#!/bin/bash
set -euo pipefail

# --- workdir + environment -------------------------------------------------
cd /dkfz/cluster/gpu/data/OE0441/t006d/Code/gijepa     # <-- adjust to the path on the cluster
source .venv/bin/activate                              # uv venv created in gijepa/

# --- paths -----------------------------------------------------------------
export IMAGENET_DIR=/dkfz/cluster/gpu/data/common/imagenet/ILSVRC/Data/CLS-LOC
export CKPT=/dkfz/cluster/gpu/checkpoints/OE0441/t006d/generalized_mim/gijepa/vith14.224-bs.2048-ep.300/jepa-ep300.pth.tar
export OUT=/dkfz/cluster/gpu/checkpoints/OE0441/t006d/generalized_mim/gijepa/linprobe_vith14
# --- sanity: print which host + GPUs we actually landed on ------------------
echo "HOST=$(hostname)"; nvidia-smi -L
export NCCL_DEBUG=WARN

# --- I-JEPA ViT-H/14 linear probe ------------------------------------------
# gijepa launches DDP via its own init_distributed + an mp spawn (one process
# per GPU), NOT torchrun. We build the --devices list from GPUS; the script
# spawns the processes itself.
#   eff_batch = PER_GPU_BS * GPUS * ACCUM
#   lr = blr * eff_batch / 256          (computed inside main_linprobe.py)
GPUS=8                 # <-- set to the #GPUs you actually requested
PER_GPU_BS=128         # ViT-H frozen-backbone forward is memory-heavy; lower if OOM
TARGET_EFF=16384
ACCUM=$(( TARGET_EFF / (PER_GPU_BS * GPUS) ))
[ "${ACCUM}" -lt 1 ] && ACCUM=1
echo "GPUS=${GPUS} PER_GPU_BS=${PER_GPU_BS} ACCUM=${ACCUM} eff_batch=$(( PER_GPU_BS * GPUS * ACCUM ))"

# build "cuda:0 cuda:1 ... cuda:(GPUS-1)"
DEVICES=""
for i in $(seq 0 $((GPUS - 1))); do DEVICES="${DEVICES} cuda:${i}"; done

# linear probe: frozen target_encoder + mean-pool + (BN -> Linear) head.
python main_linprobe.py \
    --devices ${DEVICES} \
    --checkpoint "${CKPT}" --which_encoder target_encoder \
    --model_name vit_huge --patch_size 14 --crop_size 224 \
    --nb_classes 1000 --data_path "${IMAGENET_DIR}" \
    --batch_size "${PER_GPU_BS}" --accum_iter "${ACCUM}" \
    --epochs 90 --blr 0.1 --weight_decay 0.0 --warmup_epochs 10 \
    --output_dir "${OUT}" \
    --num_workers 16 --pin_mem
