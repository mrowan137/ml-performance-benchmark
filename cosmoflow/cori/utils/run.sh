#!/bin/bash

# NCCL
if [ "$DO_NCCL_DEBUG" == "true" ]
then
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=COLL
    export NCCL_DEBUG_DIR="results/${NODES}_nodes_batchsize_${BATCHSIZE}_j${SLURM_JOB_ID}/nccl_logs"
    mkdir -p $NCCL_DEBUG_DIR
    export NCCL_DEBUG_FILE="${NCCL_DEBUG_DIR}/nccl.${SLURM_JOB_ID}.r${PMIX_RANK}.w${SLURM_NPROCS}"
fi

# Where to store results and logfiles
OUTPUT_DIR="results/${NODES}_nodes_batchsize_${BATCHSIZE}_j${SLURM_JOB_ID}"
mkdir -p ${OUTPUT_DIR}
if [ $PMIX_RANK -eq 0 ]
then
    touch ${OUTPUT_DIR}/train.out
fi

python src/train.py -d \
       --batch-size=$BATCHSIZE \
       --n-epochs=5 \
       --output-dir=$OUTPUT_DIR \
       --rank-gpu \
       --amp \
       --stage-dir=/tmp \
       --n-train=$((65536/(8/NODES))) \
       --n-valid=$((8192/(8/NODES))) 2>&1 > ${OUTPUT_DIR}/train.out
