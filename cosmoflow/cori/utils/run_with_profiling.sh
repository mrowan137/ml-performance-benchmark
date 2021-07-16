#!/bin/bash

# Where to store results and logfiles
OUTPUT_DIR="results/${NODES}_nodes_batchsize_${BATCHSIZE}_j${SLURM_JOB_ID}"
PROFILE_DIR="${OUTPUT_DIR}/profiling_results"
mkdir -p ${OUTPUT_DIR}
mkdir -p ${PROFILE_DIR}
if [ $PMIX_RANK -eq 0 ]
then
    touch ${OUTPUT_DIR}/train.out
fi

PROF_FILE=${PROFILE_DIR}/nsys.${SLURM_JOB_ID}.r${PMIX_RANK}.w${SLURM_NPROCS}
nsys profile -o ${PROF_FILE} -t cuda \
     python src/train.py -d \
     --batch-size=$BATCHSIZE \
     --n-epochs=5 \
     --output-dir=$OUTPUT_DIR \
     --rank-gpu \
     --amp \
     --stage-dir=/tmp \
     --n-train=$((65536/(8/NODES))) \
     --n-valid=$((8192/(8/NODES))) 2>&1 > ${OUTPUT_DIR}/train.out
