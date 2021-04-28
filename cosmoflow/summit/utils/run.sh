#!/bin/bash

# NCCL
if [ "$DO_NCCL_DEBUG" == "true" ]
then
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=COLL
    export NCCL_DEBUG_DIR="results/${NODES}_nodes_batchsize_${batch_size}_j${LSB_JOBID}/nccl_logs"
    mkdir -p $NCCL_DEBUG_DIR
    export NCCL_DEBUG_FILE="${NCCL_DEBUG_DIR}/nccl.${LSB_JOBID}.r${PMIX_RANK}.w${LSB_JOB_NUMPROC}"
fi

# Where to store results and logfiles
run_tag="${LSB_JOBID}"
export output_dir="results/${NODES}_nodes_batchsize_${batch_size}_j${run_tag}"

#if [ $PMIX_RANK -eq 0 ]
#then
mkdir -p ${output_dir}
touch ${output_dir}/train.out
#fi

python src/train.py -d \
       --batch-size=$batch_size \
       --data-dir=$data_dir \
       --output-dir=$output_dir \
       --n-epochs=5 \
       --rank-gpu $CONFIG |& tee -a ${output_dir}/train.out
