#!/bin/bash

# NCCL
if [ "$DO_NCCL_DEBUG" == "true" ]
then
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=COLL
    export NCCL_DEBUG_DIR="results/${NODES}_nodes_batchsize_${BATCHSIZE}_j${LSB_JOBID}/nccl_logs"
    mkdir -p $NCCL_DEBUG_DIR
    export NCCL_DEBUG_FILE=${NCCL_DEBUG_DIR}/nccl.${LSB_JOBID}.r${PMIX_RANK}.w${LSB_JOB_NUMPROC}
fi

# Where to store results and logfiles
run_tag="${LSB_JOBID}"
output_dir="results/${NODES}_nodes_batchsize_${BATCHSIZE}_j${run_tag}"

#if [ $PMIX_RANK -eq 0 ]
#then
mkdir -p ${output_dir}
touch ${output_dir}/train.out
#fi

python src/deepCam/train_hdf5_ddp.py \
     --wireup_method "nccl-slurm" \
     --run_tag ${run_tag} \
     --data_dir_prefix ${data_dir_prefix} \
     --output_dir ${output_dir} \
     --max_inter_threads 2 \
     --model_prefix "classifier" \
     --optimizer "LAMB" \
     --max_epochs 5 \
     --amp_opt_level O1 \
     --local_batch_size $BATCHSIZE |& tee -a ${output_dir}/train.out
