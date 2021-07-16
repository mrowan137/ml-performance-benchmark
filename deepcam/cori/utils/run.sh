#!/bin/bash

# NCCL
if [ "$DO_NCCL_DEBUG" == "true" ]
then
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=COLL
    export NCCL_DEBUG_DIR="results/${NODES}_nodes_batchsize_${BATCHSIZE}_j${SLURM_JOB_ID}/nccl_logs"
    mkdir -p $NCCL_DEBUG_DIR
    export NCCL_DEBUG_FILE=${NCCL_DEBUG_DIR}/nccl.${SLURM_JOB_ID}.r${PMIX_RANK}.w${SLURM_NPROCS}
fi

# Where to store results and logfiles
DATA_DIR_PREFIX="/global/cfs/cdirs/nstaff/ai_benchmark/michael/data/cam5_data/All-Hist_small_split_${SLURM_NNODES}"
RUN_TAG="{SLURM_JOB_ID}"
OUTPUT_DIR="results/${NODES}_nodes_batchsize_${BATCHSIZE}_j${RUN_TAG}"
mkdir -p ${OUTPUT_DIR}
if [ $PMIX_RANK -eq 0 ]
then
    touch ${OUTPUT_DIR}/train.out
fi

# Run training
python src/train_hdf5_ddp.py \
     --wireup_method "nccl-slurm" \
     --run_tag ${RUN_TAG} \
     --data_dir_prefix ${DATA_DIR_PREFIX} \
     --output_dir ${OUTPUT_DIR} \
     --max_inter_threads 2 \
     --model_prefix "classifier" \
     --optimizer "LAMB" \
     --start_lr 1e-3 \
     --lr_schedule type="multistep",milestones="8192 16384",decay_rate="0.1" \
     --lr_warmup_steps 0 \
     --lr_warmup_factor 1. \
     --weight_decay 1e-2 \
     --validation_frequency 200 \
     --training_visualization_frequency 0 \
     --validation_visualization_frequency 0 \
     --logging_frequency 10 \
     --save_frequency 400 \
     --max_epochs 5 \
     --amp_opt_level O1 \
     --enable_wandb \
     --wandb_certdir $HOME \
     --local_batch_size $BATCHSIZE |& tee -a ${OUTPUT_DIR}/train.out
