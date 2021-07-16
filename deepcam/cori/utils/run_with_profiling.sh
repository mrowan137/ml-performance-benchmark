#!/bin/bash

# Where to store results and logfiles
DATA_DIR_PREFIX="/global/cfs/cdirs/nstaff/ai_benchmark/michael/data/cam5_data/All-Hist_small_split_${NODES}"
RUN_TAG="{SLURM_JOB_ID}"
OUTPUT_DIR="results/${NODES}_nodes_batchsize_${BATCHSIZE}_j${RUN_TAG}"
PROFILE_DIR="${OUTPUT_DIR}/profiling_results"
mkdir -p ${OUTPUT_DIR}
mkdir -p ${PROFILE_DIR}
if [ $PMIX_RANK -eq 0 ]
then
    touch ${OUTPUT_DIR}/train.out
fi

PROF_FILE=${profile_dir}/nsys.${LSB_JOBID}.r${PMIX_RANK}.w${LSB_JOB_NUMPROC}
# Run training
nsys profile -o $prof_file -t cuda \
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
