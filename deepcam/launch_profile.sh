#!/bin/bash

SCRIPT=$1
output_dir=$2
data_dir_prefix=$3
PROFILE_DIR=$4
run_tag=4

RUN=j_${SLURM_JOB_ID}.s_${SLURM_STEP_ID}
RES_DIR=$5/$RUN

if [ $SLURM_PROCID == '0' ]; then
  mkdir $RES_DIR
fi

PROF_FILE=$PROFILE_DIR/nsys.${SLURM_JOB_ID}.${SLURM_STEP_ID}.r${SLURM_PROCID}.w${SLURM_NPROCS}

#nsys profile -o $PROF_FILE -t cuda,cudnn,nvtx,mpi,osrt python $SCRIPT \
#nsys profile -o $PROF_FILE -t cuda,cudnn,nvtx,mpi,osrt --mpi-impl=openmpi python $SCRIPT \
#nsys profile -o $PROF_FILE -t cuda --mpi-impl=mpich python $SCRIPT \
python $SCRIPT \
     --wireup_method "nccl-slurm" \
     --run_tag ${run_tag} \
     --data_dir_prefix ${data_dir_prefix} \
     --output_dir ${output_dir} \
     --max_inter_threads 2 \
     --model_prefix "classifier" \
     --optimizer "LAMB" \
     --max_epochs 70 \
     --amp_opt_level O1 \
     --local_batch_size $BATCH_SIZE |& tee -a ${output_dir}/train.out


cp $PROF_FILE.qdrep $RES_DIR
