#!/bin/bash

# Horovod distributed strategy
MULTI_WORKER_FLAGS=""

# Hyperparameters tuned at scale (1024 nodes)
export HOROVOD_HIERARCHICAL_ALLGATHER=0
export HOROVOD_HIERARCHICAL_ALLREDUCE=0
export HOROVOD_GROUPED_ALLREDUCES=1
export HOROVOD_CYCLE_TIME=1
export HOROVOD_FUSION_THRESHOLD=8388608

# Where to store results and logfiles
RES_DIR=profiling_results/nsight/${NODES}_nodes_batchsize_${BATCHSIZE}_j${SLURM_JOB_ID}
if [ $SLURM_PROCID -eq 0 ]
then
    mkdir -p $RES_DIR
fi
PROF_FILE=${RES_DIR}/nsys.${LSB_JOBID}.r${PMIX_RANK}.w${LSB_JOB_NUMPROC}

# Short run -- NSight Systems profiling
# Memory error if collecting mpi trace like so:
# nsys profile -o $PROF_FILE -t cuda,nvtx,mpi,osrt --mpi-impl=openmpi ...
nsys profile -o $PROF_FILE -t cuda \
     python -u ./official/resnet/imagenet_main.py \
     --clean \
     --distribution_strategy=horovod \
     $MULTI_WORKER_FLAGS \
     --data_dir=$DATADIR \
     --stop_threshold=0.9999 \
	 --train_epochs=5 \
     --label_smoothing=0 \
     --batch_size=$BATCHSIZE \
     --enable_lars \
     --fp16_implementation=casting \
     --dtype=fp16 \
     --inter_op_parallelism_threads=4 \
     --intra_op_parallelism_threads=7 \
     --tf_gpu_thread_mode=gpu_private \
     --per_gpu_thread_count=4 \
     --hooks=ExamplesPerSecondHook,LoggingTensorHook \
     --model_dir=$MODELDIR \
     2>&1 > ./log.${SLURM_JOB_ID}
