#!/bin/bash

# distributed strategy
if [ "$STRATEGY" == "multi_worker_mirrored" ]
then 
  MULTI_WORKER_FLAGS="--use_train_and_evaluate --all_reduce_alg=nccl"
else
  MULTI_WORKER_FLAGS=""
fi 
# Hyperparameters tuned at scale (1024 nodes)
export HOROVOD_HIERARCHICAL_ALLGATHER=0
export HOROVOD_HIERARCHICAL_ALLREDUCE=0
export HOROVOD_GROUPED_ALLREDUCES=1
export HOROVOD_CYCLE_TIME=1
export HOROVOD_FUSION_THRESHOLD=8388608

# NCCL
if [ "$DO_NCCL_DEBUG" == "true" ]
then
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=COLL
    export NCCL_DEBUG_DIR="logs/${DATA_MODE}/${NODES}_nodes_batchsize_${BATCHSIZE}/j_${LSB_JOBID}"
    mkdir -p $NCCL_DEBUG_DIR
    export NCCL_DEBUG_FILE="${NCCL_DEBUG_DIR}/nccl.${LSB_JOBID}.r${PMIX_RANK}.w${LSB_JOB_NUMPROC}"
    echo $NCCL_DEBUG_FILE
fi

# Where to store results and logfiles
RUN=j_${LSB_JOBID}
LOG_DIR=logs/${DATA_MODE}/${NODES}_nodes_batchsize_${BATCHSIZE}/$RUN
if [ $PMIX_RANK -eq 0 ]
then
    mkdir -p $LOG_DIR
fi

if [ "$DATA_MODE" == "real" ]
then 
    # Real data benchmarking (convergence testing)
    python -u ./official/resnet/imagenet_main.py \
        --clean \
        --distribution_strategy=$STRATEGY \
        $MULTI_WORKER_FLAGS \
        --data_dir=$DATADIR \
        --stop_threshold=0.99 \
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
        2>&1 > /mnt/bb/$USER/log.${LSB_JOBID}
else
    # Synthetic data benchmarking
    python -u ./official/resnet/imagenet_main.py \
         --clean \
         --distribution_strategy=$STRATEGY \
         $MULTI_WORKER_FLAGS \
         --use_synthetic_data \
	 --stop_threshold=0.99 \
         --train_epochs=5 \
         --batch_size=$BATCHSIZE \
         --enable_lars \
         --fp16_implementation=casting \
         --dtype=fp16 \
         --inter_op_parallelism_threads=4 \
         --intra_op_parallelism_threads=7 \
         --tf_gpu_thread_mode=gpu_private \
         --per_gpu_thread_count=4 \
         --hooks=ExamplesPerSecondHook,LoggingTensorHook \
         --model_dir=$MODELDIR 2>&1 > /mnt/bb/$USER/log.${LSB_JOBID}    
fi

if [ $PMIX_RANK -eq 0 ]
then
    cp /mnt/bb/$USER/log.${LSB_JOBID} .
fi
