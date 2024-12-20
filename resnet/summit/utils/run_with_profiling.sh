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


# Where to store profiling results and logfiles
RUN=j_${LSB_JOBID}
RES_DIR=profiling_results/${DATA_MODE}/${NODES}_nodes_batchsize_${BATCHSIZE}/$RUN
LOG_DIR=logs/${DATA_MODE}/${NODES}_nodes_batchsize_${BATCHSIZE}/$RUN
if [ $PMIX_RANK -eq 0 ]
then
    mkdir -p $RES_DIR
    mkdir -p $LOG_DIR
fi
PROF_FILE=$DATADIR/nsys.${LSB_JOBID}.r${PMIX_RANK}.w${LSB_JOB_NUMPROC}

if [ "$DATA_MODE" == "real" ]
then 
    # Real data benchmarking (convergence testing)
    # Memory error if collecting mpi trace
    #nsys profile -o $PROF_FILE -t cuda,nvtx,mpi,osrt --mpi-impl=openmpi \
    nsys profile -o $PROF_FILE -t cuda \
	python -u ./official/resnet/imagenet_main.py \
        --clean \
        --distribution_strategy=$STRATEGY \
        $MULTI_WORKER_FLAGS \
        --data_dir=$DATADIR \
        --label_smoothing=0 \
        --train_epochs=5 \
        --stop_threshold=0.9999 \
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
else
    # Synthetic data benchmarking (images/sec with batch size = 128)
    nsys profile -o $PROF_FILE -t cuda \
         python -u ./official/resnet/imagenet_main.py \
         --clean \
         --distribution_strategy=$STRATEGY \
         $MULTI_WORKER_FLAGS \
         --use_synthetic_data \
         --train_epochs=5 \
	 --stop_threshold=0.99 \
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

cp ${PROF_FILE}.qdrep ${RES_DIR}

if [ $PMIX_RANK -eq 0 ]
then
    cp /mnt/bb/$USER/log.${LSB_JOBID} .
fi

